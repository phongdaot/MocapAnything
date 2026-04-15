import torch
import torch.nn as nn


# =========================================================
# RoPE helpers for per-joint temporal attention
# =========================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1)
    return x_rot.flatten(-2)


def apply_rope_1d_perjoint(q: torch.Tensor, k: torch.Tensor):
    """
    q, k: [BJ, H, T, Dh]
    """
    BJ, H, T, Dh = q.shape
    assert Dh % 2 == 0, f"RoPE head dim must be even, got {Dh}"

    device = q.device
    dtype = q.dtype
    half_dim = Dh // 2

    pos = torch.arange(T, device=device, dtype=torch.float32)
    freq_seq = torch.arange(half_dim, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

    freqs = torch.outer(pos, inv_freq)  # [T, half_dim]
    cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=dtype)  # [T, Dh]
    sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=dtype)

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,Dh]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


# =========================================================
# Per-joint Temporal Attention with RoPE + Window
# =========================================================
class TemporalPerJointMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nheads=8,
        dropout=0.1,
        temporal_window=2,
        use_temporal_bias=True,
    ):
        super().__init__()
        assert d_model % nheads == 0
        self.d_model = d_model
        self.nheads = nheads
        self.att_size = d_model // nheads
        self.scale = self.att_size ** -0.5
        assert self.att_size % 2 == 0, f"RoPE requires even head dim, got {self.att_size}"

        self.temporal_window = temporal_window
        self.use_temporal_bias = use_temporal_bias

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_temporal_bias:
            self.temporal_bias = nn.Embedding(2 * temporal_window + 1, nheads)

    def _expand_mask_to_btj(self, mask, B, T, J, device):
        if mask is None:
            return torch.ones(B, T, J, device=device, dtype=torch.bool)
        if mask.dim() == 2:
            return mask.unsqueeze(1).expand(-1, T, -1).bool()
        elif mask.dim() == 3:
            return mask.bool()
        else:
            raise ValueError(f"mask dim should be 2 or 3, got {mask.dim()}")

    def _build_window_mask(self, T, device):
        time_ids = torch.arange(T, device=device)
        delta_t = time_ids[:, None] - time_ids[None, :]
        visible = delta_t.abs() <= self.temporal_window
        return visible, delta_t

    def forward(
        self,
        q,   # [B,T,J,D]
        k,   # [B,T,J,D]
        v,   # [B,T,J,D]
        mask=None,  # [B,J] or [B,T,J]
    ):
        B, T, J, D = q.shape
        device = q.device
        orig_size = q.size()

        token_mask_btj = self._expand_mask_to_btj(mask, B, T, J, device)

        # [B,T,J,D] -> [B,J,T,D] -> [BJ,T,D]
        q = q.permute(0, 2, 1, 3).contiguous().view(B * J, T, D)
        k = k.permute(0, 2, 1, 3).contiguous().view(B * J, T, D)
        v = v.permute(0, 2, 1, 3).contiguous().view(B * J, T, D)

        # [B,T,J] -> [B,J,T] -> [BJ,T]
        token_mask = token_mask_btj.permute(0, 2, 1).contiguous().view(B * J, T)

        q = self.linear_q(q).view(B * J, T, self.nheads, self.att_size).transpose(1, 2)  # [BJ,H,T,d]
        k = self.linear_k(k).view(B * J, T, self.nheads, self.att_size).transpose(1, 2)
        v = self.linear_v(v).view(B * J, T, self.nheads, self.att_size).transpose(1, 2)

        q, k = apply_rope_1d_perjoint(q, k)

        attn_score = torch.matmul(q, k.transpose(2, 3))  # [BJ,H,T,T]

        window_vis, delta_t = self._build_window_mask(T, device)
        if self.use_temporal_bias:
            dt_clamped = delta_t.clamp(-self.temporal_window, self.temporal_window)
            dt_index = dt_clamped + self.temporal_window
            t_bias = self.temporal_bias(dt_index)         # [T,T,H]
            t_bias = t_bias.permute(2, 0, 1).unsqueeze(0) # [1,H,T,T]
            attn_score = attn_score + t_bias

        attn_score = attn_score * self.scale

        attn_score = attn_score.masked_fill(
            ~window_vis.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        attn_score = attn_score.masked_fill(
            ~token_mask[:, None, None, :], float("-inf")
        )

        invalid_rows = torch.isinf(attn_score).all(dim=-1, keepdim=True)
        attn_score = torch.where(invalid_rows, torch.zeros_like(attn_score), attn_score)

        attn = torch.softmax(attn_score.float(), dim=-1).to(attn_score.dtype)
        attn = attn * token_mask[:, None, :, None].float()
        attn = attn * window_vis.unsqueeze(0).unsqueeze(0).float()
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [BJ,H,T,d]
        out = out.transpose(1, 2).contiguous().view(B * J, T, D)
        out = self.output_layer(out)
        out = self.dropout(out)

        out = out * token_mask.unsqueeze(-1).float()
        out = out.view(B, J, T, D).permute(0, 2, 1, 3).contiguous()

        assert out.size() == orig_size
        return out


# =========================================================
# Per-joint Temporal Transformer Block
# =========================================================
class TemporalPerJointTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        nheads=8,
        dropout=0.1,
        ff_mult=4,
        temporal_window=2,
        use_temporal_bias=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TemporalPerJointMultiHeadAttention(
            d_model=dim,
            nheads=nheads,
            dropout=dropout,
            temporal_window=temporal_window,
            use_temporal_bias=use_temporal_bias,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, joint_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask=joint_mask)
        x = residual + x

        if joint_mask is not None:
            if joint_mask.dim() == 2:
                mask4d = joint_mask.unsqueeze(1).unsqueeze(-1).float()
            else:
                mask4d = joint_mask.unsqueeze(-1).float()
            x = x * mask4d

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        if joint_mask is not None:
            x = x * mask4d

        return x

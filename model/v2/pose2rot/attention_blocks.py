import torch.nn as nn
import torch

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
# helper: conditioning
# =========================================================
class FiLMCondition(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
        )

    def forward(self, x, cond):
        """
        x:    [..., D]
        cond: same shape as x
        """
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + 0.1 * scale) + shift


# =========================================================
# Graph Attention
# =========================================================
class GraphMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nheads=4,
        dropout=0.1,
        max_path_len=5,
        value_emb=False,
        use_tree_mask=False,
    ):
        super().__init__()
        assert d_model % nheads == 0
        self.nheads = nheads
        self.att_size = d_model // nheads
        self.scale = self.att_size ** -0.5

        self.linear_q = nn.Linear(d_model, nheads * self.att_size)
        self.linear_k = nn.Linear(d_model, nheads * self.att_size)
        self.linear_v = nn.Linear(d_model, nheads * self.att_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(nheads * self.att_size, d_model)

        self.max_path_len = max_path_len
        self.value_emb_flag = value_emb
        self.use_tree_mask = use_tree_mask

        self.topology_key_emb = nn.Embedding(max_path_len + 1, d_model)
        self.edge_key_emb = nn.Embedding(6, d_model)
        self.topology_query_emb = nn.Embedding(max_path_len + 1, d_model)
        self.edge_query_emb = nn.Embedding(6, d_model)

        if value_emb:
            self.topology_value_emb = nn.Embedding(max_path_len + 1, d_model)
            self.edge_value_emb = nn.Embedding(6, d_model)

    def forward(
        self,
        q,               # [B,J,D]
        k,               # [B,J,D]
        v,               # [B,J,D]
        distance,        # [B,J,J] or [B,F,J,J]
        edge_attr,       # [B,J,J] or [B,F,J,J]
        mask=None,       # [B,J] or [B,F,J]
        tree_mask=None,  # [B,J,J] or [B,F,J,J]
    ):
        if distance.dim() == 4:
            B, F_, J, _ = distance.shape
            distance = distance.reshape(B * F_, J, J)
            edge_attr = edge_attr.reshape(B * F_, J, J)
            if mask is not None:
                mask = mask.reshape(B * F_, J)
            if tree_mask is not None:
                tree_mask = tree_mask.reshape(B * F_, J, J)

        orig_q_size = q.size()
        batch_size = q.size(0)
        d_k = self.att_size
        d_v = self.att_size

        q = self.linear_q(q).view(batch_size, -1, self.nheads, d_k).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, -1, self.nheads, d_k).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, -1, self.nheads, d_v).transpose(1, 2)

        seq_len = v.shape[2]
        num_hop_types = self.max_path_len + 1
        num_edge_types = 6

        query_hop_emb = self.topology_query_emb.weight.view(1, num_hop_types, self.nheads, d_k).transpose(1, 2)
        query_edge_emb = self.edge_query_emb.weight.view(1, num_edge_types, self.nheads, d_k).transpose(1, 2)
        key_hop_emb = self.topology_key_emb.weight.view(1, num_hop_types, self.nheads, d_k).transpose(1, 2)
        key_edge_emb = self.edge_key_emb.weight.view(1, num_edge_types, self.nheads, d_k).transpose(1, 2)

        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = torch.gather(query_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1))

        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = torch.gather(query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1))

        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = torch.gather(key_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1))

        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = torch.gather(key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1))

        spatial_bias = query_hop + key_hop
        edge_bias = query_edge + key_edge

        attn_score = torch.matmul(q, k.transpose(2, 3)) + spatial_bias + edge_bias
        attn_score = attn_score * self.scale

        if tree_mask is not None and self.use_tree_mask:
            attn_score = attn_score.masked_fill(~tree_mask[:, None, :, :], float("-inf"))

        if mask is not None:
            mask_k = mask[:, None, None, :]
            attn_score = attn_score.masked_fill(~mask_k, float("-inf"))

        invalid_rows = torch.isinf(attn_score).all(dim=-1, keepdim=True)
        attn_score = torch.where(invalid_rows, torch.zeros_like(attn_score), attn_score)

        attn = torch.softmax(attn_score.float(), dim=-1).to(attn_score.dtype)

        if mask is not None:
            mask_q = mask[:, None, :, None].float()
            attn = attn * mask_q

        if tree_mask is not None and self.use_tree_mask:
            attn = attn * tree_mask[:, None, :, :].float()

        attn = self.dropout(attn)

        if self.value_emb_flag:
            value_hop_emb = self.topology_value_emb.weight.view(1, num_hop_types, self.nheads, d_k).transpose(1, 2)
            value_edge_emb = self.edge_value_emb.weight.view(1, num_edge_types, self.nheads, d_k).transpose(1, 2)

            value_hop_att = torch.zeros((batch_size, self.nheads, seq_len, num_hop_types), device=q.device, dtype=attn.dtype)
            value_hop_att = torch.scatter_add(
                value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1), attn
            )

            value_edge_att = torch.zeros((batch_size, self.nheads, seq_len, num_edge_types), device=q.device, dtype=attn.dtype)
            value_edge_att = torch.scatter_add(
                value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1), attn
            )

        x = torch.matmul(attn, v)
        if self.value_emb_flag:
            x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.nheads * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x

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

# =========================================================
# Per-joint Memory Cross Attention
# =========================================================
class JointMemoryCrossAttention(nn.Module):
    def __init__(self, d_model, nheads=8, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_feat,   # [B,T,J,D]
        memory_feat,  # [B,N,J,D]
        joint_mask,   # [B,J]
    ):
        B, T, J, D = query_feat.shape
        N = memory_feat.shape[1]

        q = query_feat.permute(0, 2, 1, 3).contiguous().reshape(B * J, T, D)
        m = memory_feat.permute(0, 2, 1, 3).contiguous().reshape(B * J, N, D)

        q = self.norm_q(q)
        m = self.norm_kv(m)

        joint_valid = joint_mask.reshape(B * J)
        key_padding_mask = (~joint_valid[:, None]).expand(-1, N)

        out, _ = self.attn(
            query=q,
            key=m,
            value=m,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.out_proj(out)
        out = self.dropout(out)

        out = out.reshape(B, J, T, D).permute(0, 2, 1, 3).contiguous()
        out = out * joint_mask.unsqueeze(1).unsqueeze(-1).float()
        return out

# =========================================================
# Rot Decoder
# =========================================================
class RotDecoderBlock(nn.Module):
    def __init__(
        self,
        q_dim=256,
        num_heads=8,
        dropout=0.1,
        temporal_window=2,
        use_tree_mask=False,
        use_rest_film=True,
        use_cross_attn=True,
    ):
        super().__init__()
        self.use_rest_film = use_rest_film
        self.use_cross_attn = use_cross_attn

        self.film = FiLMCondition(q_dim) if use_rest_film else None
        self.temporal = TemporalPerJointTransformerBlock(
            dim=q_dim,
            nheads=num_heads,
            dropout=dropout,
            ff_mult=4,
            temporal_window=temporal_window,
            use_temporal_bias=True,
        )
        self.graph_norm = nn.LayerNorm(q_dim)
        self.graph = GraphMultiHeadAttention(
            q_dim,
            num_heads,
            dropout=dropout,
            use_tree_mask=use_tree_mask,
        )
        if self.use_cross_attn:
            self.cross = JointMemoryCrossAttention(
                d_model=q_dim,
                nheads=num_heads,
                dropout=dropout,
            )
        self.ffn_norm = nn.LayerNorm(q_dim)
        self.ffn = nn.Sequential(
            nn.Linear(q_dim, q_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(q_dim * 4, q_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,               # [B,T,J,D]
        rest_t,          # [B,T,J,D]
        memory_feat,     # [B,N,J,D]
        joint_mask,      # [B,J]
        ancestor_mask,      # [B,J,J]
        graph_hop,       # [B,J,J]
        graph_edge,      # [B,J,J]
    ):
        B, T, J, D = x.shape
        joint_mask_bt = joint_mask.unsqueeze(1).expand(-1, T, -1)

        if self.film is not None:
            x = self.film(x, rest_t)
            x = x * joint_mask_bt.unsqueeze(-1).float()

        x = self.temporal(x, joint_mask=joint_mask_bt)

        x2 = self.graph_norm(x).reshape(B * T, J, D)
        jm = joint_mask_bt.reshape(B * T, J)
        gm = ancestor_mask.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)
        gh = graph_hop.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)
        ge = graph_edge.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)

        x2 = self.graph(
            x2, x2, x2,
            gh, ge,
            mask=jm,
            tree_mask=gm,
        )
        x = x + x2.reshape(B, T, J, D)
        x = x * joint_mask_bt.unsqueeze(-1).float()

        if self.use_cross_attn:
            x = x + self.cross(x, memory_feat, joint_mask)
            x = x * joint_mask_bt.unsqueeze(-1).float()

        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        x = x * joint_mask_bt.unsqueeze(-1).float()
        return x


import torch
import torch.nn.functional as Func


# 1D rope
# copy from https://github.com/tairov/llama2.py/blob/6a102a5f71e50f868cd2d577724cefc7275b833c/model.py#L37C1-L79C50
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def apply_rotary_emb_1d_on_frame(
        x: torch.Tensor,  # [B, F, S, H, D]
        freqs_cos: torch.Tensor,  # [F, D]
        freqs_sin: torch.Tensor  # [F, D]
) -> torch.Tensor:
    """
    Apply RoPE (rotary positional encoding) along the frame dimension F.

    x: input tensor with shape [B, F, S, H, D]
    freqs_cos, freqs_sin: precomputed cosine and sine frequencies with shape [F, D]
    """
    # Check dimensions
    B, F, S, H, D = x.shape
    assert D % 2 == 0, "D must be even for complex rotary"

    # Split real and imaginary parts -> [B, F, S, H, D/2]
    x_r, x_i = x.float().reshape(B, F, S, H, D // 2, 2).unbind(-1)  # both [B, F, S, H, D/2]

    # reshape freqs for broadcasting: [1, F, 1, 1, D/2]
    freqs_cos = freqs_cos.view(1, F, 1, 1, D // 2)
    freqs_sin = freqs_sin.view(1, F, 1, 1, D // 2)

    # Apply RoPE rotation
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # Merge back to [B, F, S, H, D]
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(-2)

    return x_out.type_as(x)  # keep original precision



class SlidingWindowAttnProcessor:
    def __init__(self, use_frame_rope=True):
        self.use_frame_rope = use_frame_rope

    def __call__(
            self,
            attn,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            temb=None,
            att_frames: int = None,
            att_slidwindow: int = None,
            joint_mask=None,
    ):
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        BF, joint_num, dim = query.shape
        heads = attn.heads
        head_dim = dim // heads
        query = query.view(BF, joint_num, heads, head_dim)
        key = key.view(BF, -1, heads, head_dim)
        value = value.view(BF, -1, heads, head_dim)

        if att_frames is not None:
            F = att_frames
            BF = query.shape[0]
            B = BF // F
            N = query.shape[1]
            query = query.view(B, F, N, heads, head_dim)
            key = key.view(B, F, -1, heads, head_dim)
            value = value.view(B, F, -1, heads, head_dim)

            if self.use_frame_rope:
                freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, F)
                freqs_cos = freqs_cos.to(query.device)
                freqs_sin = freqs_sin.to(query.device)
                query = apply_rotary_emb_1d_on_frame(query, freqs_cos, freqs_sin)
                key = apply_rotary_emb_1d_on_frame(key, freqs_cos, freqs_sin)
            if att_slidwindow is not None:
                win = att_slidwindow
                pad = win // 2
                key_pad = Func.pad(key, (0, 0, 0, 0, 0, 0, pad, pad), value=0)
                value_pad = Func.pad(value, (0, 0, 0, 0, 0, 0, pad, pad), value=0)
                idx = torch.arange(F, device=key.device)
                win_idx = idx.unsqueeze(1) + torch.arange(win, device=key.device)
                key_win = key_pad[:, win_idx, :, :, :]
                value_win = value_pad[:, win_idx, :, :, :]
                key = key_win.reshape(B * F, win * key.shape[2], heads, head_dim)
                value = value_win.reshape(B * F, win * value.shape[2], heads, head_dim)
                query = query.reshape(B * F, N, heads, head_dim)
            else:
                key = key.reshape(B * F, -1, heads, head_dim)
                value = value.reshape(B * F, -1, heads, head_dim)
                query = query.reshape(B * F, N, heads, head_dim)
            BF, joint_num = B * F, N
        # else: 默认全局
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)

        # === ✅ 加 joint_mask ===
        if joint_mask is not None and encoder_hidden_states is hidden_states:
            B, seq_len, J = joint_mask.shape  # [B, F, J]
            heads = attn.heads

            # === sliding window 模式 ===
            if att_slidwindow is not None:
                win = att_slidwindow
                pad = win // 2
                # === (1) pad joint_mask（保持与 key/value pad 对齐）
                joint_mask_pad = Func.pad(joint_mask, (0, 0, pad, pad), value=False)  # [B, F+2*pad, J]
                # === (2) 构造滑窗索引，与 key/value 一致
                idx = torch.arange(seq_len, device=joint_mask.device)
                win_idx = idx.unsqueeze(1) + torch.arange(win, device=joint_mask.device)  # win_idx: [F, win]
                # === (3) 提取窗口内的 joint_mask → [B, F, win, J]
                joint_mask_win = joint_mask_pad[:, win_idx, :]
                # === (4) 对窗口内 joint_mask
                joint_mask_win = joint_mask_win.reshape(B * seq_len, 1, 1, win * J)  # 改到这里了, 主要是和上面对应
                # === (6) 扩展到多头 & flatten
                mask = joint_mask_win.repeat(1, heads, J, 1)
                # === (7) 应用 mask
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

            # === 普通（非滑窗）模式 ===
            else:
                # [B, F, J] → [B*F, heads, J, J]
                base_mask = joint_mask.view(B, seq_len, 1, 1, J).expand(B, seq_len, heads, J, J)
                mask = base_mask.reshape(B * seq_len, heads, J, J)
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        invalid_rows = torch.isinf(attn_scores).all(dim=-1, keepdim=True)
        attn_scores = torch.where(invalid_rows, torch.zeros_like(attn_scores), attn_scores)

        attn_probs = attn_scores.softmax(dim=-1)
        
        if joint_mask is not None and encoder_hidden_states is hidden_states:
            if att_frames is not None:
                B, seq_len, J = joint_mask.shape
                mask_q = joint_mask.view(B * seq_len, 1, J, 1).float()
                attn_probs = attn_probs * mask_q
                
        attn_out = torch.matmul(attn_probs, value)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(BF, joint_num, heads * head_dim)
        attn_out = attn.to_out(attn_out)
        return attn_out



# processor
class SimpleAttnProcessor:
    """标准 attention（无 sliding window/RoPE），支持 joint_mask"""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 joint_mask=None, **kwargs):
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        value = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)

        B, N, D = query.shape
        heads = attn.heads
        head_dim = D // heads

        query = query.view(B, heads, N, head_dim)
        key = key.view(B, heads, -1, head_dim)
        value = value.view(B, heads, -1, head_dim)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)

        # === 加入 joint_mask ===
        if joint_mask is not None and encoder_hidden_states is None:
            # 仅在 self-attn 时使用（N=J）
            attn_scores = attn_scores.masked_fill(
                ~joint_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        invalid_rows = torch.isinf(attn_scores).all(dim=-1, keepdim=True)
        attn_scores = torch.where(invalid_rows, torch.zeros_like(attn_scores), attn_scores)

        attn_probs = attn_scores.softmax(dim=-1)

        if joint_mask is not None and encoder_hidden_states is None:
            attn_probs = attn_probs * joint_mask.unsqueeze(1).unsqueeze(-1).float()
        
        out = torch.matmul(attn_probs, value)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return attn.to_out(out)

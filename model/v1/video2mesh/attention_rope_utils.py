# attention_processor_0821
import torch


# copy from https://github.com/tairov/llama2.py/blob/6a102a5f71e50f868cd2d577724cefc7275b833c/model.py#L37C1-L79C50
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def apply_rotary_emb_1d_on_frame(
    x: torch.Tensor,                      # [B, F, S, H, D]
    freqs_cos: torch.Tensor,             # [F, D]
    freqs_sin: torch.Tensor              # [F, D]
) -> torch.Tensor:
    """
    在帧维度F上应用RoPE旋转位置编码
    x: 输入张量，形状为 [B, F, S, H, D]
    freqs_cos, freqs_sin: 预计算的频率余弦和正弦，形状为 [F, D]
    """
    # 检查维度
    B, F, S, H, D = x.shape
    assert D % 2 == 0, "D must be even for complex rotary"

    # 拆分实部虚部 -> [B, F, S, H, D/2]
    x_r, x_i = x.float().reshape(B, F, S, H, D // 2, 2).unbind(-1)  # 都是 [B, F, S, H, D/2]

    # reshape freqs for broadcasting: [1, F, 1, 1, D/2]
    freqs_cos = freqs_cos.view(1, F, 1, 1, D // 2)
    freqs_sin = freqs_sin.view(1, F, 1, 1, D // 2)

    # 应用RoPE旋转
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # 合并回 [B, F, S, H, D]
    x_out = torch.stack([x_out_r, x_out_i], dim=-1).flatten(-2)

    return x_out.type_as(x)  # 保持精度一致

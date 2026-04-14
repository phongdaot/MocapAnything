# attention_processor_0821
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as Func
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils import logging
from einops import rearrange


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


"""
新的attention设计:

已知F, 在输入是 hidden_states [B * F, S1, D1], encoder_hidden_states [B * F, S2, D2] 下: 
query
- q从hidden_states [B * F, S1, D1]获取, shape: [B * F, S1, D], 
- q拆分head, 变成shape [B * F, S1, head, head_dim], 
- 再根据已知F, 把q reshape到 [B, F, S1, head, head_dim]
- 在q的F维度上做一维的rope,
- 把q reshape成[B * F, S1, head, head_dim] 进入scale product
key/value
- cross attention的 encoder_hidden_states 是一个[B * F, S2, D2]的tensor
- kv从encoder_hidden_states获取, shape是[B * F, S2, D],  
- kv拆分head, 变成shape [B * F, S2, head, head_dim]
- 再根据已知F, 把kv reshape到 [B, F, S2, head, head_dim]
- 在k的F维度上做一维的rope
- 对kv做一下reshape, 变成[B, 1, F*S2, head, head_dim]
- 对kv做一下repeat, 变成[B, F, F*S2, head, head_dim] 
- 对kv做一下reshape, 变成[B*F, F*S2, head, head_dim] 进入scale product
解释: B是数据的batch, F是frame, S是token长度, D是dim,
目标: attention的时候希望q可以看到其他frame的信息, 用rope编码frame之间的距离. 
"""
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


class FlashTripoSGAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Tripo2DiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self, topk=True):
        if not hasattr(Func, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.topk = topk

    def qkv(self, attn, q, k, v, attn_mask, dropout_p, is_causal):
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = k.shape[-2] // 3

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = q1 @ k.transpose(-1, -2)
            sim = torch.mean(sim, -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
            topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
            v0 = torch.gather(v, dim=-2, index=topk_ind)
            k0 = torch.gather(k, dim=-2, index=topk_ind)
            out = Func.scaled_dot_product_attention(q, k0, v0)
        elif self.topk is False:
            out = Func.scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            outs = []
            for grid_coord, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                q1 = q_chunk[:, :, ::50, :]
                sim = q1 @ k.transpose(-1, -2)
                sim = torch.mean(sim, -2)
                topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
                topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
                v0 = torch.gather(v, dim=-2, index=topk_ind)
                k0 = torch.gather(k, dim=-2, index=topk_ind)
                out = Func.scaled_dot_product_attention(q_chunk, k0, v0)
                outs.append(out)
                start += count
            out = torch.cat(outs, dim=-2)
        self.topk = False
        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert False, "0821: 等新设计的attention在pipline需要的时候改放进来."
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that tripo2 split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # flashvdm topk
        hidden_states = self.qkv(attn, query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class TripoSGAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the TripoSG model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(Func, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        att_frames: Optional[int] = None,
        att_slidwindow: Optional[int] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that pre-trained models split heads first then split qkv or kv, like .view(..., attn.heads, 3, dim)
        # instead of .view(..., 3, attn.heads, dim). So we need to re-split here.
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        if att_frames is None:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        else:
            query = query.view(batch_size, -1, attn.heads, head_dim)  # .transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim)  # .transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim)  # .transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if att_frames is None:
            # Apply RoPE if needed
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                if not attn.is_cross_attention:
                    key = apply_rotary_emb(key, image_rotary_emb)
        else:
            # 新版的apply rope
            # 假设你在 init 中传入了 att_frames
            F = att_frames
            BF, S1, H, D = query.shape
            B = BF // F
            _, S2, _, _ = key.shape

            # # 在inference的时候, 会出现BF < F的情况 导致B=0, 这个是在视频末尾的时候凑补齐F个, 改成B=1, F=BF, 
            # if B == 0:
            #     print(f"[Check] Using short sequence: B={B}, F={F}, att_frames={att_frames} is not ok")
            #     F = BF
            #     B = BF // F
            #     print(f"[Check] Using short sequence: B={B}, F={F}, att_frames={att_frames} is ok")

            query = query.view(B, F, S1, H, D)  # -> [B, F, S1, H, D]
            key = key.view(B, F, S2, H, D)  # -> [B, F, S2, H, D]
            value = value.view(B, F, S2, H, D)  # -> [B, F, S2, H, D]

            # ====== 预计算frame位置编码 freq_cos, freq_sin ======
            freqs_cos, freqs_sin = precompute_freqs_cis(D, end=F)  # [F, D]
            freqs_cos = freqs_cos.to(query.device)
            freqs_sin = freqs_sin.to(query.device)

            # ====== 在帧维度上做RoPE ======
            query = apply_rotary_emb_1d_on_frame(query, freqs_cos, freqs_sin)  # [B, F, S1, H, D]
            key = apply_rotary_emb_1d_on_frame(key, freqs_cos, freqs_sin)  # [B, F, S2, H, D]

            if att_slidwindow is None:
                # ====== 处理key/value展开，使q看到所有frame的k/v ======

                # key: [B, F, S2, H, D] -> [B, 1, F*S2, H, D] -> expand到 [B, F, F*S2, H, D] -> reshape回 [B*F, F*S2, H, D]
                key = key.reshape(B, F * S2, H, D).unsqueeze(1)  # [B, 1, F*S2, H, D]
                key = key.expand(B, F, F * S2, H, D)  # [B, F, F*S2, H, D]
                key = key.reshape(B * F, F * S2, H, D).transpose(1, 2)  # [B*F, H, F*S2, D]

                # value 同理
                value = value.reshape(B, F * S2, H, D).unsqueeze(1)
                value = value.expand(B, F, F * S2, H, D)
                value = value.reshape(B * F, F * S2, H, D).transpose(1, 2)

                # query reshape 回去
                query = query.reshape(B * F, S1, H, D).transpose(1, 2)  # [B*F, H, S1, D]
            else:
                # ====== 处理key/value滑动窗口：先pad再切片 ======
                # 动态构建 K 和 V，使其具有滑动窗口特性
                F_s = att_slidwindow
                # 计算需要在F维度前后填充的帧数
                pad_frames = F_s // 2

                # 在帧维度（dim=1）上进行填充
                # pad(input, pad, mode='constant', value=0)
                # pad: (padding_left, padding_right)
                key_padded = Func.pad(key, (0, 0, 0, 0, 0, 0, pad_frames, pad_frames), 'constant', 0)
                value_padded = Func.pad(value, (0, 0, 0, 0, 0, 0, pad_frames, pad_frames), 'constant', 0)

                # 创建一个索引张量
                # 目标：对于每个查询帧 i (从0到F-1)，我们需要从key_padded中提取 i 到 i+F_s-1 范围内的所有帧
                base_indices = torch.arange(F, device=key.device)
                indices = base_indices.unsqueeze(1) + torch.arange(F_s, device=key.device).unsqueeze(0)

                # 使用高级索引直接提取所有窗口
                # key_padded 的形状是 [B, F+2*pad_frames, S2, H, D]
                # 索引张量 indices 的形状是 [F, F_s]
                key_windows = key_padded[:, indices, :, :, :]  # [B, F, F_s, S2, H, D]
                value_windows = value_padded[:, indices, :, :, :]  # [B, F, F_s, S2, H, D]

                # 合并 B 和 F 维度，以及 F_s 和 S2 维度
                # 最终形状：[B*F, F_s*S2, H, D]
                key = key_windows.reshape(B * F, F_s * S2, H, D)
                value = value_windows.reshape(B * F, F_s * S2, H, D)

                # 将 key 和 value 的维度调整为SDP attention需要的格式
                key = key.transpose(1, 2)  # [B*F, H, F_s*S2, D]
                value = value.transpose(1, 2)  # [B*F, H, F_s*S2, D]

                # query reshape 回去
                query = query.reshape(B * F, S1, H, D).transpose(1, 2)  # [B*F, H, S1, D]

            # 完成新设计的attention
            ##############################################################

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = Func.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class FusedTripoSGAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0) with fused
    projection layers. This is used in the HunyuanDiT model. It applies a s normalization layer and rotary embedding on
    query and key vector.
    """

    def __init__(self):
        if not hasattr(Func, "scaled_dot_product_attention"):
            raise ImportError(
                "FusedTripoSGAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert False, "0821: 等新设计的attention在pipline需要的时候改放进来."
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # NOTE that pre-trained split heads first, then split qkv
        if encoder_hidden_states is None:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )
            query = attn.to_q(hidden_states)

            kv = attn.to_kv(encoder_hidden_states)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = Func.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

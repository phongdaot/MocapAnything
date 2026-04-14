# attention_processor_0821
from typing import Optional

import torch
import torch.nn.functional as Func
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.utils import logging

from .attention_rope_utils import apply_rotary_emb_1d_on_frame, precompute_freqs_cis


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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

# triposg_transformer_0821
#
# DiTBlock extracted from triposg_transformer.py. Implementation is preserved
# byte-for-byte; only the surrounding imports have been narrowed to what this
# block actually uses.

from typing import Any, Dict, Optional

import torch
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch import nn

from .attention_processor_v2 import TripoSGAttnProcessor2_0


@maybe_allow_in_graph
class DiTBlock(nn.Module):
    r"""
    Transformer block used in Hunyuan-DiT model (https://github.com/Tencent/HunyuanDiT). Allow skip connection and
    QKNorm

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of headsto use for multi-head attention.
        cross_attention_dim (`int`,*optional*):
            The size of the encoder_hidden_states vector for cross attention.
        dropout(`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        activation_fn (`str`,*optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward. .
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*):
            The size of the hidden layer in the feed-forward block. Defaults to `None`.
        ff_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the feed-forward block.
        skip (`bool`, *optional*, defaults to `False`):
            Whether to use skip connection. Defaults to `False` for down-blocks and mid-blocks.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use normalization in QK calculation. Defaults to `True`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        use_self_attention: bool = True,
        self_attention_norm_type: Optional[str] = None,
        use_cross_attention: bool = True, # ada layer norm
        cross_attention_dim: Optional[int] = None,
        cross_attention_norm_type: Optional[str] = "fp32_layer_norm",
        use_cross_attention_2: bool = False,
        cross_attention_2_dim: Optional[int] = None,
        cross_attention_2_norm_type: Optional[str] = None,
        dropout=0.0,
        activation_fn: str = "gelu",
        norm_type: str = "fp32_layer_norm",  # TODO
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,  # int(dim * 4) if None
        ff_bias: bool = True,
        skip: bool = False,
        skip_concat_front: bool = False,  # [x, skip] or [skip, x]
        skip_norm_last: bool = False,  # this is an error
        qk_norm: bool = True,
        qkv_bias: bool = True,
        layer_idx: int = None,
    ):
        super().__init__()

        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.use_cross_attention_2 = use_cross_attention_2
        self.skip_concat_front = skip_concat_front
        self.skip_norm_last = skip_norm_last
        self.layer_idx = layer_idx
        # Define 3 blocks. Each block has its own normalization layer.
        # NOTE: when new version comes, check norm2 and norm 3
        # 1. Self-Attn
        if use_self_attention:
            if (
                self_attention_norm_type == "fp32_layer_norm"
                or self_attention_norm_type is None
            ):
                self.norm1 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                raise NotImplementedError

            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 2. Cross-Attn
        if use_cross_attention:
            assert cross_attention_dim is not None

            self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                cross_attention_norm=cross_attention_norm_type,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        if use_cross_attention_2:
            assert cross_attention_2_dim is not None

            self.norm2_2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2_2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_2_dim,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                cross_attention_norm=cross_attention_2_norm_type,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=final_dropout,  ### 0.0
            inner_dim=ff_inner_dim,  ### int(dim * mlp_ratio)
            bias=ff_bias,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_topk(self, topk):
        self.flash_processor.topk = topk

    def set_flash_processor(self, flash_processor):
        self.flash_processor = flash_processor
        self.attn2.processor = self.flash_processor

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Prepare attention kwargs
        attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        cross_attention_scale = attention_kwargs.pop("cross_attention_scale", 1.0)
        cross_attention_2_scale = attention_kwargs.pop("cross_attention_2_scale", 1.0)
        # attention_design的设计, 分别是self和cross能不能看到所有frame.
        seq_len = attention_kwargs.pop("seq_len", None)
        selfatt_temporal_layer_flag = attention_kwargs.pop("selfatt_temporal_layer_flag", None)
        selfatt_frame = seq_len if attention_kwargs.pop("selfatt_temporal", None) else None
        crossatt2_frame = seq_len if attention_kwargs.pop("crossatt2_temporal", None) else None
        selfatt_slidwindow = attention_kwargs.pop("selfatt_slidwindow", None)
        crossatt2_slidwindow = attention_kwargs.pop("crossatt2_slidwindow", None)

        # print(f'[DEBUG]: layer_idx - {self.layer_idx},  selfatt_temporal_layer_flag {selfatt_temporal_layer_flag}' )

        if selfatt_temporal_layer_flag is None:
            pass
        else:
            if self.layer_idx in selfatt_temporal_layer_flag:
                # print(f'[DEBUG]: use selselfatt_temporal at layer {self.layer_idx}')
                pass
            else:
                selfatt_frame = None
                selfatt_slidwindow = None


        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat(
                (
                    [skip, hidden_states]
                    if self.skip_concat_front
                    else [hidden_states, skip]
                ),
                dim=-1,
            )
            if self.skip_norm_last:
                # don't do this
                hidden_states = self.skip_linear(cat)
                hidden_states = self.skip_norm(hidden_states)
            else:
                cat = self.skip_norm(cat)
                hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        if self.use_self_attention:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(
                norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                att_frames=selfatt_frame,
                att_slidwindow=selfatt_slidwindow,
                **attention_kwargs,
            )
            hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        if self.use_cross_attention:
            if self.use_cross_attention_2:
                hidden_states = (
                    hidden_states
                    + self.attn2(
                        self.norm2(hidden_states),
                        encoder_hidden_states=encoder_hidden_states,
                        image_rotary_emb=image_rotary_emb,
                        att_frames=crossatt2_frame,
                        att_slidwindow=crossatt2_slidwindow,
                        **attention_kwargs,
                    ) * cross_attention_scale
                    + self.attn2_2(
                        self.norm2_2(hidden_states),
                        encoder_hidden_states=encoder_hidden_states_2,
                        image_rotary_emb=image_rotary_emb,
                        att_frames=crossatt2_frame,
                        att_slidwindow=crossatt2_slidwindow,
                        **attention_kwargs,
                    ) * cross_attention_2_scale
                )
            else:
                hidden_states = hidden_states + self.attn2(
                    self.norm2(hidden_states),
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    att_frames=crossatt2_frame,
                    att_slidwindow=crossatt2_slidwindow,
                    **attention_kwargs,
                ) * cross_attention_scale

        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states

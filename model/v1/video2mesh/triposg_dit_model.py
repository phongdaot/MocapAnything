# triposg_transformer_0821
#
# TripoSGDiTModel4D extracted from triposg_transformer.py. Implementation is
# preserved byte-for-byte; only the surrounding imports have been narrowed to
# what this model actually uses.

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import LayerNorm
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from torch import nn

from .attention_processor_v2 import FusedTripoSGAttnProcessor2_0, TripoSGAttnProcessor2_0
from .dit_block import DiTBlock
from TripoSG.triposg.models.transformers.modeling_outputs import Transformer1DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class TripoSGDiTModel4D(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    TripoSG: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        patch_size (`int`, *optional*):
            The size of the patch to use for the input.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward.
        sample_size (`int`, *optional*):
            The width of the latent images. This is fixed during training since it is used to learn a number of
            position embeddings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of dimension in the clip text embedding.
        hidden_size (`int`, *optional*):
            The size of hidden layer in the conditioning embedding layers.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden layer size to the input size.
        learn_sigma (`bool`, *optional*, defaults to `True`):
             Whether to predict variance.
        cross_attention_dim_t5 (`int`, *optional*):
            The number dimensions in t5 text embedding.
        pooled_projection_dim (`int`, *optional*):
            The size of the pooled projection.
        text_len (`int`, *optional*):
            The length of the clip text embedding.
        text_len_t5 (`int`, *optional*):
            The length of the T5 text embedding.
        use_style_cond_and_image_meta_size (`bool`,  *optional*):
            Whether or not to use style condition and image meta size. True for version <=1.1, False for version >= 1.2
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 64,
        num_layers: int = 21,
        cross_attention_dim: int = 1024,
        use_cross_attention_2: bool = False,
        cross_attention_2_dim: Optional[int] = None
    ):
        super().__init__()
        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width
        self.mlp_ratio = 4.0

        time_embed_dim, timestep_input_dim = self._set_time_proj(
            "positional",
            inner_dim=self.inner_dim,
            flip_sin_to_cos=False,
            freq_shift=0,
            time_embedding_dim=None,
        )
        self.time_proj = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="gelu", out_dim=self.inner_dim
        )
        self.proj_in = nn.Linear(self.config.in_channels, self.inner_dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=True,
                    cross_attention_dim=cross_attention_dim,
                    cross_attention_norm_type=None,
                    use_cross_attention_2=use_cross_attention_2,
                    cross_attention_2_dim=cross_attention_2_dim,
                    cross_attention_2_norm_type=None,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",  # TODO
                    norm_eps=1e-5,
                    ff_inner_dim=int(self.inner_dim * self.mlp_ratio),
                    skip=layer > num_layers // 2,
                    skip_concat_front=True,
                    skip_norm_last=True,  # this is an error
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    qkv_bias=False,
                    layer_idx=layer,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = LayerNorm(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _set_time_proj(
        self,
        time_embedding_type: str,
        inner_dim: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or inner_dim * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_embed = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or inner_dim * 4

            self.time_embed = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            timestep_input_dim = inner_dim
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedTripoSGAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is �� experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        assert False, "目前0821还没支持"
        self.set_attn_processor(FusedTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is �� experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(TripoSGAttnProcessor2_0())

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs, # <--- 在这里添加 **kwargs
    ) -> Union[Transformer1DModelOutput, Tuple]:

        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`): torch.Size([90, 2048, 64]) 
            The input tensor.  
        timestep ( `torch.LongTensor`, *optional*): torch.Size([90])
            Used to indicate denoising step. 
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*): torch.Size([90, 257, 1024])
            Conditional embeddings for cross attention layer.
        return_dict: bool
            Whether to return a dictionary.
        """
        # # 你可以在这里打印 kwargs 来查看 PEFT 到底传递了哪些额外参数
        # if kwargs:
        #     logger.warning(f"TripoSGDiTModel received unexpected keyword arguments: {kwargs}")

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # _, N, _ = hidden_states.shape

        # temb = self.time_embed(timestep).to(hidden_states.dtype)
        # temb = self.time_proj(temb)
        # temb = temb.unsqueeze(dim=1)  # unsqueeze to concat with hidden_states  torch.Size([90, 1, 2048])

        # hidden_states = self.proj_in(hidden_states) # torch.Size([90, 2048, 2048])

        # # N + 1 token
        # hidden_states = torch.cat([temb, hidden_states], dim=1)  # torch.Size([90, 2049, 2048])

        B1, F1, N1, D1 = hidden_states.shape  # 1, 20, 2048, 64

        temb = self.time_embed(timestep).to(hidden_states.dtype)  # [B, D]
        temb = self.time_proj(temb)  # [B, D2]

        hidden_states = self.proj_in(hidden_states) # B, F, N, D -> B, F, N, D2

        _, D2 = temb.shape  # [B, D2]

        # 扩展成 N 维度的 token（[B, F, 1, D]）拼接到 N 前
        temb_n = temb[:, None, None, :].expand(B1, F1, 1, D2)  # [B, F, 1, D]
        hidden_states = torch.cat([temb_n, hidden_states], dim=2)  # [B, F, 1+N, D]

        B2, F2, N2, D2 = hidden_states.shape
        # reshape → (B*F, S, D)
        hidden_states = hidden_states.reshape(B2 * F2, N2, D2)

        B3, F3, N3, D3 = encoder_hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.view(B3 * F3, N3, D3)

        if encoder_hidden_states_2 is not None:
            B4, F4, N4, D4 = encoder_hidden_states_2.shape
            encoder_hidden_states_2 = encoder_hidden_states_2.view(B4 * F4, N4, D4)

        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_2,
                    temb,
                    image_rotary_emb,
                    skip,
                    attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_2=encoder_hidden_states_2,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=attention_kwargs,
                )  # (N, L, D)

            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states) # torch.Size([90, 2049, 2048])
        hidden_states = hidden_states.reshape(B2, F2, N2, D2)
        hidden_states = hidden_states[:, :, -N1:]  # torch.Size([90, 2049, 2048])
        hidden_states = self.proj_out(hidden_states)  # torch.Size([90, 2048, 64])


        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

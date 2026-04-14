import torch.nn as nn
from .attention_processors import SlidingWindowAttnProcessor


# base attention
class SlidingWindowAttention(nn.Module):
    def __init__(self, dim, heads, cross_attention_dim=None, processor=None):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim or dim, dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim or dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.processor = processor or SlidingWindowAttnProcessor()

    def forward(self, x, encoder_hidden_states=None, **kwargs):
        return self.processor(
            self, x, encoder_hidden_states=encoder_hidden_states, **kwargs
        )


# DIT模块
class SlidingWindowDiTBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_attention_heads,
            cross_attention_dim=None,
            activation_fn="gelu",
            norm_eps=1e-5,
            use_cross_attention=False,
            layer_idx=0,
            processor=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.self_attn = SlidingWindowAttention(dim, num_attention_heads, processor=processor)
        self.cross_attn = (
            SlidingWindowAttention(dim, num_attention_heads, cross_attention_dim, processor=processor)
            if use_cross_attention else None
        )
        # self.ffn = nn.Sequential(
        #     nn.LayerNorm(dim, eps=norm_eps),
        #     nn.Linear(dim, dim * 4),
        #     nn.GELU() if activation_fn == "gelu" else nn.ReLU(),
        #     nn.Linear(dim * 4, dim),
        # )
        self.layer_idx = layer_idx

    def forward(
            self,
            x,
            encoder_hidden_states=None,
            attention_kwargs=None,
            joint_mask=None,
    ):
        # 1. 每层独立解析 attention_kwargs
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}

        # 通用解析
        cross_attention_scale = block_attention_kwargs.pop("cross_attention_scale", 1.0)
        seq_len = block_attention_kwargs.pop("seq_len", None)
        selfatt_temporal_layer_flag = block_attention_kwargs.pop("selfatt_temporal_layer_flag", None)
        selfatt_frame = seq_len if block_attention_kwargs.pop("selfatt_temporal", None) else None
        crossatt2_frame = seq_len if block_attention_kwargs.pop("crossatt2_temporal", None) else None
        selfatt_slidwindow = block_attention_kwargs.pop("selfatt_slidwindow", None)
        crossatt2_slidwindow = block_attention_kwargs.pop("crossatt2_slidwindow", None)

        # 2. 当前层是否启用时序/窗口
        if selfatt_temporal_layer_flag is not None:
            if self.layer_idx not in selfatt_temporal_layer_flag:
                pass
            else:
                selfatt_frame = None
                selfatt_slidwindow = None

        if joint_mask is not None:
            B, seq_len, J = joint_mask.shape
            x = x * joint_mask.reshape(B * seq_len, J).unsqueeze(-1)

        # 3. Self-attention
        x1 = self.norm1(x)
        x_sa = self.self_attn(
            x1,
            att_frames=selfatt_frame,
            att_slidwindow=selfatt_slidwindow,
            joint_mask=joint_mask,
        )
        x = x + x_sa

        if joint_mask is not None:
            B, seq_len, J = joint_mask.shape
            x = x * joint_mask.reshape(B * seq_len, J).unsqueeze(-1)  # ✅ 残差后归零，阻断信息扩散

        # 4. Cross-attention（如有）
        if self.cross_attn is not None and encoder_hidden_states is not None:
            x2 = self.norm2(x)
            x_ca = self.cross_attn(
                x2,
                encoder_hidden_states=encoder_hidden_states,
                att_frames=crossatt2_frame,
                att_slidwindow=crossatt2_slidwindow,
            )
            x = x + x_ca * cross_attention_scale

        if joint_mask is not None:
            B, seq_len, J = joint_mask.shape
            x = x * joint_mask.reshape(B * seq_len, J).unsqueeze(-1)  # ✅ 残差后归零，阻断信息扩散

        # x = x + self.ffn(x)
        return x

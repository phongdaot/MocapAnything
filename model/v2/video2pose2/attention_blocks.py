import torch.nn as nn
import torch
from .attention_processors import *

class FrequencyPositionalEmbedding(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(
            self,
            num_freqs: int = 6,
            logspace: bool = True,
            input_dim: int = 3,
            include_input: bool = True,
            include_pi: bool = True,
    ) -> None:
        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

# ====================================================================
# ==================== 模型定义 =======================================
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

        # tree mask
        if tree_mask is not None and self.use_tree_mask:
            attn_score = attn_score.masked_fill(~tree_mask[:, None, :, :], float("-inf"))

        # key mask
        if mask is not None:
            mask_k = mask[:, None, None, :]
            attn_score = attn_score.masked_fill(~mask_k, float("-inf"))

        # 避免整行都是 -inf 导致 NaN
        invalid_rows = torch.isinf(attn_score).all(dim=-1, keepdim=True)
        attn_score = torch.where(invalid_rows, torch.zeros_like(attn_score), attn_score)

        attn = torch.softmax(attn_score.float(), dim=-1).to(attn_score.dtype)

        # query mask
        if mask is not None:
            mask_q = mask[:, None, :, None].float()
            attn = attn * mask_q

        # 再乘一次 tree mask，确保非法边为0
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


# base attention
class SimpleAttention(nn.Module):
    def __init__(self, dim, heads, cross_attention_dim=None, processor=None):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim or dim, dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim or dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.processor = processor or SimpleAttnProcessor()

    def forward(self, x, encoder_hidden_states=None, **kwargs):
        return self.processor(
            self, x, encoder_hidden_states=encoder_hidden_states, **kwargs
        )

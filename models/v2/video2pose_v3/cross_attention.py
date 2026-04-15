### cross_attention.py ###
import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    """
    Pre-norm multi-head self-attention over joints, with padding-joint masking.
    Input: [B,J,D] -> Output: [B,J,D]
    """

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, joint_mask=None):
        h = self.norm(x)
        key_padding_mask = None
        if joint_mask is not None:
            key_padding_mask = ~joint_mask.bool()

        out, _ = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.out(out)
        out = self.dropout(out)
        x = x + out

        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()
        return x


class JointImageCrossAttention(nn.Module):
    """
    Cross-attention from joint queries to image patch features.

    Supports both the static case (query [B,J,D], kv [B,P,D]) and the temporal
    case (query [B,T,J,D], kv [B,T,P,D]) by folding time into the batch
    dimension in the temporal case.
    """

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

    def forward(self, query_feat, image_feat, joint_mask=None):
        if query_feat.dim() == 3:
            q = self.norm_q(query_feat)
            kv = self.norm_kv(image_feat)

            out, _ = self.attn(
                query=q,
                key=kv,
                value=kv,
                need_weights=False,
            )
            out = self.out_proj(out)
            out = self.dropout(out)
            out = query_feat + out

            if joint_mask is not None:
                out = out * joint_mask.unsqueeze(-1).float()
            return out

        if query_feat.dim() == 4:
            B, T, J, D = query_feat.shape
            P = image_feat.shape[2]

            q = query_feat.reshape(B * T, J, D)
            kv = image_feat.reshape(B * T, P, D)

            q = self.norm_q(q)
            kv = self.norm_kv(kv)

            out, _ = self.attn(
                query=q,
                key=kv,
                value=kv,
                need_weights=False,
            )
            out = self.out_proj(out)
            out = self.dropout(out)
            out = out.reshape(B, T, J, D)
            out = query_feat + out

            if joint_mask is not None:
                if joint_mask.dim() == 2:
                    out = out * joint_mask.unsqueeze(1).unsqueeze(-1).float()
                else:
                    out = out * joint_mask.unsqueeze(-1).float()
            return out

        raise ValueError(f"Unsupported query_feat dim: {query_feat.dim()}")

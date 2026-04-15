import torch.nn as nn


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

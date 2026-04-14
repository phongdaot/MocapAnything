import torch.nn as nn

from .film import FiLMCondition
from .graph_attention import GraphMultiHeadAttention
from .temporal_attention import TemporalPerJointTransformerBlock
from .cross_attention import JointMemoryCrossAttention


# =========================================================
# Rot Decoder
# =========================================================
class RotDecoderBlock(nn.Module):
    def __init__(
        self,
        q_dim=256,
        num_heads=8,
        dropout=0.1,
        temporal_window=2,
        use_tree_mask=False,
        use_rest_film=True,
        use_cross_attn=True,
    ):
        super().__init__()
        self.use_rest_film = use_rest_film
        self.use_cross_attn = use_cross_attn

        self.film = FiLMCondition(q_dim) if use_rest_film else None
        self.temporal = TemporalPerJointTransformerBlock(
            dim=q_dim,
            nheads=num_heads,
            dropout=dropout,
            ff_mult=4,
            temporal_window=temporal_window,
            use_temporal_bias=True,
        )
        self.graph_norm = nn.LayerNorm(q_dim)
        self.graph = GraphMultiHeadAttention(
            q_dim,
            num_heads,
            dropout=dropout,
            use_tree_mask=use_tree_mask,
        )
        if self.use_cross_attn:
            self.cross = JointMemoryCrossAttention(
                d_model=q_dim,
                nheads=num_heads,
                dropout=dropout,
            )
        self.ffn_norm = nn.LayerNorm(q_dim)
        self.ffn = nn.Sequential(
            nn.Linear(q_dim, q_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(q_dim * 4, q_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,               # [B,T,J,D]
        rest_t,          # [B,T,J,D]
        memory_feat,     # [B,N,J,D]
        joint_mask,      # [B,J]
        ancestor_mask,      # [B,J,J]
        graph_hop,       # [B,J,J]
        graph_edge,      # [B,J,J]
    ):
        B, T, J, D = x.shape
        joint_mask_bt = joint_mask.unsqueeze(1).expand(-1, T, -1)

        if self.film is not None:
            x = self.film(x, rest_t)
            x = x * joint_mask_bt.unsqueeze(-1).float()

        x = self.temporal(x, joint_mask=joint_mask_bt)

        x2 = self.graph_norm(x).reshape(B * T, J, D)
        jm = joint_mask_bt.reshape(B * T, J)
        gm = ancestor_mask.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)
        gh = graph_hop.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)
        ge = graph_edge.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, J, J)

        x2 = self.graph(
            x2, x2, x2,
            gh, ge,
            mask=jm,
            tree_mask=gm,
        )
        x = x + x2.reshape(B, T, J, D)
        x = x * joint_mask_bt.unsqueeze(-1).float()

        if self.use_cross_attn:
            x = x + self.cross(x, memory_feat, joint_mask)
            x = x * joint_mask_bt.unsqueeze(-1).float()

        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        x = x * joint_mask_bt.unsqueeze(-1).float()
        return x

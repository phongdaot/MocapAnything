### temporal_trunk.py ###
import torch
import torch.nn as nn

from .graph_attention import GraphMultiHeadAttention
from .temporal_attention import TemporalPerJointTransformerBlock
from .cross_attention import JointImageCrossAttention


class Video2PoseTemporalBlock(nn.Module):
    """
    One temporal trunk block:
      1) per-joint temporal transformer (with sliding window + RoPE)
      2) optional graph attention over the skeleton (per-frame)
      3) cross-attention from joint queries to per-frame image tokens
      4) feed-forward network
    """

    def __init__(
        self,
        q_dim=256,
        num_heads=8,
        dropout=0.1,
        temporal_window=2,
        use_graph=True,
        use_tree_mask=True,
    ):
        super().__init__()
        self.use_graph = use_graph

        self.temporal = TemporalPerJointTransformerBlock(
            dim=q_dim,
            nheads=num_heads,
            dropout=dropout,
            ff_mult=4,
            temporal_window=temporal_window,
            use_temporal_bias=True,
        )

        if use_graph:
            self.graph_norm = nn.LayerNorm(q_dim)
            self.graph = GraphMultiHeadAttention(
                q_dim,
                num_heads,
                dropout=dropout,
                use_tree_mask=use_tree_mask,
            )

        self.cross_img = JointImageCrossAttention(
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
        x,
        img_feat,
        joint_mask,
        graph_hop=None,
        graph_edge=None,
        tree_mask=None,
    ):
        """
        x       : [B,T,J,D]
        img_feat: [B,T,P,D]
        """
        B, T, J, D = x.shape

        x = self.temporal(x, joint_mask=joint_mask)

        if self.use_graph and graph_hop is not None:
            x2 = self.graph_norm(x).reshape(B * T, J, D)
            jm = joint_mask.reshape(B * T, J)
            gh = graph_hop.reshape(B * T, J, J)
            ge = graph_edge.reshape(B * T, J, J)
            tm = tree_mask.reshape(B * T, J, J) if tree_mask is not None else None

            x2 = self.graph(
                x2, x2, x2,
                gh, ge,
                mask=jm,
                tree_mask=tm,
            )
            x = x + x2.reshape(B, T, J, D)
            x = x * joint_mask.unsqueeze(-1).float()

        x = self.cross_img(x, img_feat, joint_mask=joint_mask)

        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h
        x = x * joint_mask.unsqueeze(-1).float()

        return x


class Video2PoseModelSliding(nn.Module):
    """
    Video2Pose temporal trunk: stacks Video2PoseTemporalBlocks on top of the
    per-joint reference query, attending to per-frame image features and
    producing per-frame 3D joint positions.
    """

    def __init__(
        self,
        num_layers=12,
        q_dim=256,
        img_dim=1024,
        num_joints=150,
        num_heads=8,
        temporal_window=2,
        use_graph_temporal_inner=False,
        dropout=0.1,
    ):
        super().__init__()
        self.use_graph_temporal_inner = use_graph_temporal_inner
        self.img_proj = nn.Linear(img_dim, q_dim)

        self.blocks = nn.ModuleList([
            Video2PoseTemporalBlock(
                q_dim=q_dim,
                num_heads=num_heads,
                dropout=dropout,
                temporal_window=temporal_window,
                use_graph=use_graph_temporal_inner,
                use_tree_mask=(i % 2 == 0),
            )
            for i in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(q_dim),
            nn.Linear(q_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(
        self,
        ref_query,
        cond_img,
        joint_mask=None,
        graph_hop=None,
        graph_edge=None,
        tree_mask=None,
    ):
        """
        ref_query: [B,J,D]
        cond_img : [B,T,P,img_dim]
        """
        B, T, _, _ = cond_img.shape
        _, J, D = ref_query.shape

        x = ref_query.unsqueeze(1).expand(B, T, J, D).contiguous()
        img_feat = self.img_proj(cond_img)

        for blk in self.blocks:
            x = blk(
                x=x,
                img_feat=img_feat,
                joint_mask=joint_mask,
                graph_hop=graph_hop if self.use_graph_temporal_inner else None,
                graph_edge=graph_edge if self.use_graph_temporal_inner else None,
                tree_mask=tree_mask if self.use_graph_temporal_inner else None,
            )

        out = self.head(x)
        return out

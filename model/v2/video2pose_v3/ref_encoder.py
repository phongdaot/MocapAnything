### ref_encoder.py ###
import torch
import torch.nn as nn

from .positional_embedding import FrequencyPositionalEmbedding
from .graph_attention import GraphMultiHeadAttention
from .cross_attention import SimpleSelfAttention, JointImageCrossAttention


class RefFusionBlock(nn.Module):
    """
    One fusion layer for the reference-frame query encoder.

    The block applies (optionally) graph attention over the skeleton,
    followed by joint self-attention, cross-attention to the reference
    image, and a feed-forward network.
    """

    def __init__(self, q_dim=256, num_heads=8, dropout=0.1, use_tree_mask=True):
        super().__init__()
        self.graph = GraphMultiHeadAttention(
            q_dim,
            num_heads,
            dropout=dropout,
            use_tree_mask=use_tree_mask,
        )
        self.self_attn = SimpleSelfAttention(q_dim, heads=num_heads, dropout=dropout)
        self.cross_img = JointImageCrossAttention(
            q_dim, nheads=num_heads, dropout=dropout
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
        img_cond,
        joint_mask=None,
        graph_hop=None,
        graph_edge=None,
        tree_mask=None,
    ):
        if graph_hop is not None and graph_edge is not None:
            x2 = self.graph(
                x, x, x,
                graph_hop, graph_edge,
                mask=joint_mask,
                tree_mask=tree_mask,
            )
            x = x + x2
            if joint_mask is not None:
                x = x * joint_mask.unsqueeze(-1).float()

        x = self.self_attn(x, joint_mask=joint_mask)
        x = self.cross_img(x, img_cond, joint_mask=joint_mask)

        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h

        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()

        return x


class RefQueryEncoder(nn.Module):
    """
    Encodes the reference pose + reference image into a per-joint query
    embedding used by the downstream temporal model.

    Input:
      - ref_position     [B,J,3]
      - ref_image_embed  [B,P,img_dim]
      - joint_mask       [B,J]
      - graph_hop        [B,J,J]
      - graph_edge       [B,J,J]
      - joint_t5embed    [B,J,joint_embed_dim] (optional)
      - tree_mask        [B,J,J] (optional ancestor mask)

    Output:
      - per-joint query  [B,J,q_dim]
    """

    def __init__(
        self,
        q_dim=256,
        img_dim=1024,
        num_heads=8,
        num_layers=4,
        joint_embed_dim=768,
        use_joint_embed=False,
        use_graph_ref_inner=False,
        dropout=0.1,
    ):
        super().__init__()
        self.pos_embedder = FrequencyPositionalEmbedding(num_freqs=8, input_dim=3)
        self.pose_proj = nn.Linear(self.pos_embedder.out_dim, q_dim)
        self.img_proj = nn.Linear(img_dim, q_dim)

        self.use_joint_embed = use_joint_embed
        self.use_graph_ref_inner = use_graph_ref_inner

        if self.use_joint_embed:
            self.joint_t5proj = nn.Linear(joint_embed_dim, q_dim)

        self.fusion_blocks = nn.ModuleList([
            RefFusionBlock(
                q_dim=q_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_tree_mask=(i % 2 == 0),
            )
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(q_dim)

    def forward(
        self,
        ref_position,
        ref_image_embed,
        joint_mask=None,
        graph_hop=None,
        graph_edge=None,
        joint_t5embed=None,
        tree_mask=None,
    ):
        ref_position_enc = self.pos_embedder(ref_position)
        x = self.pose_proj(ref_position_enc)

        if self.use_joint_embed and joint_t5embed is not None:
            x = x + self.joint_t5proj(joint_t5embed)

        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()

        img_cond = self.img_proj(ref_image_embed)

        for blk in self.fusion_blocks:
            x = blk(
                x,
                img_cond,
                joint_mask=joint_mask,
                graph_hop=graph_hop if self.use_graph_ref_inner else None,
                graph_edge=graph_edge if self.use_graph_ref_inner else None,
                tree_mask=tree_mask if self.use_graph_ref_inner else None,
            )

        x = self.final_norm(x)
        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()
        return x

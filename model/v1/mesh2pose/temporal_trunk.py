import torch
import torch.nn as nn

from .attention_processors import SlidingWindowAttnProcessor
from .positional_embedding import FrequencyPositionalEmbedding
from .graph_attention import GraphMultiHeadAttention
from .sliding_window_attention import SlidingWindowDiTBlock


# 预测网络:
# 重复添加cross mesh和cross image.
class Mesh2PoseModelSliding(nn.Module):
    def __init__(self, num_layers=4, q_dim=256, mesh_dim=64, img_dim=1024, num_joints=52, num_heads=8,
                 use_graph_temporal_outer=False, use_graph_temporal_inner=False):

        super().__init__()
        self.use_graph_temporal_outer = use_graph_temporal_outer
        self.use_graph_temporal_inner = use_graph_temporal_inner
        self.pos_embedder = FrequencyPositionalEmbedding(num_freqs=8, input_dim=3)
        self.mesh_proj = nn.Linear(mesh_dim + self.pos_embedder.out_dim, q_dim)
        self.img_proj = nn.Linear(img_dim, q_dim)
        if self.use_graph_temporal_outer:
            # === 新增 graph attention 层 ===
            self.graph_attn = GraphMultiHeadAttention(q_dim, num_heads)
        if self.use_graph_temporal_inner:
            self.blocks_graph = nn.ModuleList([GraphMultiHeadAttention(q_dim, num_heads) for i in range(num_layers)])
        self.blocks_img = nn.ModuleList([
            SlidingWindowDiTBlock(
                q_dim, num_heads,
                cross_attention_dim=q_dim,
                use_cross_attention=True,
                layer_idx=i,
                processor=SlidingWindowAttnProcessor()
            )
            for i in range(num_layers)
        ])
        self.blocks_mesh = nn.ModuleList([
            SlidingWindowDiTBlock(
                q_dim, num_heads,
                cross_attention_dim=q_dim,
                use_cross_attention=True,
                layer_idx=i,
                processor=SlidingWindowAttnProcessor()
            )
            for i in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(q_dim),
            nn.Linear(q_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, ref_query, cond_mesh, cond_surface_normal, cond_img, joint_mask=None, attention_kwargs=None,
                graph_hop=None, graph_edge=None):

        B, T, _, _ = cond_mesh.shape
        _, J, D = ref_query.shape
        # queries = self.joint_queries.expand(B*T, -1, -1)
        queries = ref_query.unsqueeze(1).expand(B, T, J, D).reshape(B * T, J, D)  # 这里是不是应该保留T的信息好一些.?
        cond_mesh_enc = torch.cat([self.pos_embedder(cond_mesh), cond_surface_normal], dim=-1)
        cond_mesh = self.mesh_proj(cond_mesh_enc)
        cond_img = self.img_proj(cond_img)

        # ✅ (1) 可选在整个 temporal 前加入 Graph Attention
        if self.use_graph_temporal_outer and graph_hop is not None:
            queries = queries + self.graph_attn(queries, queries, queries, graph_hop, graph_edge, joint_mask)

        # attention_kwargs 可直接外部传入，不要提前pop
        for i, (block_img, block_mesh) in enumerate(zip(self.blocks_img, self.blocks_mesh)):
            queries = block_img(
                queries,
                encoder_hidden_states=cond_img,
                attention_kwargs=attention_kwargs,  # 原样传递
                joint_mask=joint_mask,  # ✅ 传进去
            )

            if self.use_graph_temporal_inner and graph_hop is not None:
                queries = queries + self.blocks_graph[i](queries, queries, queries, graph_hop, graph_edge, joint_mask)

            queries = block_mesh(
                queries,
                encoder_hidden_states=cond_mesh,
                attention_kwargs=attention_kwargs,  # 原样传递
                joint_mask=joint_mask,  # ✅ 传进去
            )
        out = self.head(queries)
        out = out.reshape(B, T, out.size(1), out.size(2))
        return out

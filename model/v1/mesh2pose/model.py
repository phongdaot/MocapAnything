import torch.nn as nn
import torch
from .attention_blocks import *

# ---- 单层融合模块：self + cross1 + cross2 ----
class RefFusionBlock(nn.Module):
    def __init__(self, q_dim=256, num_heads=8, cross_dim=256):
        super().__init__()
        processor = SimpleAttnProcessor()
        # === 新增 graph attention 层 ===
        self.graph_attn = GraphMultiHeadAttention(q_dim, num_heads)
        self.self_attn = SimpleAttention(q_dim, num_heads, processor=processor)
        self.cross_img = SimpleAttention(q_dim, num_heads, processor=processor)
        self.cross_mesh = SimpleAttention(q_dim, num_heads, processor=processor)
        self.norm = nn.LayerNorm(q_dim)
        self.res_scale = nn.Parameter(torch.ones(1))  # 可学习残差比例

    def forward(self, x, img_cond, mesh_cond, joint_mask=None, attention_kwargs=None, graph_hop=None, graph_edge=None):
        # === 新增: Graph Attention ===
        if graph_hop is not None and graph_edge is not None:
            x = x + self.graph_attn(x, x, x, graph_hop, graph_edge, joint_mask)  # debug进去看下, mask需要改一下-1e9

        # --- self-attn ---  (有了graph attention之后还需要这个吗?)
        x = x + self.self_attn(x, joint_mask=joint_mask)

        # --- cross-attn1 (image) ---
        x = x + self.cross_img(x, encoder_hidden_states=img_cond, joint_mask=joint_mask)

        # --- cross-attn2 (mesh) ---
        x = x + self.cross_mesh(x, encoder_hidden_states=mesh_cond, joint_mask=joint_mask)

        # --- 输出层 ---
        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()

        return self.norm(x * self.res_scale)


class RefQueryEncoder(nn.Module):
    def __init__(self, q_dim=256, mesh_dim=64, img_dim=1024,
                 num_heads=8, num_layers=3, joint_embed_dim=768,
                 use_joint_embed=False, use_graph_ref_outer=False, use_graph_ref_inner=False):
        super().__init__()
        self.pos_embedder = FrequencyPositionalEmbedding(num_freqs=8, input_dim=3)
        self.pose_proj = nn.Linear(self.pos_embedder.out_dim, q_dim)
        self.img_proj = nn.Linear(img_dim, q_dim)
        self.mesh_proj = nn.Linear(mesh_dim + self.pos_embedder.out_dim, q_dim)

        self.use_joint_embed = use_joint_embed
        self.use_graph_ref_outer = use_graph_ref_outer
        self.use_graph_ref_inner = use_graph_ref_inner
        if self.use_joint_embed:
            self.joint_t5proj = nn.Linear(joint_embed_dim, q_dim)
        if self.use_graph_ref_outer:
            self.graph_attn = GraphMultiHeadAttention(q_dim, num_heads)

        # === 叠加若干融合层 ===
        self.fusion_blocks = nn.ModuleList([
            RefFusionBlock(q_dim=q_dim, num_heads=num_heads, cross_dim=q_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(q_dim)

    def forward(self, ref_position, ref_image_embed, ref_mesh, ref_surface_normal,
                joint_mask=None, attention_kwargs=None, graph_hop=None, graph_edge=None, joint_t5embed=None):
        ref_position_enc = self.pos_embedder(ref_position)
        x = self.pose_proj(ref_position_enc)
        # === 融合 joint embedding ===
        if self.use_joint_embed and joint_t5embed is not None:
            joint_feat = self.joint_t5proj(joint_t5embed)  # [B,J,q_dim]
            x = x + joint_feat

        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()

        # ✅ 2. 可选 graph attention 先于融合层
        if self.use_graph_ref_outer and graph_hop is not None:
            x = x + self.graph_attn(x, x, x, graph_hop, graph_edge, joint_mask)

        img_cond = self.img_proj(ref_image_embed)
        ref_mesh_enc = torch.cat([self.pos_embedder(ref_mesh), ref_surface_normal], dim=-1)
        mesh_cond = self.mesh_proj(ref_mesh_enc)

        for blk in self.fusion_blocks:
            x = blk(x, img_cond, mesh_cond, joint_mask=joint_mask,
                    attention_kwargs=attention_kwargs,
                    graph_hop=graph_hop if self.use_graph_ref_inner else None,
                    graph_edge=graph_edge if self.use_graph_ref_inner else None)

        if joint_mask is not None:
            x = x * joint_mask.unsqueeze(-1).float()

        return self.final_norm(x)


### 合并模型:
class RefGuidedMesh2PoseModel(nn.Module):
    def __init__(self,
                 num_layers=4,
                 q_dim=256,
                 mesh_dim=64,
                 img_dim=1024,
                 num_joints=150,
                 num_heads=8,
                 ref_layers=3,
                 use_joint_embed=False,
                 use_graph_ref_outer=False,
                 use_graph_ref_inner=False,
                 use_graph_temporal_outer=False,
                 use_graph_temporal_inner=False,
                 ):
        super().__init__()
        # === 参考帧编码器 ===
        self.ref_encoder = RefQueryEncoder(
            q_dim=q_dim, mesh_dim=mesh_dim, img_dim=img_dim,
            num_heads=num_heads, num_layers=ref_layers,
            use_joint_embed=use_joint_embed,
            use_graph_ref_outer=use_graph_ref_outer,
            use_graph_ref_inner=use_graph_ref_inner
        )
        # === 时序模型 ===
        self.temporal_model = Mesh2PoseModelSliding(
            num_layers=num_layers,
            q_dim=q_dim, mesh_dim=mesh_dim, img_dim=img_dim,
            num_joints=num_joints, num_heads=num_heads,
            use_graph_temporal_outer=use_graph_temporal_outer,
            use_graph_temporal_inner=use_graph_temporal_inner
        )

    def forward(self, batch, attention_kwargs=None):
        # 解包 batch
        # position = batch["position"]          # [B,F,J,3]
        image_embed = batch["image_embed"]  # [B,F,P,1024]
        mesh = batch["mesh"]  # [B,F,L,64]
        surface_normal = batch["surface_normal"]
        ref_pos = batch["ref_position"]  # [B*1,J,3]
        ref_img = batch["ref_image_embed"]  # [B*1,P,1024]
        ref_mesh = batch["ref_mesh"]  # [B*1,L,64]
        ref_surface_normal = batch["ref_surface_normal"]
        joint_mask = batch["joint_mask"]  # [B*1,J]
        graph_hop = batch["graph_hop"]  # B J J
        graph_edge = batch["graph_edge"]  # B J J
        joint_t5embed = batch["joint_t5embed"]

        # 1️⃣ 得到参考帧的融合query
        ref_query = self.ref_encoder(
            ref_position=ref_pos,
            ref_image_embed=ref_img,
            ref_mesh=ref_mesh,
            ref_surface_normal=ref_surface_normal,
            joint_mask=joint_mask,
            attention_kwargs=attention_kwargs,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
            joint_t5embed=joint_t5embed,
        )

        # 重新shape一下 joint_mask for temporal  [B*1,J]  -> [B, F,J]
        F = image_embed.shape[1]
        joint_mask = joint_mask.unsqueeze(1).expand(-1, F, -1)
        graph_hop = graph_hop.unsqueeze(1).expand(-1, F, -1, -1)
        graph_edge = graph_edge.unsqueeze(1).expand(-1, F, -1, -1)

        # 2️⃣ 时序模型
        pose_pred = self.temporal_model(
            ref_query=ref_query,
            cond_mesh=mesh,
            cond_surface_normal=surface_normal,
            cond_img=image_embed,
            joint_mask=joint_mask,
            attention_kwargs=attention_kwargs,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
        )

        # 把静止的joint替换一下.
        static_mask = batch["static_mask"].bool()  # [B, J]
        ref_pos = batch["ref_position"]  # [B, J, 3]
        # 先扩展维度
        static_mask_4d = static_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, J, 1]
        ref_pos_4d = ref_pos.unsqueeze(1).expand(-1, pose_pred.shape[1], -1, -1)
        # 替换静止关节
        pose_pred = pose_pred * (~static_mask_4d) + ref_pos_4d * static_mask_4d

        return pose_pred

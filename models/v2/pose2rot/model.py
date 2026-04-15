import torch.nn as nn
import torch
from .attention_blocks import *
from torch.utils.checkpoint import checkpoint
from typing import Optional

# =========================================================
# Rest Encoder
# =========================================================
class RestPoseEncoder(nn.Module):
    def __init__(
        self,
        q_dim=256,
        joint_embed_dim=768,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        use_grad_checkpoint=False,
    ):
        super().__init__()
        self.offset_proj = nn.Linear(3, q_dim)
        self.joint_t5proj = nn.Linear(joint_embed_dim, q_dim)
        self.use_grad_checkpoint = use_grad_checkpoint

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(q_dim),
                "graph": GraphMultiHeadAttention(
                    q_dim,
                    num_heads,
                    dropout=dropout,
                    use_tree_mask=(layer_idx % 2 == 0),
                ),
                "ffn_norm": nn.LayerNorm(q_dim),
                "ffn": nn.Sequential(
                    nn.Linear(q_dim, q_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(q_dim * 4, q_dim),
                    nn.Dropout(dropout),
                )
            })
            for layer_idx in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(q_dim)

    def _run_block(self, blk, x, joint_mask, ancestor_mask, graph_hop, graph_edge):
        h = blk["norm"](x)
        h = blk["graph"](
            h, h, h,
            graph_hop, graph_edge,
            mask=joint_mask,
            tree_mask=ancestor_mask,
        )
        x = x + h
        x = x * joint_mask.unsqueeze(-1).float()

        h = blk["ffn_norm"](x)
        h = blk["ffn"](h)
        x = x + h
        x = x * joint_mask.unsqueeze(-1).float()
        return x

    def forward(
        self,
        offset,         # [B,J,3]
        joint_t5embed,  # [B,J,Dt5]
        joint_mask,     # [B,J]
        ancestor_mask,     # [B,J,J]
        graph_hop,      # [B,J,J]
        graph_edge,     # [B,J,J]
    ):
        x = self.offset_proj(offset) + self.joint_t5proj(joint_t5embed)
        x = x * joint_mask.unsqueeze(-1).float()

        for blk in self.layers:
            if self.use_grad_checkpoint and self.training:
                def custom_forward(x_in):
                    return self._run_block(blk, x_in, joint_mask, ancestor_mask, graph_hop, graph_edge)
                x = checkpoint(custom_forward, x, use_reentrant=False)
            else:
                x = self._run_block(blk, x, joint_mask, ancestor_mask, graph_hop, graph_edge)

        x = self.out_norm(x)
        x = x * joint_mask.unsqueeze(-1).float()
        return x

# =========================================================
# Pose Encoder Lite
# =========================================================
class PoseEncoderLite(nn.Module):
    def __init__(
        self,
        q_dim=256,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        temporal_window=2,
        joint_embed_dim=768,
        use_rest_film=True,
        use_graph=True,
        use_grad_checkpoint=False,
    ):
        super().__init__()
        self.in_proj = nn.Linear(3, q_dim)
        self.joint_t5proj = nn.Linear(joint_embed_dim, q_dim)
        self.use_rest_film = use_rest_film
        self.use_graph = use_graph
        self.use_grad_checkpoint = use_grad_checkpoint

        if use_rest_film:
            self.rest_film_blocks = nn.ModuleList([FiLMCondition(q_dim) for _ in range(num_layers)])
        else:
            self.rest_film_blocks = None

        self.temporal_blocks = nn.ModuleList([
            TemporalPerJointTransformerBlock(
                dim=q_dim,
                nheads=num_heads,
                dropout=dropout,
                ff_mult=4,
                temporal_window=temporal_window,
                use_temporal_bias=True,
            )
            for _ in range(num_layers)
        ])

        if use_graph:
            self.graph_blocks = nn.ModuleList([
                GraphMultiHeadAttention(
                    q_dim,
                    num_heads,
                    dropout=dropout,
                    use_tree_mask=(layer_idx % 2 == 0),
                )
                for layer_idx in range(num_layers)
            ])
        else:
            self.graph_blocks = None

        self.out_norm = nn.LayerNorm(q_dim)

    def _run_block(
        self,
        temporal_blk,
        graph_blk,
        film_blk,
        x,
        rest_t,
        joint_mask_bt,
        ancestor_mask_bt,
        graph_hop_bt,
        graph_edge_bt,
    ):
        B, T, J, D = x.shape

        if film_blk is not None:
            x = film_blk(x, rest_t)
            x = x * joint_mask_bt.unsqueeze(-1).float()

        x = temporal_blk(x, joint_mask=joint_mask_bt)

        if graph_blk is not None:
            x2 = x.reshape(B * T, J, D)
            jm = joint_mask_bt.reshape(B * T, J)
            gm = ancestor_mask_bt.reshape(B * T, J, J)
            gh = graph_hop_bt.reshape(B * T, J, J)
            ge = graph_edge_bt.reshape(B * T, J, J)

            x2 = graph_blk(
                x2, x2, x2,
                gh, ge,
                mask=jm,
                tree_mask=gm,
            )
            x = x + x2.reshape(B, T, J, D)
            x = x * joint_mask_bt.unsqueeze(-1).float()

        return x

    def forward(
        self,
        pose,           # [B,T,J,3]
        rest_embed,     # [B,J,D]
        joint_t5embed,  # [B,J,Dt5]
        joint_mask,     # [B,J]
        ancestor_mask,     # [B,J,J]
        graph_hop,      # [B,J,J]
        graph_edge,     # [B,J,J]
    ):
        B, T, J, _ = pose.shape
        pose_feat = self.in_proj(pose)
        joint_sem = self.joint_t5proj(joint_t5embed).unsqueeze(1).expand(-1, T, -1, -1)
        rest_t = rest_embed.unsqueeze(1).expand(-1, T, -1, -1)

        x = pose_feat + joint_sem

        joint_mask_bt = joint_mask.unsqueeze(1).expand(-1, T, -1)
        ancestor_mask_bt = ancestor_mask.unsqueeze(1).expand(-1, T, -1, -1)
        graph_hop_bt = graph_hop.unsqueeze(1).expand(-1, T, -1, -1)
        graph_edge_bt = graph_edge.unsqueeze(1).expand(-1, T, -1, -1)

        x = x * joint_mask_bt.unsqueeze(-1).float()

        if self.use_rest_film:
            film_blocks = self.rest_film_blocks
        else:
            film_blocks = [None] * len(self.temporal_blocks)

        if self.use_graph:
            graph_blocks = self.graph_blocks
        else:
            graph_blocks = [None] * len(self.temporal_blocks)

        for temporal_blk, graph_blk, film_blk in zip(self.temporal_blocks, graph_blocks, film_blocks):
            if self.use_grad_checkpoint and self.training:
                def custom_forward(x_in):
                    return self._run_block(
                        temporal_blk, graph_blk, film_blk,
                        x_in, rest_t,
                        joint_mask_bt, ancestor_mask_bt, graph_hop_bt, graph_edge_bt
                    )
                x = checkpoint(custom_forward, x, use_reentrant=False)
            else:
                x = self._run_block(
                    temporal_blk, graph_blk, film_blk,
                    x, rest_t,
                    joint_mask_bt, ancestor_mask_bt, graph_hop_bt, graph_edge_bt
                )

        x = self.out_norm(x)
        x = x * joint_mask_bt.unsqueeze(-1).float()
        return x

# =========================================================
# Memory Encoder
# =========================================================
class MemoryEncoder(nn.Module):
    def __init__(
        self,
        q_dim=256,
        joint_embed_dim=768,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        use_rest_film=True,
        use_grad_checkpoint=False,
    ):
        super().__init__()
        self.pose_proj = nn.Linear(3, q_dim)
        self.rot_proj = nn.Linear(6, q_dim)
        self.joint_t5proj = nn.Linear(joint_embed_dim, q_dim)
        self.use_rest_film = use_rest_film
        self.use_grad_checkpoint = use_grad_checkpoint

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(q_dim),
                "graph": GraphMultiHeadAttention(
                    q_dim,
                    num_heads,
                    dropout=dropout,
                    use_tree_mask=(layer_idx % 2 == 0),
                ),
                "norm2": nn.LayerNorm(q_dim),
                "ffn": nn.Sequential(
                    nn.Linear(q_dim, q_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(q_dim * 4, q_dim),
                    nn.Dropout(dropout),
                ),
                "film": FiLMCondition(q_dim) if use_rest_film else nn.Identity(),
            })
            for layer_idx in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(q_dim)

    def _run_block(
        self,
        blk,
        x,
        rest_n,
        joint_mask_bn,
        ancestor_mask_bn,
        graph_hop_bn,
        graph_edge_bn,
    ):
        if self.use_rest_film:
            x = blk["film"](x, rest_n)
            x = x * joint_mask_bn.unsqueeze(-1).float()

        h = blk["norm1"](x)
        h = blk["graph"](
            h, h, h,
            graph_hop_bn, graph_edge_bn,
            mask=joint_mask_bn,
            tree_mask=ancestor_mask_bn,
        )
        x = x + h
        x = x * joint_mask_bn.unsqueeze(-1).float()

        h = blk["norm2"](x)
        h = blk["ffn"](h)
        x = x + h
        x = x * joint_mask_bn.unsqueeze(-1).float()
        return x

    def forward(
        self,
        memory_pose,    # [B,N,J,3]
        memory_rot6d,   # [B,N,J,6]
        rest_embed,     # [B,J,D]
        joint_t5embed,  # [B,J,Dt5]
        joint_mask,     # [B,J]
        ancestor_mask,     # [B,J,J]
        graph_hop,      # [B,J,J]
        graph_edge,     # [B,J,J]
    ):
        B, N, J, _ = memory_pose.shape

        pose_feat = self.pose_proj(memory_pose)
        rot_feat = self.rot_proj(memory_rot6d)
        joint_sem = self.joint_t5proj(joint_t5embed).unsqueeze(1).expand(-1, N, -1, -1)
        rest_n = rest_embed.unsqueeze(1).expand(-1, N, -1, -1)

        x = pose_feat + rot_feat + joint_sem

        x = x.reshape(B * N, J, -1)
        rest_n = rest_n.reshape(B * N, J, -1)

        joint_mask_bn = joint_mask.unsqueeze(1).expand(-1, N, -1).reshape(B * N, J)
        ancestor_mask_bn = ancestor_mask.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, J, J)
        graph_hop_bn = graph_hop.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, J, J)
        graph_edge_bn = graph_edge.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, J, J)

        x = x * joint_mask_bn.unsqueeze(-1).float()

        for blk in self.layers:
            if self.use_grad_checkpoint and self.training:
                def custom_forward(x_in):
                    return self._run_block(
                        blk, x_in, rest_n,
                        joint_mask_bn, ancestor_mask_bn, graph_hop_bn, graph_edge_bn
                    )
                x = checkpoint(custom_forward, x, use_reentrant=False)
            else:
                x = self._run_block(
                    blk, x, rest_n,
                    joint_mask_bn, ancestor_mask_bn, graph_hop_bn, graph_edge_bn
                )

        x = self.out_norm(x)
        x = x * joint_mask_bn.unsqueeze(-1).float()
        x = x.reshape(B, N, J, -1)
        return x

class RotDecoder(nn.Module):
    def __init__(
        self,
        q_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        temporal_window=2,
        joint_embed_dim=768,
        cond_mode="add",
        use_rest_film=True,
        use_grad_checkpoint=False,
        use_cross_layers=None,   # 前多少层使用 cross attn
    ):
        super().__init__()
        assert cond_mode in ["add", "concat"]
        self.cond_mode = cond_mode
        self.use_rest_film = use_rest_film
        self.use_grad_checkpoint = use_grad_checkpoint

        self.joint_t5proj = nn.Linear(joint_embed_dim, q_dim)

        if cond_mode == "concat":
            self.init_proj = nn.Linear(q_dim * 2, q_dim)
        else:
            self.init_proj = None

        if use_cross_layers is None:
            use_cross_layers = num_layers
        use_cross_layers = max(0, min(use_cross_layers, num_layers))
        self.use_cross_layers = use_cross_layers

        self.blocks = nn.ModuleList([
            RotDecoderBlock(
                q_dim=q_dim,
                num_heads=num_heads,
                dropout=dropout,
                temporal_window=temporal_window,
                use_tree_mask=(layer_idx % 2 == 0),
                use_rest_film=use_rest_film,
                use_cross_attn=(layer_idx < use_cross_layers),
            )
            for layer_idx in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(q_dim),
            nn.Linear(q_dim, q_dim),
            nn.GELU(),
            nn.Linear(q_dim, 6),
        )

    def _run_block(
        self,
        blk,
        x,
        rest_t,
        memory_feat,
        joint_mask,
        ancestor_mask,
        graph_hop,
        graph_edge,
    ):
        return blk(
            x=x,
            rest_t=rest_t,
            memory_feat=memory_feat,
            joint_mask=joint_mask,
            ancestor_mask=ancestor_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
        )

    def forward(
        self,
        query_feat,      # [B,T,J,D]
        memory_feat,     # [B,N,J,D]
        rest_embed,      # [B,J,D]
        joint_t5embed,   # [B,J,Dt5]
        joint_mask,      # [B,J]
        ancestor_mask,      # [B,J,J]
        graph_hop,       # [B,J,J]
        graph_edge,      # [B,J,J]
    ):
        B, T, J, D = query_feat.shape
        rest_t = rest_embed.unsqueeze(1).expand(-1, T, -1, -1)
        joint_sem = self.joint_t5proj(joint_t5embed).unsqueeze(1).expand(-1, T, -1, -1)

        if self.cond_mode == "add":
            x = query_feat + joint_sem
        elif self.cond_mode == "concat":
            x = self.init_proj(torch.cat([query_feat, joint_sem], dim=-1))
        else:
            raise ValueError(f"Unknown decoder cond mode: {self.cond_mode}")

        x = x * joint_mask.unsqueeze(1).unsqueeze(-1).float()

        for blk in self.blocks:
            if self.use_grad_checkpoint and self.training:
                def custom_forward(x_in):
                    return self._run_block(
                        blk, x_in, rest_t, memory_feat,
                        joint_mask, ancestor_mask, graph_hop, graph_edge
                    )
                x = checkpoint(custom_forward, x, use_reentrant=False)
            else:
                x = self._run_block(
                    blk, x, rest_t, memory_feat,
                    joint_mask, ancestor_mask, graph_hop, graph_edge
                )

        out = self.head(x.reshape(B * T, J, D)).reshape(B, T, J, 6)
        out = out * joint_mask.unsqueeze(1).unsqueeze(-1).float()
        return out

class Pose2RotMemoryRestModel(nn.Module):
    def __init__(
        self,
        q_dim=256,
        rest_layers=2,
        pose_layers=2,
        memory_layers=2,
        decoder_layers=4,
        num_heads=8,
        joint_embed_dim=768,
        temporal_window=2,
        temporal_dropout=0.1,
        decoder_cond_mode="add",
        pose_rest_film=True,
        memory_rest_film=True,
        decoder_rest_film=True,
        pose_use_graph=True,
        use_grad_checkpoint=False,
        decoder_use_cross_layers=None,
        num_memory=None,
    ):
        super().__init__()
        self.num_memory = num_memory
        
        self.rest_encoder = RestPoseEncoder(
            q_dim=q_dim,
            joint_embed_dim=joint_embed_dim,
            num_heads=num_heads,
            num_layers=rest_layers,
            dropout=temporal_dropout,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        self.pose_encoder = PoseEncoderLite(
            q_dim=q_dim,
            num_layers=pose_layers,
            num_heads=num_heads,
            dropout=temporal_dropout,
            temporal_window=temporal_window,
            joint_embed_dim=joint_embed_dim,
            use_rest_film=pose_rest_film,
            use_graph=pose_use_graph,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        self.memory_encoder = MemoryEncoder(
            q_dim=q_dim,
            joint_embed_dim=joint_embed_dim,
            num_heads=num_heads,
            num_layers=memory_layers,
            dropout=temporal_dropout,
            use_rest_film=memory_rest_film,
            use_grad_checkpoint=use_grad_checkpoint,
        )

        self.decoder = RotDecoder(
            q_dim=q_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=temporal_dropout,
            temporal_window=temporal_window,
            joint_embed_dim=joint_embed_dim,
            cond_mode=decoder_cond_mode,
            use_rest_film=decoder_rest_film,
            use_grad_checkpoint=use_grad_checkpoint,
            use_cross_layers=decoder_use_cross_layers,
        )

    def forward(self, batch, pose_override: Optional[torch.Tensor] = None):
        joint_mask = batch["joint_mask"].bool()
        ancestor_mask = batch["ancestor_mask"].bool()
        graph_hop = batch["graph_hop"]
        graph_edge = batch["graph_edge"]
        static_rot_joint_mask = batch["static_rot_joint_mask"].bool()

        pose = pose_override if pose_override is not None else batch["position"]                        # [B,T,J,3]
        memory_pose = batch["memory_pose"]              # [B,N,J,3]
        memory_rot6d = batch["memory_rot6d"]            # [B,N,J,6]
        ref_rot6d = batch["ref_rot6d_a"]                # [B,J,6]
        offset = batch["offset_a"]                      # [B,J,3]
        joint_t5embed = batch["joint_t5embed"]          # [B,J,768]

        if self.num_memory is None:
            pass
        elif self.num_memory == -1:
            # 用 reference frame 构造 1 个 memory: [B,J,C] -> [B,1,J,C]
            memory_pose = ref_pose.unsqueeze(1).contiguous()
            memory_rot6d = ref_rot6d.unsqueeze(1).contiguous()
        else:
            n_avail = memory_pose.shape[1]
            n_use = min(self.num_memory, n_avail)
            if n_use <= 0:
                raise ValueError(f"num_memory must be >= 1 or == -1, got {self.num_memory}")
            if n_use < n_avail:
                if self.training:
                    idx = torch.randperm(n_avail, device=memory_pose.device)[:n_use]
                else:
                    idx = torch.arange(n_use, device=memory_pose.device)
                memory_pose = memory_pose.index_select(1, idx)
                memory_rot6d = memory_rot6d.index_select(1, idx)

        rest_embed = self.rest_encoder(
            offset=offset,
            joint_t5embed=joint_t5embed,
            joint_mask=joint_mask,
            ancestor_mask=ancestor_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
        )

        q_feat = self.pose_encoder(
            pose=pose,
            rest_embed=rest_embed,
            joint_t5embed=joint_t5embed,
            joint_mask=joint_mask,
            ancestor_mask=ancestor_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
        )

        mem_feat = self.memory_encoder(
            memory_pose=memory_pose,
            memory_rot6d=memory_rot6d,
            rest_embed=rest_embed,
            joint_t5embed=joint_t5embed,
            joint_mask=joint_mask,
            ancestor_mask=ancestor_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
        )

        pred_rot6d = self.decoder(
            query_feat=q_feat,
            memory_feat=mem_feat,
            rest_embed=rest_embed,
            joint_t5embed=joint_t5embed,
            joint_mask=joint_mask,
            ancestor_mask=ancestor_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
        )

        static_rot_joint_mask_4d = static_rot_joint_mask.unsqueeze(1).unsqueeze(-1)
        ref_rot6d_4d = ref_rot6d.unsqueeze(1).expand_as(pred_rot6d)
        pred_rot6d = pred_rot6d * (~static_rot_joint_mask_4d) + ref_rot6d_4d * static_rot_joint_mask_4d

        return {
            "pred_rot6d": pred_rot6d,
            "rest_embed": rest_embed,
            "q_feat": q_feat,
            "mem_feat": mem_feat,
        }

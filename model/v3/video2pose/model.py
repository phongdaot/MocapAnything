### model.py ###
import torch
import torch.nn as nn

from .ref_encoder import RefQueryEncoder
from .temporal_trunk import Video2PoseModelSliding


class RefGuidedVideo2PoseModel(nn.Module):
    """
    Full Video2Pose v3 model: reference-frame query encoder + temporal trunk.

    The reference encoder fuses the reference pose (+ optional joint text
    embedding) with the reference image into per-joint query features.
    The temporal trunk consumes those queries together with per-frame
    image embeddings and predicts per-frame 3D joint positions.

    Joints flagged by ``static_pos_joint_mask`` are overridden with the
    reference position so they remain fixed across time.
    """

    def __init__(
        self,
        num_layers=12,
        q_dim=256,
        img_dim=1024,
        num_joints=150,
        num_heads=8,
        ref_layers=4,
        temporal_window=2,
        use_joint_embed=False,
        use_graph_ref_inner=False,
        use_graph_temporal_inner=False,
        dropout=0.1,
    ):
        super().__init__()

        self.ref_encoder = RefQueryEncoder(
            q_dim=q_dim,
            img_dim=img_dim,
            num_heads=num_heads,
            num_layers=ref_layers,
            use_joint_embed=use_joint_embed,
            use_graph_ref_inner=use_graph_ref_inner,
            dropout=dropout,
        )

        self.temporal_model = Video2PoseModelSliding(
            num_layers=num_layers,
            q_dim=q_dim,
            img_dim=img_dim,
            num_joints=num_joints,
            num_heads=num_heads,
            temporal_window=temporal_window,
            use_graph_temporal_inner=use_graph_temporal_inner,
            dropout=dropout,
        )

    def forward(self, batch):
        image_embed = batch["image_embed"]               # [B,F,P,img_dim]
        ref_pos = batch["ref_position"]                  # [B,J,3]
        ref_img = batch["ref_image_embed"]               # [B,P,img_dim]
        joint_mask = batch["joint_mask"].bool()          # [B,J]
        graph_hop = batch["graph_hop"]                   # [B,J,J]
        graph_edge = batch["graph_edge"]                 # [B,J,J]
        joint_t5embed = batch["joint_t5embed"]           # [B,J,joint_embed_dim]
        ancestor_mask = batch["ancestor_mask"].bool()    # [B,J,J]

        ref_query = self.ref_encoder(
            ref_position=ref_pos,
            ref_image_embed=ref_img,
            joint_mask=joint_mask,
            graph_hop=graph_hop,
            graph_edge=graph_edge,
            joint_t5embed=joint_t5embed,
            tree_mask=ancestor_mask,
        )

        F = image_embed.shape[1]

        joint_mask_t = joint_mask.unsqueeze(1).expand(-1, F, -1)
        graph_hop_t = graph_hop.unsqueeze(1).expand(-1, F, -1, -1)
        graph_edge_t = graph_edge.unsqueeze(1).expand(-1, F, -1, -1)
        ancestor_mask_t = ancestor_mask.unsqueeze(1).expand(-1, F, -1, -1)

        pose_pred = self.temporal_model(
            ref_query=ref_query,
            cond_img=image_embed,
            joint_mask=joint_mask_t,
            graph_hop=graph_hop_t,
            graph_edge=graph_edge_t,
            tree_mask=ancestor_mask_t,
        )

        static_pos_joint_mask = batch["static_pos_joint_mask"].bool()
        ref_pos_4d = ref_pos.unsqueeze(1).expand(-1, pose_pred.shape[1], -1, -1)
        static_mask_4d = static_pos_joint_mask.unsqueeze(1).unsqueeze(-1)

        pose_pred = pose_pred * (~static_mask_4d) + ref_pos_4d * static_mask_4d
        return pose_pred

import torch.nn as nn
import torch
import inspect
from typing import Optional, Dict, Any, Tuple
from utils.config_utils import instantiate_from_config


def build_input(
    gt_pose: torch.Tensor,
    pred_pose: torch.Tensor,
    mode: str,
    pred_prob: float,
    detach_pred: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mode = mode.lower()

    if mode == "gt":
        return gt_pose, {
            "pred_prob": 0.0,
            "used_pred_ratio": 0.0,
            "used_gt_ratio": 1.0,
        }

    pred_pose_used = pred_pose.detach() if detach_pred else pred_pose

    if mode == "pred":
        return pred_pose_used, {
            "pred_prob": 1.0,
            "used_pred_ratio": 1.0,
            "used_gt_ratio": 0.0,
        }

    assert mode == "mix", f"Unknown pose_source_mode: {mode}"

    B = gt_pose.shape[0]
    device = gt_pose.device
    selector = (torch.rand(B, device=device) < pred_prob).float()
    selector_4d = selector.view(B, 1, 1, 1)

    pose_for_rot = selector_4d * pred_pose_used + (1.0 - selector_4d) * gt_pose
    used_pred_ratio = float(selector.mean().item())

    return pose_for_rot, {
        "pred_prob": float(pred_prob),
        "used_pred_ratio": used_pred_ratio,
        "used_gt_ratio": 1.0 - used_pred_ratio,
    }


class Video2Pose2RotModel(nn.Module):
    def __init__(
        self,
        # -------- video2pos --------
        v2p_cfg: dict,
        # -------- pos2rot --------
        p2r_cfg: dict,
    ):
        super().__init__()

        # stage1: video -> pose
        self.video2pos = instantiate_from_config(v2p_cfg)
        self.video2pos_accepts_attention_kwargs = (
            "attention_kwargs" in inspect.signature(self.video2pos.forward).parameters
        )

        # stage2: pose -> rot
        self.pos2rot = instantiate_from_config(p2r_cfg)

    def forward(
        self,
        batch,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        pose_source_mode: str = "pred",
        pose_mix_prob: float = 1.0,
        detach_pred_pose_for_rot: bool = False,
    ):
        gt_position = batch["position"]   # [B,T,J,3]

        # -------------------------
        # stage1: video2pos
        # -------------------------
        if attention_kwargs is None or not self.video2pos_accepts_attention_kwargs:
            pred_position = self.video2pos(batch)
        else:
            pred_position = self.video2pos(
                batch,
                attention_kwargs=attention_kwargs,
            )

        # -------------------------
        # choose pose source for stage2
        # -------------------------
        pose_for_rot, pose_source_info = build_input(
            gt_pose=gt_position,
            pred_pose=pred_position,
            mode=pose_source_mode,
            pred_prob=pose_mix_prob,
            detach_pred=detach_pred_pose_for_rot,
        )

        # -------------------------
        # stage2: pos2rot
        # -------------------------
        pos2rot_out = self.pos2rot(
            batch=batch,
            pose_override=pose_for_rot,
        )

        return {
            "pred_position": pred_position,
            "pose_for_rot": pose_for_rot,
            "pred_rot6d": pos2rot_out["pred_rot6d"],
            "rest_embed": pos2rot_out.get("rest_embed"),
            "q_feat": pos2rot_out.get("q_feat"),
            "mem_feat": pos2rot_out.get("mem_feat"),
            "pose_source_info": pose_source_info,
        }

### loss.py ###
import torch
from typing import Callable, Dict, Any
from .common import apply_joint_mask
from .rotation import rot6d_to_fk_positions, rot6d_to_rotmat_batch
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch.nn as nn
import roma

def comp_geodesic(opt_mat: torch.Tensor, gt_mat: torch.Tensor):
    geodesic = roma.rotmat_geodesic_distance(
        opt_mat, gt_mat
    ).mean(-1) * 180 / np.pi  # in degrees
    geodesic = geodesic.reshape(len(geodesic), -1)
    geodesic = torch.mean(geodesic, dim=-1)
    return geodesic

def angle_L1(pred_rot6d, gt_rot6d, mask=None):
    # L1 mean angle error (degree)
    if isinstance(pred_rot6d, torch.Tensor):
        pred_rot6d = pred_rot6d.detach().cpu().numpy()
    if isinstance(gt_rot6d, torch.Tensor):
        gt_rot6d = gt_rot6d.detach().cpu().numpy()
    if mask is not None:
        pred_rot6d, gt_rot6d = apply_joint_mask(pred_rot6d, gt_rot6d, mask)
    pred_mat = rot6d_to_rotmat_batch(pred_rot6d)
    gt_mat = rot6d_to_rotmat_batch(gt_rot6d)
    angle_error = comp_geodesic(torch.from_numpy(pred_mat), torch.from_numpy(gt_mat)).cpu().numpy()
    return np.mean(angle_error)

def angle_velocity_L1(pred_rot6d, gt_rot6d, mask=None):
    B, F, J, _ = pred_rot6d.shape
    if isinstance(pred_rot6d, torch.Tensor):
        pred_rot6d = pred_rot6d.detach().cpu().numpy()
    if isinstance(gt_rot6d, torch.Tensor):
        gt_rot6d = gt_rot6d.detach().cpu().numpy()
    if mask is not None:
        pred_rot6d, gt_rot6d = apply_joint_mask(pred_rot6d, gt_rot6d, mask)
    pred_mat = rot6d_to_rotmat_batch(pred_rot6d)
    gt_mat = rot6d_to_rotmat_batch(gt_rot6d)
    # angle_error = comp_geodesic(torch.from_numpy(pred_mat), torch.from_numpy(gt_mat)).cpu().numpy()
    
    # relative rotation (velocity)
    pred_vel_matrix = pred_mat[:, 1:] @ pred_mat[:, :-1].transpose(0, 1, 2, 4, 3)
    gt_vel_matrix   = gt_mat[:, 1:] @ gt_mat[:, :-1].transpose(0, 1, 2, 4, 3)

    # error
    velocity_error_matrix = pred_vel_matrix @ gt_vel_matrix.transpose(0, 1, 2, 4, 3)

    # to rotvec
    vel_rotvec = R.from_matrix(
        velocity_error_matrix.reshape(-1, 3, 3)
    ).as_rotvec()

    # print('vel_rotvec shape:', vel_rotvec.shape)

    # compute magnitude
    return np.linalg.norm(vel_rotvec, axis=-1).mean()

def rot6d_vel_loss(pred_rot6d, gt_rot6d, joint_mask, loss_fn):
    # pred/gt: [B,F,J,6], mask: [B,J]
    if pred_rot6d.size(1) < 2:
        return torch.tensor(0.0, device=pred_rot6d.device)
    v_pred = pred_rot6d[:, 1:] - pred_rot6d[:, :-1]  # [B,F-1,J,6]
    v_gt   = gt_rot6d[:, 1:] - gt_rot6d[:, :-1]
    return masked_loss(v_pred, v_gt, joint_mask, loss_fn)

def rot6d_acc_loss(pred_rot6d, gt_rot6d, joint_mask, loss_fn):
    # 二阶差分（加速度/jerk），可选
    if pred_rot6d.size(1) < 3:
        return torch.tensor(0.0, device=pred_rot6d.device)
    a_pred = pred_rot6d[:, 2:] - 2 * pred_rot6d[:, 1:-1] + pred_rot6d[:, :-2]  # [B,F-2,J,6]
    a_gt   = gt_rot6d[:, 2:] - 2 * gt_rot6d[:, 1:-1] + gt_rot6d[:, :-2]
    return masked_loss(a_pred, a_gt, joint_mask, loss_fn)

def expand_joint_mask_like(
    joint_mask: torch.Tensor,   # [B,J] or [B,T,J]
    target: torch.Tensor,       # [B,T,J,C] or [B,T,J]
) -> torch.Tensor:
    """
    Expands the joint_mask to match the shape of target for element-wise masking.
     - If joint_mask is [B,J], it will be expanded to [B,T,J,C] or [B,T,J] by unsqueezing and expanding.
     - If joint_mask is [B,T,J], it will be expanded to [B,T,J,C] by unsqueezing and expanding if target has 4 dims, or used as is if target has 3 dims.
     - The resulting mask will be boolean and can be used for element-wise multiplication with the target or loss tensor.
     - This function ensures that the mask is correctly broadcasted across the time and channel dimensions as needed.
    """
    assert joint_mask.dim() in [2, 3], f"joint_mask dim must be 2 or 3, got {joint_mask.dim()}"
    assert target.dim() in [3, 4], f"target dim must be 3 or 4, got {target.dim()}"

    if target.dim() == 4:
        # target: [B,T,J,C]
        if joint_mask.dim() == 2:
            mask = joint_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,J,1]
            mask = mask.expand(target.shape[0], target.shape[1], target.shape[2], target.shape[3])
        else:
            mask = joint_mask.unsqueeze(-1)               # [B,T,J,1]
            mask = mask.expand(target.shape[0], target.shape[1], target.shape[2], target.shape[3])
    else:
        # target: [B,T,J]
        if joint_mask.dim() == 2:
            mask = joint_mask.unsqueeze(1)                # [B,1,J]
            mask = mask.expand(target.shape[0], target.shape[1], target.shape[2])
        else:
            mask = joint_mask                             # [B,T,J]

    return mask.bool()

def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    joint_mask: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Generic masked loss.

    Args:
        pred: predicted tensor of shape [B, F, J, C]
        target: ground truth tensor of shape [B, F, J, C]
        joint_mask: joint validity mask of shape [B, J] or [B, F, J]
        loss_fn: element-wise loss function returning tensor of shape [B, F, J, C]

    Returns:
        Mean masked loss across the batch.
    """
    loss_raw = loss_fn(pred, target)
    mask = expand_joint_mask_like(joint_mask, loss_raw).to(loss_raw.dtype)
    loss_masked = loss_raw * mask

    reduce_dims = tuple(range(1, loss_raw.dim()))
    valid_count = mask.sum(dim=reduce_dims).clamp(min=1.0)
    loss_per_sample = loss_masked.sum(dim=reduce_dims) / valid_count
    return loss_per_sample.mean()

def masked_mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    joint_mask: torch.Tensor,
) -> torch.Tensor:
    """
    pred, target: [B,T,J,3]
    joint_mask:   [B,J] or [B,F,J]
    """
    dist_err = torch.norm(pred - target, dim=-1)    # [B,T,J]
    mask = expand_joint_mask_like(joint_mask, dist_err).to(dist_err.dtype)
    valid_count = mask.sum(dim=(1, 2)).clamp(min=1.0)
    per_sample = (dist_err * mask).sum(dim=(1, 2)) / valid_count
    return per_sample.mean()


def masked_mpjve(
    pred: torch.Tensor,
    target: torch.Tensor,
    joint_mask: torch.Tensor,
) -> torch.Tensor:
    """
    MPJVE: Mean Per Joint Velocity Error.

    Computes the Euclidean error between joint velocity vectors
    of consecutive frames.

    Args:
        pred: predicted joint positions [B, F, J, 3]
        target: ground truth joint positions [B, F, J, 3]
        joint_mask: joint validity mask [B, J] or [B, F, J]

    Returns:
        Scalar MPJVE value.
    """

    if pred.size(1) < 2:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]

    if joint_mask.dim() == 2:
        mask_t = joint_mask.unsqueeze(1).expand(-1, pred_diff.size(1), -1)
    else:
        mask_t = joint_mask[:, 1:] & joint_mask[:, :-1]

    dist_err = torch.norm(pred_diff - target_diff, dim=-1)   # [B,T-1,J]
    mask = expand_joint_mask_like(mask_t, dist_err).to(dist_err.dtype)
    valid_count = mask.sum(dim=(1, 2)).clamp(min=1.0)
    per_sample = (dist_err * mask).sum(dim=(1, 2)) / valid_count
    return per_sample.mean()


def get_loss_fn(loss_type: str):
    
    if loss_type == "l1":
        return nn.L1Loss(reduction="none")
    elif loss_type == "l2":
        return nn.MSELoss(reduction="none")
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss(reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def pos_vel_loss(pred_pos, gt_pos, joint_mask, loss_fn):
    if pred_pos.size(1) < 2:
        return torch.tensor(0.0, device=pred_pos.device, dtype=pred_pos.dtype)

    v_pred = pred_pos[:, 1:] - pred_pos[:, :-1]
    v_gt   = gt_pos[:, 1:] - gt_pos[:, :-1]

    if joint_mask.dim() == 2:
        mask_t = joint_mask.unsqueeze(1).expand(-1, v_pred.size(1), -1)
    else:
        mask_t = joint_mask[:, 1:] & joint_mask[:, :-1]

    return masked_loss(v_pred, v_gt, mask_t, loss_fn)


# =========================================================
# branch-specific losses
# =========================================================
def compute_pose_losses(
    model_out: Dict[str, Any],
    batch: Dict[str, Any],
    weight_cfg: Dict[str, float],
    pose_criterion,
    pose_vel_criterion,
):
    pred_position = model_out["pred_position"]
    gt_position = batch["position"]

    joint_mask = batch["joint_mask"].bool()
    static_pos_mask = batch["static_pos_joint_mask"].bool()
    pos_joint_mask = joint_mask & (~static_pos_mask)

    loss_dict = {}

    loss_pose = masked_loss(pred_position, gt_position, pos_joint_mask, pose_criterion)
    loss_dict["loss_pose"] = loss_pose
    total = weight_cfg["pose_wt"] * loss_pose

    if weight_cfg["pose_vel_wt"] > 0:
        loss_pose_vel = pos_vel_loss(pred_position, gt_position, pos_joint_mask, pose_vel_criterion)
        loss_dict["loss_pose_vel"] = loss_pose_vel
        total = total + weight_cfg["pose_vel_wt"] * loss_pose_vel

    loss_dict["pose_total_loss"] = total
    return total, loss_dict

def compute_rot_loss(model_out, batch, weight_cfg, rot_criterion, vel_criterion, acc_criterion):
    joint_mask = batch["joint_mask"].bool()
    static_rot_mask = batch["static_rot_joint_mask"].bool()
    static_pos_mask = batch["static_pos_joint_mask"].bool()

    rot_joint_mask_for_loss = joint_mask & (~static_rot_mask)
    pos_joint_mask_for_loss = joint_mask & (~static_pos_mask)

    gt_rot6d = batch["rot6d_a"]
    target_pose = batch["position"]

    parents = batch["parent_a"]
    offsets = batch["offset_a"]
    global_scales = batch["global_scale"]

    pred_rot6d = model_out["pred_rot6d"]

    loss_dict = {}

    loss_rot = masked_loss(pred_rot6d, gt_rot6d, rot_joint_mask_for_loss, rot_criterion)
    loss_dict["loss_rot"] = loss_rot
    total_loss = weight_cfg["rot_wt"] * loss_rot

    if weight_cfg["vel_wt"] > 0:
        loss_vel = rot6d_vel_loss(pred_rot6d, gt_rot6d, rot_joint_mask_for_loss, vel_criterion)
        loss_dict["loss_vel"] = loss_vel
        total_loss = total_loss + weight_cfg["vel_wt"] * loss_vel

    if weight_cfg["acc_wt"] > 0:
        loss_acc = rot6d_acc_loss(pred_rot6d, gt_rot6d, rot_joint_mask_for_loss, acc_criterion)
        loss_dict["loss_acc"] = loss_acc
        total_loss = total_loss + weight_cfg["acc_wt"] * loss_acc

    if weight_cfg["fk_wt"] > 0:
        pred_pos = rot6d_to_fk_positions(pred_rot6d, offsets, parents, global_scales)
        loss_fk = masked_loss(
            pred_pos,
            target_pose,
            pos_joint_mask_for_loss,
            nn.SmoothL1Loss(reduction='none')
        )
        loss_dict["loss_fk"] = loss_fk
        total_loss = total_loss + weight_cfg["fk_wt"] * loss_fk

    if weight_cfg["root_wt"] > 0:
        root_mask = torch.zeros_like(joint_mask)
        root_mask[:, 0] = True
        loss_root = masked_loss(pred_rot6d, gt_rot6d, root_mask, rot_criterion)
        loss_dict["loss_root"] = loss_root
        total_loss = total_loss + weight_cfg["root_wt"] * loss_root

    loss_dict["total_loss"] = total_loss
    return total_loss, loss_dict

def compute_joint_total_loss(
    model_out: Dict[str, Any],
    batch: Dict[str, Any],
    weight_cfg: Dict[str, float],
    pose_criterion,
    pose_vel_criterion,
    rot_criterion,
    vel_criterion,
    acc_criterion,
):
    pose_total, pose_loss_dict = compute_pose_losses(
        model_out=model_out,
        batch=batch,
        weight_cfg=weight_cfg,
        pose_criterion=pose_criterion,
        pose_vel_criterion=pose_vel_criterion,
    )

    rot_total, rot_loss_dict = compute_rot_loss(
        model_out=model_out,
        batch=batch,
        weight_cfg=weight_cfg,
        rot_criterion=rot_criterion,
        vel_criterion=vel_criterion,
        acc_criterion=acc_criterion,
    )

    total_loss = pose_total + rot_total

    loss_dict = {}
    loss_dict.update(pose_loss_dict)
    loss_dict.update(rot_loss_dict)
    loss_dict["total_loss"] = total_loss
    return total_loss, loss_dict
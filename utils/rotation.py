### rotation.py ###
from animatrix.data.utils.transforms3d import rotation_6d_to_matrix, matrix_to_euler_angles
from animatrix.data.utils.mesh import extract_mesh_from_bvh
from animatrix.data.structure import bvh as BVH
import animatrix.data.structure.animation as animation

import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


def rot6d_to_fk_positions(rot6d, offsets, parents, global_scales):
    """
    Convert per-joint 6D rotations into FK joint positions.

    Args:
        rot6d: Tensor of shape [B, T, J, 6]
            Predicted joint rotations in 6D representation.
            B = batch size
            T = number of frames
            J = number of joints

        offsets: Tensor of shape [B, J, 3]
            Rest-pose joint offsets / bone offsets for each sample.

        parents: Tensor of shape [B, J] or iterable of length B
            Parent index array for each sample skeleton.
            For each joint j, parents[..., j] gives its parent joint index.

        global_scales: Tensor of shape [B]
            Global scale per sample. Currently unused because normalization
            is commented out.

    Returns:
        pred_positions: Tensor of shape [B, T, J, 3]
            FK joint positions for each frame, root-centered by subtracting
            joint 0 position from all joints.
    """
    pred_rotmat = rot6d_to_rotmat_tensor(rot6d)
    pred_positions = []

    for pred_rot, offset, parent in zip(pred_rotmat, offsets, parents):
        _, pred_pos = bvh_forward(
            pred_rot,
            torch.zeros((pred_rot.shape[0], 3), device=pred_rot.device, dtype=pred_rot.dtype),
            offset,
            parent,
        )
        pred_positions.append(pred_pos)

    pred_positions = torch.stack(pred_positions, dim=0)
    pred_positions = pred_positions - pred_positions[:, :, 0:1, :]
    # pred_positions = pred_positions / global_scales.view(-1, 1, 1, 1)
    return pred_positions

def rot6d_to_rotmat_tensor(rot_6d):
    x = rot_6d.view(-1, 6)
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    rotmat = torch.stack([b1, b2, b3], dim=-1)
    rotmat = rotmat.view(*rot_6d.shape[:-1], 3, 3)
    return rotmat

def rot6d_to_rotmat_batch(rot_6d):
    if isinstance(rot_6d, torch.Tensor):
        x = rot_6d.view(-1, 6)
        a1 = x[:, 0:3]
        a2 = x[:, 3:6]
        b1 = F.normalize(a1, dim=1)
        b2 = F.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        rotmat = torch.stack([b1, b2, b3], dim=-1)
        rotmat = rotmat.view(*rot_6d.shape[:-1], 3, 3)
        return rotmat.cpu().numpy()
    else:
        x = rot_6d.reshape(-1, 6)
        a1 = x[:, 0:3]
        a2 = x[:, 3:6]
        b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
        b2 = a2 - (b1 * a2).sum(1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
        b3 = np.cross(b1, b2)
        rotmat = np.stack([b1, b2, b3], axis=-1)
        rotmat = rotmat.reshape(*rot_6d.shape[:-1], 3, 3)
        return rotmat

def rotmat_to_euler_deg(rotmat, order='zyx'):
    shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    r = R.from_matrix(rotmat_flat)
    euler = r.as_euler(order, degrees=True)
    euler = euler.reshape(*shape, 3)
    return euler

def bvh_forward(
    rot_mat: torch.Tensor, transl: torch.Tensor, bvh_offset: torch.Tensor,
    parents: torch.Tensor
):
    """
    Compute global joint rotations and positions from local BVH rotations using forward kinematics.

    Args:
        rot_mat (torch.Tensor): Local rotation matrices in shape (T, J, 3, 3).
        transl (torch.Tensor): Root joint translation velocities, shape (T, 3).
        bvh_offset (torch.Tensor): Joint offsets in rest pose, shape (J, 3).
        parents (torch.Tensor): Parent indices of joints, shape (J,).

    Returns:
        torch.Tensor: Global rotation matrices, shape (T, J, 3, 3).
        torch.Tensor: Global joint positions, shape (T, J, 3).
    """
    device = rot_mat.device
    seq_len, num_joints, _, _ = rot_mat.shape
    offsets = bvh_offset.unsqueeze(0).repeat(seq_len, 1, 1)  # T,J,3
    root_pos = offsets[:, 0, :] + transl
    joint_pos_init = offsets[:, 1:, :]
    pos_init = torch.cat((root_pos.unsqueeze(1), joint_pos_init), dim=1)
    translation_column = pos_init.unsqueeze(-1)
    bottom_row = torch.full(
        (seq_len, num_joints, 1, 4), 0., device=device, dtype=rot_mat.dtype
    )
    bottom_row[:, :, :, 3] = 1.0
    top_half = torch.cat((rot_mat.clone(), translation_column), dim=3)
    transfrom_mat = torch.cat((top_half, bottom_row), dim=2)

    global_mats_list = []
    global_root_mat = transfrom_mat[:, 0]
    global_mats_list.append(global_root_mat)

    for i in range(1, num_joints):
        parent_mat = global_mats_list[parents[i]]
        current_joint_local_mat = transfrom_mat[:, i]
        current_global_mat = parent_mat @ current_joint_local_mat
        global_mats_list.append(current_global_mat)

    global_mat = torch.stack(global_mats_list, dim=1)
    pred_pos = global_mat[:, :, :3, 3] / global_mat[:, :, 3:, 3]

    return global_mat[:, :, :3, :3], pred_pos


def bvh_to_joints_rot(bvh_path, random_rest=False):
    """
    Load BVH file and convert to joint rotations in 6D representation.
    """
    anim, _, _ = BVH.load(bvh_path)
    if random_rest:
        anim.random_rest_pose()
    local_xforms = animation.transforms_local(anim)
    rot_matrices = local_xforms[:, :, :3, :3]
    rot_matrices_flat = rot_matrices.reshape(rot_matrices.shape[0], rot_matrices.shape[1], 9)
    rot6d = rot_matrices_flat[:, :, :6].astype(np.float32)
    return rot6d, anim

### train_utils.py ###
import torch
import numpy as np
from .logger import logger
import glob
import os
import re
import torch.distributed as dist

# ==== checkpoint相关工具 ====
def save_checkpoint_with_epoch(model: torch.nn, name, optimizer, epoch, base_dir):
    ckpt_path = os.path.join(base_dir, f"{name}_ckpt_epoch{epoch}.pt")
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")
    return ckpt_path


def cleanup_old_checkpoints(base_dir, name, max_ckpt=5):
    ckpt_list = sorted(glob.glob(os.path.join(base_dir, f"{name}_ckpt_epoch*.pt")), key=os.path.getmtime)
    if len(ckpt_list) > max_ckpt:
        for ckpt in ckpt_list[:-max_ckpt]:
            os.remove(ckpt)
            logger.info(f"Deleted old checkpoint: {ckpt}")


def find_latest_ckpt(base_dir, name):
    ckpt_files = glob.glob(os.path.join(base_dir, f"{name}_ckpt_epoch*.pt"))
    if not ckpt_files:
        return None

    def epoch_num(path):
        m = re.search(r'epoch(\d+)', path)
        return int(m.group(1)) if m else 0

    ckpt_files.sort(key=epoch_num)
    return ckpt_files[-1]


def load_checkpoint(model, optimizer, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch']
    return start_epoch


def load_partial_pretrain(model, pretrain_path):
    logger.info(f"==> Loading pretrain weights from {pretrain_path}")
    pretrain = torch.load(pretrain_path, map_location='cpu')
    pretrain_dict = pretrain['model_state'] if 'model_state' in pretrain else pretrain
    model_dict = model.state_dict()
    # 只保留能对上的参数
    load_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    # 打印加载和未加载的key
    logger.info(f"Loaded params: {list(load_dict.keys())}")
    logger.info(f"Missed params: {list(set(model_dict.keys()) - set(load_dict.keys()))}")
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
### common.py ###
import random

import numpy as np
import os
import torch
import trimesh
from PIL import Image
import .bvh as BVH
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from preprocess.image_process import prepare_image
from .finder import *
from .npy2bvh import convert_npy_to_bvh
from .rotation import rot6d_to_fk_positions, bvh_to_joints_rot, rot6d_to_rotmat_batch
from .mesh import batch_rigid_transform

def parent_to_kinematic_tree(parents: list):

    dfs_len = len(parents)
    child_lists = [[] for i in range(dfs_len)]
    for idx in range(dfs_len):
        parent = parents[idx]
        if parent != -1:
            child_lists[parent].append(idx)

    trees = []
    crt_chain = []

    def dfs(idx):
        if len(child_lists[idx]) == 0:
            crt_chain.append(idx)
            trees.append(crt_chain.copy())
            crt_chain.clear()

        for child in child_lists[idx]:
            crt_chain.append(idx)
            dfs(child)

    dfs(0)
    # print(crt_chain)
    return trees

def apply_joint_mask(pred_rot6d, gt_rot6d, mask):
    """
    根据关节mask筛选有效的rot6d数据。
    输入:
        pred_rot6d: (B, T, J, 6)
        gt_rot6d:   (B, T, J, 6)
        mask:       (B, J)
    输出:
        pred_valid: (B, T, J_valid, 6)
        gt_valid:   (B, T, J_valid, 6)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = mask.astype(bool)

    # 构造布尔索引
    B, T, J, D = pred_rot6d.shape
    pred_valid_list, gt_valid_list = [], []

    for b in range(B):
        valid_joints = np.where(mask[b])[0]
        pred_valid_list.append(pred_rot6d[b, :, valid_joints, :])
        gt_valid_list.append(gt_rot6d[b, :, valid_joints, :])

    pred_valid = np.stack(pred_valid_list, axis=0)
    gt_valid = np.stack(gt_valid_list, axis=0)
    return pred_valid, gt_valid

def load_surface_from_glb_folder(glb_folder, num_points=1024):
    """
    Input: path to glb_folder
    Returns:
        - surface_pts: (N, num_points, 3)
        - normal_pts:  (N, num_points, 3)
        - glb_files:   [glb_path1, glb_path2, ...] (in the same order as the outputs)
    """
    glb_files = sorted([
        os.path.join(glb_folder, f)
        for f in os.listdir(glb_folder)
        if f.lower().endswith('.glb')
    ])
    surface_pts = []
    normal_pts = []

    for glb_file in glb_files:
        mesh = trimesh.load(glb_file, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        pts, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        normals = mesh.face_normals[face_idx]
        surface_pts.append(pts.astype(np.float32))
        normal_pts.append(normals.astype(np.float32))

    surface_pts = np.stack(surface_pts, axis=0)
    normal_pts = np.stack(normal_pts, axis=0)
    return surface_pts, normal_pts, glb_files

def extract_and_compare_image_features_with_rmbg(
    image_folder,
    pipe,
    rmbg_net,
    device="cuda",
    dtype="float16",
    check_feature_npz=None,
):
    """
    Process all images in `image_folder` according to the same pipeline used in training/saving:
      - prepare_image (including RMBG, white background, crop, etc.)
      - feature_extractor_dinov2 + image_encoder_dinov2
      - output features and optionally compare with check_feature_npz
    Args:
      - pipe: TripoSGPipeline instance
      - rmbg_net: loaded RMBG model
      - device, dtype: inference device and precision
      - check_feature_npz: optional .npz path to compare
    """
    img_files = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.lower().endswith((".jpg", ".png"))
    ])
    if not img_files:
        raise ValueError(f"No images found in {image_folder}")

    # 1. Preprocess images using prepare_image + RMBG
    img_pil_list = []
    for image_file in img_files:
        try:
            pil = prepare_image(
                image_file, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net
            )
            img_pil_list.append(pil)
        except Exception as e:
            print(f"Failed to process {image_file}: {e}")

    # 2. Extract features using pipeline's DINO model
    all_embeds = []
    batch_size = 512
    for i in range(0, len(img_pil_list), batch_size):
        image_chunk = img_pil_list[i:i+batch_size]
        with torch.no_grad():
            pixel_values = pipe.feature_extractor_dinov2(
                image_chunk, return_tensors="pt"
            ).pixel_values.to(device, dtype=getattr(torch, dtype))
            embed = pipe.image_encoder_dinov2(pixel_values).last_hidden_state
        all_embeds.append(embed.cpu())
    image_embeds = torch.cat(all_embeds, dim=0).numpy()

    print(f"[Info] Extracted image_embed shape: {image_embeds.shape}")

    # 3. Compare features
    if check_feature_npz is not None:
        npz = np.load(check_feature_npz)
        if "image_embed" not in npz:
            raise ValueError(f"image_embed not found in {check_feature_npz}")
        gt_embed = npz["image_embed"]
        print(f"[Info] GT image_embed shape: {gt_embed.shape}")

        if image_embeds.shape != gt_embed.shape:
            print(f"[Compare] shape mismatch: {image_embeds.shape} vs {gt_embed.shape}")
        else:
            is_close = np.allclose(image_embeds, gt_embed, atol=1e-6)
            diff = np.abs(image_embeds - gt_embed)
            print(f"[Compare] allclose: {is_close}")
            print(f"[Compare] max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}, min diff: {diff.min():.6f}")
    return image_embeds

def save_pose_npy(save_dir, name, pred, gt):
    os.makedirs(save_dir, exist_ok=True)

    pred_path = os.path.join(save_dir, f"{name}_pos_pred.npy")
    gt_path = os.path.join(save_dir, f"{name}_pos_gt.npy")

    np.save(pred_path, pred)
    np.save(gt_path, gt)

    return pred_path, gt_path


def save_rot_npy(save_dir, name, pred, gt):
    os.makedirs(save_dir, exist_ok=True)

    pred_path = os.path.join(save_dir, f"{name}_rot_pred.npy")
    gt_path = os.path.join(save_dir, f"{name}_rot_gt.npy")

    np.save(pred_path, pred)
    np.save(gt_path, gt)

    return pred_path, gt_path

def visualize_joint_sample(
    save_dir,
    species_name,
    pred_pos,
    gt_pos,
    pred_rot,
    gt_rot,
):
    """
    一个 sample 输出四类结果：
        pos_pred / pos_gt
        rot_pred / rot_gt (-> bvh)
    """

    pos_pred_path, pos_gt_path = save_pose_npy(
        save_dir, species_name, pred_pos, gt_pos
    )

    rot_pred_path, rot_gt_path = save_rot_npy(
        save_dir, species_name, pred_rot, gt_rot
    )

    # 转 BVH
    convert_npy_to_bvh(rot_pred_path, species_name)
    convert_npy_to_bvh(rot_gt_path, species_name)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

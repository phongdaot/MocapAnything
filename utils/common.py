### common.py ###
import random

import numpy as np
import os
import torch
import trimesh
from PIL import Image
import animatrix.data.structure.bvh as BVH
from animatrix.data.visualizer.skeleton_visualizer import parent_to_kinematic_tree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from preprocess.image_process import prepare_image
from .finder import *
from .npy2bvh import convert_npy_to_bvh
from .rotation import rot6d_to_fk_positions, bvh_to_joints_rot, rot6d_to_rotmat_batch
from animatrix.data.utils.mesh import batch_rigid_transform

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

def plot_bvh_compare(
    pred_bvh_path: str,
    gt_bvh_path: str,
    species_name: str,
    species_actual: str = "",
    save_dir: str = "./vis_compare",
    fps: int = 20,
    front_azim: int = 60,
    side_azim: int = 150,
    image_folder: str = None,
    save_type: str = "mp4",
    bitrate: int = 4000,
    wild_flag: bool = False,
):
    """
    Compare predicted BVH vs GT BVH by directly loading joint positions from BVH.

    Args:
        pred_bvh_path: path to predicted BVH
        gt_bvh_path: path to GT BVH
        species_name: species name for title / output filename
        species_actual: optional species_actual species string for output filename
        save_dir: output folder
        fps: animation fps
        front_azim: azimuth for front-view subplot
        side_azim: azimuth for side-view subplot
        image_folder: optional image sequence folder
        save_type: "mp4" or "gif"
        bitrate: bitrate for mp4
        wild_flag: if True, only show prediction and ignore GT
    """

    def _build_edges_from_parents(parents, joint_count):
        edges = []
        for j in range(joint_count):
            p = parents[j]
            if p >= 0 and p < joint_count:
                edges.append((p, j))

        if len(edges) == 0 and joint_count > 1:
            edges = [(i, i + 1) for i in range(joint_count - 1)]

        return edges

    def _normalize_positions(positions):
        """
        positions: [F, J, 3]
        Root-center each frame for stable comparison.
        """
        positions = np.asarray(positions, dtype=np.float32)
        positions = positions - positions[:, 0:1, :]
        return positions

    def _to_vis_coords(joints):
        """
        Match your existing visualization convention:
        xyz -> xzy, and flip x
        """
        joints = joints[..., [0, 2, 1]].copy()
        joints[..., 0] *= -1
        return joints

    # --------------------------------------------------
    # Load prediction BVH
    # --------------------------------------------------
    rot6d, pred_anim = bvh_to_joints_rot(pred_bvh_path)
    pred_parents = np.asarray(pred_anim.parents)
    pred_offsets = np.asarray(pred_anim.offsets)     # loaded as requested
    pred_positions = rot6d_to_fk_positions(torch.from_numpy(rot6d).unsqueeze(0), torch.from_numpy(pred_offsets).unsqueeze(0), torch.from_numpy(pred_parents).unsqueeze(0), torch.ones(1, dtype=torch.float32)).squeeze(0)
    
    joints_pred = _normalize_positions(pred_positions)
    F_pred, J_pred = joints_pred.shape[:2]

    ktree = parent_to_kinematic_tree(pred_anim.parents)
    J = joints_pred.shape[1]
    ktree = [chain for chain in ktree if np.all(np.array(chain) < J)]
    if len(ktree) == 0:
        print(f"[WARN] BVH chains do not match predicted joint count, using sequential connections.")
        ktree = [[i, i + 1] for i in range(J - 1)]
    print(f"[INFO] Using skeleton structure: {pred_bvh_path}, valid joints={J}, valid chains={len(ktree)}")

    # --------------------------------------------------
    # Load GT BVH if needed
    # --------------------------------------------------
    if not wild_flag:
        gt_rot6d, gt_anim = bvh_to_joints_rot(gt_bvh_path)
        gt_parents = np.asarray(gt_anim.parents)
        gt_offsets = np.asarray(gt_anim.offsets)     
        gt_positions = rot6d_to_fk_positions(torch.from_numpy(gt_rot6d).unsqueeze(0), torch.from_numpy(gt_offsets).unsqueeze(0), torch.from_numpy(gt_parents).unsqueeze(0), torch.ones(1, dtype=torch.float32)).squeeze(0)

        joints_gt = _normalize_positions(gt_positions)

        # align frame count
        F = min(F_pred, joints_gt.shape[0])
        joints_pred = joints_pred[:F]
        joints_gt = joints_gt[:F]

        # align joint count
        J = min(joints_pred.shape[1], joints_gt.shape[1])
        joints_pred = joints_pred[:, :J]
        joints_gt = joints_gt[:, :J]

        pred_parents = pred_parents[:J]
        gt_parents = gt_parents[:J]
    else:
        gt_anim = None
        gt_offsets = None
        gt_parents = None
        joints_gt = None
        F = F_pred
        J = J_pred
        joints_pred = joints_pred[:F]
        pred_parents = pred_parents[:J]

    # --------------------------------------------------
    # Coordinate adjustment
    # --------------------------------------------------
    joints_pred = _to_vis_coords(joints_pred)
    if joints_gt is not None:
        joints_gt = _to_vis_coords(joints_gt)

    # --------------------------------------------------
    # Load image sequence if provided
    # --------------------------------------------------
    if image_folder is not None:
        img_files = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        assert len(img_files) >= F, (
            f"image_folder has fewer images than animation frames: "
            f"{len(img_files)} vs {F}"
        )
    else:
        img_files = None

    # --------------------------------------------------
    # Compute plot limits
    # --------------------------------------------------
    if joints_gt is not None:
        all_points = np.concatenate(
            [joints_pred.reshape(-1, 3), joints_gt.reshape(-1, 3)],
            axis=0
        )
    else:
        all_points = joints_pred.reshape(-1, 3)

    xyz_min = all_points.min(axis=0)
    xyz_max = all_points.max(axis=0)
    max_range = (xyz_max - xyz_min).max() / 2.0
    center = (xyz_max + xyz_min) / 2.0

    lims = [
        (center[0] - max_range, center[0] + max_range),
        (center[1] - max_range, center[1] + max_range),
        (center[2] - max_range, center[2] + max_range),
    ]

    # === Subplot layout ===
    if img_files is not None:
        fig = plt.figure(figsize=(9, 3))
        ax1 = fig.add_subplot(132, projection='3d')
        ax2 = fig.add_subplot(133, projection='3d')
        ax_img = fig.add_subplot(131)
        ax_img.axis('off')
    else:
        fig = plt.figure(figsize=(6, 3))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

    for ax in [ax1, ax2]:
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        ax.set_xlabel('-X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

    ax1.view_init(elev=15, azim=front_azim)
    ax2.view_init(elev=15, azim=side_azim)
    ax1.set_title(f"{species_name} (Front)")
    ax2.set_title(f"{species_name} (Side)")

    # === Initialize scatter ===
    scat_pred1 = ax1.scatter([], [], [], s=10, c='blue', alpha=0.8, label='Pred')
    scat_pred2 = ax2.scatter([], [], [], s=10, c='blue', alpha=0.8)
    scat_gt1 = scat_gt2 = None

    if joints_gt is not None:
        scat_gt1 = ax1.scatter([], [], [], s=10, c='red', alpha=0.6, label='GT')
        scat_gt2 = ax2.scatter([], [], [], s=10, c='red', alpha=0.6)

    # Show legend externally with explicit handles
    handles = [scat_pred2]
    labels = ['Pred']
    if joints_gt is not None:
        handles.append(scat_gt2)
        labels.append('GT')
    ax2.legend(handles, labels)

    # === Initialize lines ===
    lines_pred1, lines_pred2 = [], []
    lines_gt1, lines_gt2 = [], []
    
    for chain in ktree:
        lp1, = ax1.plot([], [], [], lw=2, color='blue', alpha=0.8)
        lp2, = ax2.plot([], [], [], lw=2, color='blue', alpha=0.8)
        lines_pred1.append((lp1, chain))
        lines_pred2.append((lp2, chain))

        if joints_gt is not None:
            lg1, = ax1.plot([], [], [], lw=2, color='red', alpha=0.6)
            lg2, = ax2.plot([], [], [], lw=2, color='red', alpha=0.6)
            lines_gt1.append((lg1, chain))
            lines_gt2.append((lg2, chain))

    # === Update function ===
    def update(frame):
        jp = joints_pred[frame]
        ims = []

        # Update predicted lines
        for (lp1, c), (lp2, _) in zip(lines_pred1, lines_pred2):
            lp1.set_data(jp[c, 0], jp[c, 1])
            lp1.set_3d_properties(jp[c, 2])
            lp2.set_data(jp[c, 0], jp[c, 1])
            lp2.set_3d_properties(jp[c, 2])
            ims += [lp1, lp2]

        scat_pred1._offsets3d = (jp[:, 0], jp[:, 1], jp[:, 2])
        scat_pred2._offsets3d = (jp[:, 0], jp[:, 1], jp[:, 2])
        ims += [scat_pred1, scat_pred2]

        # Update GT lines
        if joints_gt is not None:
            jg = joints_gt[frame]
            for (lg1, c), (lg2, _) in zip(lines_gt1, lines_gt2):
                lg1.set_data(jg[c, 0], jg[c, 1])
                lg1.set_3d_properties(jg[c, 2])
                lg2.set_data(jg[c, 0], jg[c, 1])
                lg2.set_3d_properties(jg[c, 2])
                ims += [lg1, lg2]

            scat_gt1._offsets3d = (jg[:, 0], jg[:, 1], jg[:, 2])
            scat_gt2._offsets3d = (jg[:, 0], jg[:, 1], jg[:, 2])
            ims += [scat_gt1, scat_gt2]

        # Image frame
        if img_files is not None:
            img = Image.open(img_files[frame])
            ax_img.clear()
            ax_img.axis('off')
            ax_img.imshow(img)

        return ims

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    ext = ".mp4" if save_type == "mp4" else ".gif"
    save_path = os.path.join(save_dir, f"{species_name}_{species_actual}_bvh_compare{ext}")

    ani = FuncAnimation(fig, update, frames=F, interval=1000 / fps, blit=True)

    if save_type == "mp4":
        ani.save(save_path, writer=FFMpegWriter(fps=fps, bitrate=bitrate))
    else:
        ani.save(save_path, writer=PillowWriter(fps=fps))

    plt.close(fig)

    mode = "WILD" if wild_flag else "NORMAL"
    print(f"[DONE] Saved {save_type.upper()} ({mode} mode) -> {save_path}")

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

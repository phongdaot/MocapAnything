
import os
from typing import Optional
import numpy as np
import torch
from utils.logger import logger
from utils.finder import get_image_seq_relpath
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from . import bvh as BVH
from .common import parent_to_kinematic_tree
import shutil

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

def plot_pose_compare_from_npy(
        pred_npy_path: str,
        gt_npy_path: str,
        species_name: str,
        save_dir: str = "./vis_compare",
        fps: int = 20,
        camera_azim: Optional[int] = None,
        front_azim: int = 60,
        side_azim: int = 150,
        image_folder: str = None,
        save_type: str = "mp4",
        bitrate: int = 4000,
        wild_flag: bool = False,
        species_actual: str = "",
        bvh_roots=None,
):
    """
    Supports two modes:
      - Normal mode: draw prediction (Pred, blue) vs GT (GT, red) comparison
      - Wild mode: draw only predicted skeleton (no GT)

    Output:
      - separate image video (if image_folder is provided)
      - separate front-view video
      - separate side-view video
      - concatenated final video [image | front | side] if image exists
        or [front | side] if image does not exist
    """

    # === Load prediction ===
    joints_pred = np.load(pred_npy_path)
    F = joints_pred.shape[0]

    # === Load GT (only in non-wild mode) ===
    if not wild_flag:
        joints_gt = np.load(gt_npy_path)
        joints_gt = joints_gt[:F]
    else:
        joints_gt = None

    # === Coordinate adjustment ===
    joints_pred = joints_pred[..., [0, 2, 1]]
    joints_pred[..., 0] *= -1
    if joints_gt is not None:
        joints_gt = joints_gt[..., [0, 2, 1]]
        joints_gt[..., 0] *= -1

    # === Load image sequence ===
    if image_folder is not None:
        img_files = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        assert len(img_files) >= F, f"image_folder has fewer images than animation frames: {len(img_files)} vs {F}"
    else:
        img_files = None

    # === Find corresponding BVH skeleton (supports multiple BVH folders) ===
    bvh_path = None
    if bvh_roots is not None:
        for root_dir in bvh_roots:
            for root, _, files in os.walk(root_dir):
                for f in files:
                    dir_name = os.path.basename(root)
                    if f.endswith(".bvh") and dir_name.startswith(species_name + "#"):
                        bvh_path = os.path.join(root, f)
                        break
                if bvh_path:
                    break
            if bvh_path:
                break

    if bvh_path is None:
        print(f"[WARN] Cannot find BVH file for {species_name}, using a dummy chain structure.")
        J = joints_pred.shape[1]
        ktree = [[i, i + 1] for i in range(0, min(J - 1, 10))]
    else:
        anim, _, _ = BVH.load(bvh_path)
        ktree = parent_to_kinematic_tree(anim.parents)
        J = joints_pred.shape[1]
        ktree = [chain for chain in ktree if np.all(np.array(chain) < J)]
        if len(ktree) == 0:
            print(f"[WARN] BVH chains do not match predicted joint count, using sequential connections.")
            ktree = [[i, i + 1] for i in range(J - 1)]
        print(f"[INFO] Using skeleton structure: {bvh_path}, valid joints={J}, valid chains={len(ktree)}")

    # === Coordinate range ===
    if joints_gt is not None:
        all_points = np.concatenate([joints_pred.reshape(-1, 3), joints_gt.reshape(-1, 3)], axis=0)
    else:
        all_points = joints_pred.reshape(-1, 3)

    xyz_min, xyz_max = all_points.min(0), all_points.max(0)
    max_range = (xyz_max - xyz_min).max() / 2
    center = (xyz_max + xyz_min) / 2
    lims = [
        (center[0] - max_range, center[0] + max_range),
        (center[1] - max_range, center[1] + max_range),
        (center[2] - max_range, center[2] + max_range)
    ]

    # === Save paths ===
    os.makedirs(save_dir, exist_ok=True)
    ext = ".mp4" if save_type == "mp4" else ".gif"
    base_name = f"{species_name}_{species_actual}_pose_compare"

    front_path = os.path.join(save_dir, f"{base_name}_front{ext}")
    side_path = os.path.join(save_dir, f"{base_name}_side{ext}")
    camera_path = os.path.join(save_dir, f"{base_name}_camera{ext}")
    image_path = os.path.join(save_dir, f"{base_name}_image{ext}") if img_files is not None else None
    final_path = os.path.join(save_dir, f"{base_name}{ext}")

    writer = FFMpegWriter(fps=fps, bitrate=bitrate) if save_type == "mp4" else PillowWriter(fps=fps)

    if camera_azim is not None and not os.path.exists(camera_path):
        fig_camera = plt.figure(figsize=(4, 4))
        ax_camera = fig_camera.add_subplot(111, projection='3d')
        ax_camera.set_xlim(*lims[0])
        ax_camera.set_ylim(*lims[1])
        ax_camera.set_zlim(*lims[2])
        ax_camera.set_xlabel('-X')
        ax_camera.set_ylabel('Z')
        ax_camera.set_zlabel('Y')
        ax_camera.view_init(elev=15, azim=camera_azim)
        ax_camera.set_title(f"{species_name} (Camera View)")
        
        scat_pred_camera = ax_camera.scatter([], [], [], s=10, c='blue', alpha=0.8, label='Pred')
        scat_gt_camera = None
        if joints_gt is not None:
            scat_gt_camera = ax_camera.scatter([], [], [], s=10, c='red', alpha=0.6, label='GT')
            ax_camera.legend()
            
        lines_pred_camera = []
        lines_gt_camera = []
        for chain in ktree:
            lp, = ax_camera.plot([], [], [], lw=2, color='blue', alpha=0.8)
            lines_pred_camera.append((lp, chain))
            if joints_gt is not None:
                lg, = ax_camera.plot([], [], [], lw=2, color='red', alpha=0.6)
                lines_gt_camera.append((lg, chain))
        
        def update_camera(frame):
            jp = joints_pred[frame]
            ims = []

            for lp, c in lines_pred_camera:
                lp.set_data(jp[c, 0], jp[c, 1])
                lp.set_3d_properties(jp[c, 2])
                ims.append(lp)

            scat_pred_camera._offsets3d = (jp[:, 0], jp[:, 1], jp[:, 2])
            ims.append(scat_pred_camera)

            if joints_gt is not None:
                jg = joints_gt[frame]
                for lg, c in lines_gt_camera:
                    lg.set_data(jg[c, 0], jg[c, 1])
                    lg.set_3d_properties(jg[c, 2])
                    ims.append(lg)

                scat_gt_camera._offsets3d = (jg[:, 0], jg[:, 1], jg[:, 2])
                ims.append(scat_gt_camera)

            return ims
        
        ani_camera = FuncAnimation(fig_camera, update_camera, frames=F, interval=1000 / fps, blit=True)
        ani_camera.save(camera_path, writer=writer)
        plt.close(fig_camera)
        print(f"[DONE] Saved camera view → {camera_path}")

    if front_azim is not None and not os.path.exists(front_path):
        # =========================================================
        # Front view figure
        # =========================================================
        fig_front = plt.figure(figsize=(4, 4))
        ax_front = fig_front.add_subplot(111, projection='3d')
        ax_front.set_xlim(*lims[0])
        ax_front.set_ylim(*lims[1])
        ax_front.set_zlim(*lims[2])
        ax_front.set_xlabel('-X')
        ax_front.set_ylabel('Z')
        ax_front.set_zlabel('Y')
        ax_front.view_init(elev=15, azim=front_azim)
        ax_front.set_title(f"{species_name} (Front)")

        scat_pred_front = ax_front.scatter([], [], [], s=10, c='blue', alpha=0.8, label='Pred')
        scat_gt_front = None
        if joints_gt is not None:
            scat_gt_front = ax_front.scatter([], [], [], s=10, c='red', alpha=0.6, label='GT')
            ax_front.legend()

        lines_pred_front = []
        lines_gt_front = []
        for chain in ktree:
            lp, = ax_front.plot([], [], [], lw=2, color='blue', alpha=0.8)
            lines_pred_front.append((lp, chain))
            if joints_gt is not None:
                lg, = ax_front.plot([], [], [], lw=2, color='red', alpha=0.6)
                lines_gt_front.append((lg, chain))

        def update_front(frame):
            jp = joints_pred[frame]
            ims = []

            for lp, c in lines_pred_front:
                lp.set_data(jp[c, 0], jp[c, 1])
                lp.set_3d_properties(jp[c, 2])
                ims.append(lp)

            scat_pred_front._offsets3d = (jp[:, 0], jp[:, 1], jp[:, 2])
            ims.append(scat_pred_front)

            if joints_gt is not None:
                jg = joints_gt[frame]
                for lg, c in lines_gt_front:
                    lg.set_data(jg[c, 0], jg[c, 1])
                    lg.set_3d_properties(jg[c, 2])
                    ims.append(lg)

                scat_gt_front._offsets3d = (jg[:, 0], jg[:, 1], jg[:, 2])
                ims.append(scat_gt_front)

            return ims

        ani_front = FuncAnimation(fig_front, update_front, frames=F, interval=1000 / fps, blit=True)
        ani_front.save(front_path, writer=writer)
        plt.close(fig_front)
        print(f"[DONE] Saved front view → {front_path}")

    
    if side_azim is not None and not os.path.exists(side_path):
        # =========================================================
        # Side view figure
        # =========================================================
        fig_side = plt.figure(figsize=(4, 4))
        ax_side = fig_side.add_subplot(111, projection='3d')
        ax_side.set_xlim(*lims[0])
        ax_side.set_ylim(*lims[1])
        ax_side.set_zlim(*lims[2])
        ax_side.set_xlabel('-X')
        ax_side.set_ylabel('Z')
        ax_side.set_zlabel('Y')
        ax_side.view_init(elev=15, azim=side_azim)
        ax_side.set_title(f"{species_name} (Side)")

        scat_pred_side = ax_side.scatter([], [], [], s=10, c='blue', alpha=0.8, label='Pred')
        scat_gt_side = None
        if joints_gt is not None:
            scat_gt_side = ax_side.scatter([], [], [], s=10, c='red', alpha=0.6, label='GT')
            ax_side.legend()

        lines_pred_side = []
        lines_gt_side = []
        for chain in ktree:
            lp, = ax_side.plot([], [], [], lw=2, color='blue', alpha=0.8)
            lines_pred_side.append((lp, chain))
            if joints_gt is not None:
                lg, = ax_side.plot([], [], [], lw=2, color='red', alpha=0.6)
                lines_gt_side.append((lg, chain))

        def update_side(frame):
            jp = joints_pred[frame]
            ims = []

            for lp, c in lines_pred_side:
                lp.set_data(jp[c, 0], jp[c, 1])
                lp.set_3d_properties(jp[c, 2])
                ims.append(lp)

            scat_pred_side._offsets3d = (jp[:, 0], jp[:, 1], jp[:, 2])
            ims.append(scat_pred_side)

            if joints_gt is not None:
                jg = joints_gt[frame]
                for lg, c in lines_gt_side:
                    lg.set_data(jg[c, 0], jg[c, 1])
                    lg.set_3d_properties(jg[c, 2])
                    ims.append(lg)

                scat_gt_side._offsets3d = (jg[:, 0], jg[:, 1], jg[:, 2])
                ims.append(scat_gt_side)

            return ims

        ani_side = FuncAnimation(fig_side, update_side, frames=F, interval=1000 / fps, blit=True)
        ani_side.save(side_path, writer=writer)
        plt.close(fig_side)
        print(f"[DONE] Saved side view → {side_path}")

    # =========================================================
    # Image figure
    # =========================================================
    if img_files is not None:
        fig_img = plt.figure(figsize=(4, 4))
        ax_img = fig_img.add_subplot(111)
        ax_img.axis('off')

        img_artist = ax_img.imshow(Image.open(img_files[0]))

        def update_image(frame):
            img = Image.open(img_files[frame])
            img_artist.set_data(img)
            return [img_artist]

        ani_img = FuncAnimation(fig_img, update_image, frames=F, interval=1000 / fps, blit=True)
        ani_img.save(image_path, writer=writer)
        plt.close(fig_img)
        print(f"[DONE] Saved image view → {image_path}")

    # =========================================================
    # Concatenate videos
    # =========================================================
    if save_type == "mp4":
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            print("[WARN] ffmpeg not found. Separate videos were saved, but concatenation was skipped.")
        else:
            try:
                include_paths = []
                if img_files is not None:
                    include_paths.append(image_path)

                if camera_azim is not None:
                    include_paths.append(camera_path)

                include_paths.append(front_path)
                include_paths.append(side_path)

                num_inputs = len(include_paths)
                patterns = "".join([f"[{i}:v]" for i in range(num_inputs)])
                filter_complex = f"{patterns}hstack=inputs={num_inputs}[v]"

                cmd = [ffmpeg_path, "-y"]
                for path in include_paths:
                    cmd.extend(["-i", path])

                cmd.extend([
                    "-filter_complex", filter_complex,
                    "-map", "[v]",
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    final_path,
                ])

                subprocess.run(cmd, check=True)
                print(f"[DONE] Saved concatenated video → {final_path}")
            except subprocess.CalledProcessError as e:
                print(f"[WARN] ffmpeg concatenation failed: {e}")
                print("[INFO] Separate videos are still available.")
    else:
        print("[INFO] GIF concatenation is not implemented. Separate GIF files were saved.")

    mode = "WILD" if wild_flag else "NORMAL"
    print(f"[DONE] Finished {save_type.upper()} ({mode} mode)")

def add_background_to_image(image_path, bg_color):
    """
    Add a solid background color behind an image and overwrite the original file.

    Useful for PNGs with transparency.

    Args:
        image_path (str): Path to input image.
        bg_color (tuple): RGB tuple like (255, 255, 255).

    Returns:
        str: Path to the overwritten image.
    """
    img = Image.open(image_path).convert("RGBA")

    background = Image.new("RGBA", img.size, (*bg_color, 255))
    composited = Image.alpha_composite(background, img).convert("RGB")

    composited.save(image_path)
    return image_path

def add_background_to_image_folder(image_dir, bg_color):
    """
    Add solid background to all images in a folder and overwrite them.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp"}

    for name in sorted(os.listdir(image_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in exts:
            continue

        image_path = os.path.join(image_dir, name)
        add_background_to_image(image_path, bg_color)

    return image_dir

import os
import subprocess


def convert_images_to_video(image_dir, video_path, fps=20, crf=18, preset="slow"):
    """
    Convert a folder of images to a video file using ffmpeg directly.

    This is much faster than matplotlib + FFMpegWriter because ffmpeg reads
    the image sequence itself.

    Args:
        image_dir (str): Path to folder containing image sequence.
        video_path (str): Path to output video file (e.g. .mp4).
        fps (int): Frames per second for the output video.
        crf (int): Constant rate factor for x264 quality. Lower = better quality.
        preset (str): x264 preset, e.g. ultrafast, fast, medium, slow.

    Returns:
        str: Path to the saved video file.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    img_files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not img_files:
        raise ValueError(f"No image files found in: {image_dir}")

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Choose extension based on what exists
    exts = [os.path.splitext(f)[1].lower() for f in img_files]
    if ".png" in exts:
        pattern = os.path.join(image_dir, "*.png")
    elif ".jpg" in exts:
        pattern = os.path.join(image_dir, "*.jpg")
    else:
        pattern = os.path.join(image_dir, "*.jpeg")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", pattern,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        video_path,
    ]

    subprocess.run(cmd, check=True)
    return video_path
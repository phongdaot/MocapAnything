import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import bvh as BVH
from utils import animation as animation
from utils.common import parent_to_kinematic_tree
from utils.mesh import read_obj_mesh
from utils.bvh_tools import get_diameter


# ====== Global cache ======
species_scale_cache = {}


def get_species_name(bvh_pth):
    """bvh/Alligator#AlligatorALL-BigMouth/y0.bvh -> Alligator"""
    folder = os.path.basename(os.path.dirname(bvh_pth))
    species = folder.split("#")[0]
    return species


def get_base_mesh_path_from_bvh(bvh_pth):
    """Derive the base mesh path from a BVH path."""
    species = get_species_name(bvh_pth)
    root_dir = 'zoo'
    return os.path.join(root_dir, 'characters_fix_facezplus', species, 'base_mesh.obj')


def get_scale_cached(bvh_pth, auto_scale='base_mesh'):
    """Compute the scale only once per species."""
    species = get_species_name(bvh_pth)
    if species in species_scale_cache:
        return species_scale_cache[species]

    if auto_scale == 'bvh':
        anim, names, frametime = BVH.load(bvh_pth)
        bone_length = np.linalg.norm(anim.offsets, axis=1)
        diameter, path = get_diameter(anim.parents, bone_length)
        scale = 1 / diameter
    elif auto_scale == 'base_mesh':
        base_mesh_path = get_base_mesh_path_from_bvh(bvh_pth)
        temp_vertices, temp_faces = read_obj_mesh(base_mesh_path)
        v_max, v_min = temp_vertices.max(axis=0), temp_vertices.min(axis=0)
        scale_factor = (v_max - v_min).max()
        scale = 1 / scale_factor
    else:
        scale = 1.0

    species_scale_cache[species] = scale
    return scale


def extract_positions_from_bvh(bvh_pth, auto_scale='base_mesh'):
    """
    Given a BVH file, returns:
      - position_before (T, J, 3): joint positions BEFORE scaling
      - position (T, J, 3): joint positions AFTER scaling
      - traj_before (T, 3): root-joint trajectory BEFORE scaling
      - traj (T, 3): root-joint trajectory AFTER scaling
      - frametime, scale, ktree
    """
    scale = get_scale_cached(bvh_pth, auto_scale=auto_scale)
    anim, _, frametime = BVH.load(bvh_pth)
    global_transf = animation.transforms_global(anim)

    # (T, J, 3) before scaling
    position_before = global_transf[:, :, :3, 3] / global_transf[:, :, 3, 3:]
    # (T, J, 3) after scaling
    position = position_before * scale

    traj_before = position_before[:, 0, :]
    traj = position[:, 0, :]

    ktree = parent_to_kinematic_tree(anim.parents)

    return position_before, position, traj_before, traj, frametime, scale, ktree


def bvh_to_joints_rot(bvh_path):
    """
    Load a BVH file and return its rotations as rot6d, shape (F, J, 6), dtype float32.
    """
    anim, _, _ = BVH.load(bvh_path)
    local_xforms = animation.transforms_local(anim)
    rot_matrices = local_xforms[:, :, :3, :3]
    rot_matrices_flat = rot_matrices.reshape(rot_matrices.shape[0], rot_matrices.shape[1], 9)
    rot6d = rot_matrices_flat[:, :, :6].astype(np.float32)
    return rot6d


def plot_pose_with_traj_gif(
    position,
    traj,
    ktree,
    save_path,
    fps=20,
    bone_color="black",
    traj_color="red",
    point_color="blue",
    frametime=1/30,
    title=None
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Skeleton lines
    chains = []
    for chain in ktree:
        line, = ax.plot([], [], [], lw=2, color=bone_color, alpha=0.7)
        chains.append((line, chain))

    # Trajectory
    traj_line, = ax.plot([], [], [], lw=2, color=traj_color, label="traj")
    traj_point, = ax.plot([], [], [], "o", color=point_color, label="root")

    # Fixed axis ranges
    xlim = (position[:, :, 0].min(), position[:, :, 0].max())
    ylim = (position[:, :, 1].min(), position[:, :, 1].max())
    zlim = (position[:, :, 2].min(), position[:, :, 2].max())
    max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2.0
    mid_x, mid_y, mid_z = np.mean(xlim), np.mean(ylim), np.mean(zlim)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title if title else "Pose + Trajectory")
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)

    # Camera angle for y-up orientation
    ax.view_init(elev=20, azim=70)

    def update(frame):
        joints = position[frame]
        for line, chain in chains:
            line.set_data(joints[chain, 0], joints[chain, 1])
            line.set_3d_properties(joints[chain, 2])
        traj_line.set_data(traj[:frame, 0], traj[:frame, 1])
        traj_line.set_3d_properties(traj[:frame, 2])
        traj_point.set_data(traj[frame:frame+1, 0], traj[frame:frame+1, 1])
        traj_point.set_3d_properties(traj[frame:frame+1, 2])
        time_text.set_text(f"Frame {frame} | Time {frame*frametime:.2f}s")
        return [line for line,_ in chains] + [traj_line, traj_point, time_text]

    ani = FuncAnimation(fig, update, frames=len(traj), interval=1000/fps, blit=True)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def process_bvh_root(bvh_root, out_pose_npz_root,
                     auto_scale='base_mesh', fps=20):
    """Walk over every .bvh file under bvh_root."""
    bvh_files = []
    for root, _, files in os.walk(bvh_root):
        for fname in files:
            if fname.endswith(".bvh"):
                bvh_files.append(os.path.join(root, fname))

    bvh_files = sorted(bvh_files)

    for bvh_path in tqdm(bvh_files, desc="Processing BVH files"):
        rel_path = os.path.relpath(bvh_path, bvh_root)

        npz_path = os.path.join(out_pose_npz_root, rel_path.replace(".bvh", ".npz"))
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)

        # Extract the full set of pose data
        position_before, position, traj_before, traj, frametime, scale, ktree = extract_positions_from_bvh(
            bvh_path, auto_scale=auto_scale
        )

        # Extract rot6d data
        rot6d = bvh_to_joints_rot(bvh_path)

        # Save npz
        np.savez_compressed(
            npz_path,
            position_before=position_before,
            position=position,
            traj_before=traj_before,
            traj=traj,
            rot6d=rot6d,
            frametime=frametime,
            scale=scale
        )


# ==== Usage example ====
if __name__ == "__main__":
    zoo_root = "zoo"
    bvh_root = os.path.join(zoo_root, "bvh")
    out_pose_npz_root = os.path.join(zoo_root, "bvh_pose")

    process_bvh_root(bvh_root, out_pose_npz_root,
                     auto_scale="base_mesh", fps=30)

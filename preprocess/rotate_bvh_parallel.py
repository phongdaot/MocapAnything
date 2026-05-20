import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import bvh as BVH
from utils.transforms3d import (
    quaternion_to_matrix,
    euler_angles_to_matrix,
    matrix_to_euler_angles
)

# Y-axis rotations in degrees, paired with output filenames below.
rotations = [
    [0,   0, 0], [0,  15, 0], [0,  30, 0], [0,  45, 0],
    [0,  60, 0], [0,  75, 0], [0,  90, 0], [0, 135, 0],
    [0, 180, 0], [0, 225, 0], [0, 270, 0], [0, 315, 0],
]
save_names = ['y0', 'y15', 'y30', 'y45', 'y60', 'y75', 'y90', 'y135', 'y180', 'y225', 'y270', 'y315']

bvh_directory = 'zoo/motions'
output_directory = 'zoo/bvh'
os.makedirs(output_directory, exist_ok=True)

def rotate_bvh(base_bvh_path, rotation, save_path):
    anim, joint_name, frame_time = BVH.load(base_bvh_path)
    root_quat = torch.tensor(np.array(anim.rotations[:, 0]))
    root_matrix = quaternion_to_matrix(root_quat).float()
    add_euler = torch.deg2rad(torch.tensor(rotation)).repeat(root_quat.shape[0], 1)
    add_matrix = euler_angles_to_matrix(add_euler, convention='ZYX')
    new_matrix = torch.matmul(add_matrix, root_matrix)
    new_root_euler = matrix_to_euler_angles(new_matrix, convention='ZYX')
    all_euler = matrix_to_euler_angles(
        quaternion_to_matrix(torch.tensor(np.array(anim.rotations))), convention='ZYX'
    )
    all_euler[:, 0] = new_root_euler

    all_positions = torch.tensor(anim.positions)
    new_positions = torch.matmul(
        add_matrix, all_positions[:, 0].unsqueeze(-1).float()
    ).squeeze(-1)
    all_positions[:, 0] = new_positions

    BVH.save_dict(
        filename=save_path,
        data={
            'rotations': torch.rad2deg(all_euler),
            'positions': all_positions,
            'offsets': anim.offsets,
            'parents': anim.parents,
            'names': joint_name,
            'order': 'zyx',
            'frametime': frame_time,
        }
    )
    return f"[DONE] {os.path.basename(base_bvh_path)} -> {os.path.basename(save_path)}"

def generate_tasks():
    tasks = []
    for filename in sorted(os.listdir(bvh_directory)):
        if filename.endswith('.bvh'):
            base_bvh_path = os.path.join(bvh_directory, filename)
            name, _ = os.path.splitext(filename)
            safe_name = name.replace(' ', '_')
            out_folder = os.path.join(output_directory, safe_name)
            os.makedirs(out_folder, exist_ok=True)
            for rotation, save_name in zip(rotations, save_names):
                save_path = os.path.join(out_folder, f"{save_name}.bvh")
                tasks.append((base_bvh_path, rotation, save_path))
    return tasks

def main(num_workers=32):
    tasks = generate_tasks()
    print(f"==> Processing {len(tasks)} rotation tasks with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(rotate_bvh, *args) for args in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Rotating BVH"):
            try:
                print(fut.result())
            except Exception as e:
                print(f"[ERROR] {e}")

    print("==> All BVH rotations completed.")

if __name__ == "__main__":
    main(num_workers=32)

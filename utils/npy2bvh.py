### npy2bvh.py ###
import numpy as np
import torch
from .transforms3d import rotation_6d_to_matrix, matrix_to_euler_angles
from . import bvh as BVH
import os
import sys, subprocess
import argparse
import os
import glob
from pathlib import Path
def rot6d_to_bvh(
    rot_6d, transl_rec, bvh_offset, parents, joint_names, num_joints, save_path = 'dummy.bvh'
):
    seq_len = rot_6d.shape[0]

    # Convert 6D rotations to rotation matrices
    rot_mat = rotation_6d_to_matrix(rot_6d)

    # Convert rotation matrices to Euler angles
    rot_euler = matrix_to_euler_angles(rot_mat, 'ZYX').numpy()

    # Create positions using offset and translation
    positions = np.tile(bvh_offset[:num_joints], (seq_len, 1, 1))
    positions[:, 0] += transl_rec

    # Create BVH data dictionary
    bvh_data = {
        'rotations': np.degrees(rot_euler),
        'positions': positions,
        'offsets': bvh_offset[:num_joints],
        'parents': parents[:num_joints],
        'names': joint_names[:num_joints],
        'order': 'zyx',
        'frametime': 1. / 30.,
    }

    BVH.save_dict(save_path,bvh_data,fps=30)
    return bvh_data

def convert_npy_to_bvh(npy_path, character_base_dir, species_name):
    print(f"Detected species: {species_name}")

    # 1. Build base directory
    base_dir = os.path.join(character_base_dir, species_name)

    # 2. Find template BVH
    search_pattern = os.path.join(base_dir, "*_ffs.bvh")
    found_bvh = glob.glob(search_pattern)
    if not found_bvh:
        raise FileNotFoundError(
            f"Could not find any file ending with '_ffs.bvh' in {base_dir}"
        )
    tpl_bvh_path = found_bvh[0]

    # 3. Set other paths
    base_mesh_path = os.path.join(base_dir, "base_mesh.obj")
    skin_weight_path = os.path.join(base_dir, "skinning_weights.npy")
    bvh_save_path = npy_path.replace(".npy", ".bvh")

    print("--- Paths configured ---")
    print(f"Template BVH path:   {tpl_bvh_path}")
    print(f"Base mesh path:      {base_mesh_path}")
    print(f"Skin weight path:    {skin_weight_path}")
    print(f"Output BVH path:     {bvh_save_path}")

    # 4. Load template BVH
    anim, joint_names, _ = BVH.load(tpl_bvh_path)

    # 5. Load prediction npy
    pred = np.load(npy_path)
    frame_cnt = pred.shape[0]

    # 6. Convert to BVH
    rot6d_to_bvh(
        torch.tensor(pred),
        np.zeros((frame_cnt, 3)),
        anim.offsets,
        anim.parents,
        joint_names,
        len(joint_names),
        save_path=bvh_save_path,
    )

    print(f"BVH saved to {bvh_save_path}")
    return bvh_save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPY motion file to BVH")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to the input .npy file")
    parser.add_argument("--character_base_dir", type=str, required=True, help="Base directory for character files")
    parser.add_argument("--species_name", type=str, required=True, help="Species name")
    args = parser.parse_args()

    convert_npy_to_bvh(args.npy_path, args.character_base_dir, args.species_name)
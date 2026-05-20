"""
Per-species bbox-normalize rotated meshes for the TripoSG VAE.

Logic:
1. For each frame, subtract the root trajectory (from bvh_pose) so the
   character is centered in place.
2. Collect every motion for one species, compute the species-wide bbox
   over all centered vertices.
3. bbox center becomes the offset; bbox half-extent (max axis) becomes the
   global scale. Vertices are normalized to [-1, 1].

Input  : zoo/npz_remesh/{motion}/y{deg}.npz       (vertices, normals)
         zoo/bvh_pose/{motion}/y{deg}.npz         (traj used for centering)
Output : zoo/npz_mesh_normed/{motion}/y{deg}.npz  with keys
           vertices, normals, traj, vertices_normed, bbox_center, global_scale

Species is parsed from the motion folder name before the '#', so
'Alligator#AlligatorALL-Bite' belongs to species 'Alligator' and shares its
bbox with every other Alligator motion.
"""
import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np


def load_traj_for_mesh_npz(mesh_npz_file, input_root, pose_root):
    rel = os.path.relpath(mesh_npz_file, input_root)
    pose_npz_file = os.path.join(pose_root, rel)
    return np.load(pose_npz_file)["traj"]


def get_species_bbox_center_and_scale(npz_files, input_root, pose_root):
    all_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    all_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for f in npz_files:
        data = np.load(f)
        verts = data["vertices"]
        traj = load_traj_for_mesh_npz(f, input_root, pose_root)
        verts_centered = verts - traj[:, np.newaxis, :]
        all_min = np.minimum(all_min, verts_centered.min(axis=(0, 1)))
        all_max = np.maximum(all_max, verts_centered.max(axis=(0, 1)))

    bbox_center = (all_max + all_min) / 2.0
    global_scale = float(((all_max - all_min) / 2.0).max())
    return bbox_center, global_scale


def normalize_with_bbox(vertices, traj, bbox_center, global_scale, eps=1e-8):
    verts_centered = vertices - traj[:, np.newaxis, :]
    verts_offset = verts_centered - bbox_center[np.newaxis, np.newaxis, :]
    return verts_offset / max(global_scale, eps)


def process_one_file(npz_file, out_dir, bbox_center, global_scale, input_root, pose_root):
    try:
        data = np.load(npz_file)
        verts = data["vertices"]
        normals = data["normals"]
        traj = load_traj_for_mesh_npz(npz_file, input_root, pose_root)

        verts_normed = normalize_with_bbox(verts, traj, bbox_center, global_scale)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(npz_file))
        np.savez_compressed(
            out_path,
            vertices=verts.astype(np.float16),
            normals=normals.astype(np.float16),
            traj=traj.astype(np.float16),
            vertices_normed=verts_normed.astype(np.float16),
            bbox_center=bbox_center.astype(np.float16),
            global_scale=np.array([global_scale], dtype=np.float16),
        )
        return f"[DONE] {os.path.basename(npz_file)} -> {out_path}"
    except Exception as e:
        return f"[ERROR] {npz_file}: {e}"


def process_species(species, dirnames, input_root, pose_root, output_root, file_workers):
    npz_files = []
    for d in dirnames:
        npz_files += sorted(glob(os.path.join(input_root, d, "y*.npz")))
    if not npz_files:
        return f"[WARN] species {species} has no npz, skipping"

    bbox_center, global_scale = get_species_bbox_center_and_scale(
        npz_files, input_root, pose_root
    )
    print(f"[INFO] {species}: bbox_center={bbox_center}, "
          f"global_scale={global_scale:.6f} ({len(npz_files)} files)")

    for d in dirnames:
        in_dir = os.path.join(input_root, d)
        out_dir = os.path.join(output_root, d)
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob(os.path.join(in_dir, "y*.npz")))
        with ProcessPoolExecutor(max_workers=file_workers) as ex:
            futures = [
                ex.submit(process_one_file, f, out_dir, bbox_center,
                          global_scale, input_root, pose_root)
                for f in files
            ]
            for fut in as_completed(futures):
                print(fut.result())

    return f"[DONE] {species}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", default="zoo/npz_remesh")
    p.add_argument("--pose_root", default="zoo/bvh_pose")
    p.add_argument("--output_root", default="zoo/npz_mesh_normed")
    p.add_argument("--species_workers", type=int, default=8)
    p.add_argument("--file_workers", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    dirs = [
        d for d in os.listdir(args.input_root)
        if os.path.isdir(os.path.join(args.input_root, d))
    ]
    species_to_dirs = defaultdict(list)
    for d in dirs:
        species_to_dirs[d.split("#")[0]].append(d)

    print(f"[INFO] Found {len(species_to_dirs)} species: "
          f"{list(species_to_dirs.keys())}")

    with ProcessPoolExecutor(max_workers=args.species_workers) as ex:
        futures = [
            ex.submit(process_species, species, dirnames,
                      args.input_root, args.pose_root,
                      args.output_root, args.file_workers)
            for species, dirnames in species_to_dirs.items()
        ]
        for fut in as_completed(futures):
            print(fut.result())


if __name__ == "__main__":
    main()

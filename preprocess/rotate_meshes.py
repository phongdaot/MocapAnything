"""
Rotate remeshed character meshes around the Y axis by 12 fixed angles.

Input  : --input_root/{name}.npz  with `vertices` (T,N,3) and `normals` (T,N,3)
Output : --output_root/{name}/y{deg}.npz  one file per angle, same keys
"""
import argparse
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


ANGLES = [0, 15, 30, 45, 60, 75, 90, 135, 180, 225, 270, 315]


def rotate_points_y(points, angle_deg):
    rot = R.from_euler('y', np.deg2rad(angle_deg)).as_matrix()
    return points @ rot.T


def is_already_processed(out_dir, angles):
    return all(os.path.exists(os.path.join(out_dir, f"y{a}.npz")) for a in angles)


def process_one(task):
    input_npz, angles, out_dir, name = task
    try:
        mesh = np.load(input_npz)
        verts = mesh["vertices"]
        normals = mesh["normals"]
        os.makedirs(out_dir, exist_ok=True)
        for angle in angles:
            save_path = os.path.join(out_dir, f"y{angle}.npz")
            if os.path.exists(save_path):
                continue
            np.savez(
                save_path,
                vertices=rotate_points_y(verts, angle).astype(np.float16),
                normals=rotate_points_y(normals, angle).astype(np.float16),
            )
        return f"[OK] {name}"
    except Exception as e:
        return f"[ERROR] {name}: {e}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", default="zoo/remesh_npz")
    p.add_argument("--output_root", default="zoo/npz_remesh")
    p.add_argument("--num_workers", type=int, default=min(50, cpu_count()))
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(f for f in os.listdir(args.input_root) if f.endswith(".npz"))
    print(f"Found {len(files)} files under {args.input_root}.")

    tasks = []
    for f in files:
        name = os.path.splitext(f)[0]
        in_npz = os.path.join(args.input_root, f)
        out_dir = os.path.join(args.output_root, name)
        if is_already_processed(out_dir, ANGLES):
            print(f"[SKIP] {name}")
            continue
        tasks.append((in_npz, ANGLES, out_dir, name))

    print(f"Will process {len(tasks)} files with {args.num_workers} workers.")
    if not tasks:
        return

    with Pool(processes=min(args.num_workers, len(tasks))) as pool:
        for result in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)):
            print(result)


if __name__ == "__main__":
    main()

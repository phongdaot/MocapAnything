"""
Normalize rotated meshes into a unit cube and rename `vertices` ->
`vertices_normed` for the TripoSG VAE encoder.

Input  : zoo/npz_remesh/{name}/y{deg}.npz   keys: vertices, normals
Output : zoo/npz_mesh_normed/{name}/y{deg}.npz   keys: vertices_normed, normals

Normalization (per file): center bbox at origin, scale longest axis to [-1, 1].
Normals are direction-only so they pass through unchanged.
"""
import argparse
import os
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """vertices: (..., 3). Returns same shape, fit into [-1, 1] cube."""
    v = vertices.astype(np.float32)
    vmin = v.reshape(-1, 3).min(axis=0)
    vmax = v.reshape(-1, 3).max(axis=0)
    center = (vmin + vmax) / 2.0
    scale = 2.0 / max((vmax - vmin).max(), 1e-8)
    return (v - center) * scale


def process_one(task):
    in_npz, out_npz = task
    try:
        if os.path.exists(out_npz):
            return f"[SKIP] {out_npz}"
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        data = np.load(in_npz)
        verts = data["vertices"].astype(np.float32)
        normals = data["normals"].astype(np.float32)

        # Normalize each frame independently if T,N,3; else single mesh.
        if verts.ndim == 3:
            normed = np.stack([normalize_vertices(v) for v in verts], axis=0)
        else:
            normed = normalize_vertices(verts)

        np.savez(
            out_npz,
            vertices_normed=normed.astype(np.float16),
            normals=normals.astype(np.float16),
        )
        return f"[OK] {out_npz}"
    except Exception as e:
        return f"[ERROR] {in_npz}: {e}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", default="zoo/npz_remesh")
    p.add_argument("--output_root", default="zoo/npz_mesh_normed")
    p.add_argument("--num_workers", type=int, default=min(32, cpu_count()))
    return p.parse_args()


def main():
    args = parse_args()
    in_files = sorted(glob(os.path.join(args.input_root, "*", "*.npz")))
    print(f"Found {len(in_files)} files under {args.input_root}.")

    tasks = []
    for in_npz in in_files:
        rel = os.path.relpath(in_npz, args.input_root)
        out_npz = os.path.join(args.output_root, rel)
        tasks.append((in_npz, out_npz))

    with Pool(processes=min(args.num_workers, max(len(tasks), 1))) as pool:
        for result in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)):
            print(result)


if __name__ == "__main__":
    main()

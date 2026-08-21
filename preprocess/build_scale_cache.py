"""
Build the per-species scale cache pickle consumed by loader_v2 / species_fps_memory.

Scale definition (matches how the loader normalizes):
    pos_rel   = position - position[:, 0:1, :]      # root-relative point cloud
    global_scale = max(|pos_rel|) over all frames/joints/xyz of the species
    => pos_rel / global_scale lies within [-1, +1]  (bbox fits the unit cube)

Output pickle format:
    { species_name: {"global_scale": float, "bbox_center": np.array([0,0,0], float32)} }

bbox_center is stored as zeros (the cloud is already root-centered); loader_v2 only
reads global_scale, species_fps_memory reads global_scale (+ optional bbox_center).
"""

import os
import glob
import pickle
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

EPS = 1e-6


POS_FIELD = "position"  # overridden by --pos_field

def safe_load_positions(npz_path):
    """Return root-relative position [F, J, 3], or None if empty/corrupt."""
    try:
        if os.path.getsize(npz_path) == 0:
            return None
        data = np.load(npz_path)
        pos = data[POS_FIELD].astype(np.float64)          # [F, J, 3]
        if pos.ndim != 3 or pos.shape[0] == 0:
            return None
        return pos - pos[:, 0:1, :]                       # root-relative
    except Exception as e:
        print(f"  ⚠️ skip unreadable npz: {npz_path} ({e})")
        return None


def collect_by_species(bvh_pose_root):
    species_dirs = defaultdict(list)
    for name in sorted(os.listdir(bvh_pose_root)):
        full = os.path.join(bvh_pose_root, name)
        if os.path.isdir(full) and "#" in name:
            species_dirs[name.split("#")[0]].append(full)
    return species_dirs


def species_global_scale(seq_dirs):
    """max abs coordinate of root-relative positions across all valid npz."""
    max_abs = 0.0
    n_used = 0
    for d in seq_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            pos_rel = safe_load_positions(p)
            if pos_rel is None:
                continue
            m = float(np.abs(pos_rel).max())
            if m > max_abs:
                max_abs = m
            n_used += 1
    return max_abs, n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_pose_root", default=os.path.join(os.environ.get("ZOO_ROOT", "zoo"), "bvh_pose"))
    ap.add_argument("--output", default=os.path.join(
        os.environ.get("ZOO_ROOT", "zoo"), "cache", "__mesh2pose1002_species_scale_cache.pkl"))
    ap.add_argument("--pos_field", default="position",
                    help="field to measure; use position_before to match a raw_position loader")
    args = ap.parse_args()

    global POS_FIELD
    POS_FIELD = args.pos_field
    print(f"[INFO] pos_field = {POS_FIELD}")

    species_dirs = collect_by_species(args.bvh_pose_root)
    scale_dict = {}
    skipped = []

    for sp in tqdm(sorted(species_dirs), desc="scale cache"):
        max_abs, n_used = species_global_scale(species_dirs[sp])
        if n_used == 0 or max_abs < EPS:
            skipped.append((sp, f"no valid pose (n_used={n_used}, max_abs={max_abs:.3g})"))
            continue
        scale_dict[sp] = {
            "global_scale": float(max_abs),
            "bbox_center": np.zeros(3, dtype=np.float32),
        }
        print(f"  {sp:<16} global_scale={max_abs:.6f}  (npz used={n_used})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(scale_dict, f)

    print(f"\n✅ wrote scale cache for {len(scale_dict)} species -> {args.output}")
    if skipped:
        print(f"⚠️ skipped {len(skipped)} species (no usable pose):")
        for sp, why in skipped:
            print(f"   {sp}: {why}")


if __name__ == "__main__":
    main()

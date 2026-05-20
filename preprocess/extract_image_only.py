"""
Strip the `latent` field from npz_train/* to produce npz_train_image_only/*.

Input  : zoo/npz_train/{motion}/{view}.npz   keys: latent, image_embed
Output : zoo/npz_train_image_only/{motion}/{view}.npz   key: image_embed
"""
import argparse
import os
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


def process_one(task):
    in_npz, out_npz = task
    try:
        if os.path.exists(out_npz):
            return f"[SKIP] {out_npz}"
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        data = np.load(in_npz)
        if "image_embed" not in data.files:
            return f"[WARN] no image_embed in {in_npz}"
        np.savez_compressed(out_npz, image_embed=data["image_embed"])
        return f"[OK] {out_npz}"
    except Exception as e:
        return f"[ERROR] {in_npz}: {e}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", default="zoo/npz_train")
    p.add_argument("--output_root", default="zoo/npz_train_image_only")
    p.add_argument("--num_workers", type=int, default=min(32, cpu_count()))
    return p.parse_args()


def main():
    args = parse_args()
    in_files = sorted(glob(os.path.join(args.input_root, "*", "*.npz")))
    print(f"Found {len(in_files)} files under {args.input_root}.")

    tasks = [
        (f, os.path.join(args.output_root, os.path.relpath(f, args.input_root)))
        for f in in_files
    ]

    with Pool(processes=min(args.num_workers, max(len(tasks), 1))) as pool:
        for result in tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)):
            print(result)


if __name__ == "__main__":
    main()

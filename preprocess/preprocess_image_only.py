"""
Encode per-frame images into per-sequence training npz (image embed only).

Use this when you do NOT have animated meshes (i.e. you skipped stages 7-9 of the
preprocessing pipeline). The output drops the `latent` field and matches what
`loader_v2.py` expects, so it can be consumed directly without running stage 11.

Reads:
    {dataset_root}/image/{motion}/{view}/*.png            (one per frame)

Writes:
    {dataset_root}/npz_train_image_only/{motion}/{view}.npz   (image_embed)
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.v1.video2mesh.pipeline_triposg import TripoSGPipeline
from preprocess.briarmbg import BriaRMBG
from preprocess.image_process import prepare_image


def process_sample(img_files, pipe, rmbg_net, device, dtype, out_file):
    img_pil_list = []
    for image_file in img_files:
        try:
            img_pil = prepare_image(image_file, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
            img_pil_list.append(img_pil)
        except Exception as e:
            print(f"Failed preprocessing {image_file}: {e}")

    with torch.no_grad():
        batch_size = 512  # reduce if GPU memory is tight
        all_embeds = []
        for i in range(0, len(img_pil_list), batch_size):
            chunk = pipe.feature_extractor_dinov2(
                img_pil_list[i:i + batch_size], return_tensors="pt"
            ).pixel_values.to(device, dtype=dtype)
            embeds_chunk = pipe.image_encoder_dinov2(chunk).last_hidden_state
            all_embeds.append(embeds_chunk.cpu())
        image_embeds = torch.cat(all_embeds, dim=0)  # [T, 257, 1024]

    np.savez_compressed(
        out_file,
        image_embed=image_embeds.cpu().numpy(),
    )


def find_view_dirs(image_root):
    """Return sorted list of (motion, view) pairs under image_root/{motion}/{view}/*.png."""
    pairs = []
    if not os.path.isdir(image_root):
        return pairs
    for motion in sorted(os.listdir(image_root)):
        motion_dir = os.path.join(image_root, motion)
        if not os.path.isdir(motion_dir):
            continue
        for view in sorted(os.listdir(motion_dir)):
            view_dir = os.path.join(motion_dir, view)
            if os.path.isdir(view_dir):
                pairs.append((motion, view))
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0,
                        help="current shard id (default 0)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="total number of shards (default 1, single GPU)")
    parser.add_argument("--dataset_root", type=str, default="zoo",
                        help="dataset root containing image/")
    parser.add_argument("--triposg_weights_dir", type=str, default="checkpoints/TripoSG")
    parser.add_argument("--rmbg_weights_dir", type=str, default="checkpoints/RMBG-1.4")
    args = parser.parse_args()

    image_folder = os.path.join(args.dataset_root, "image")
    output_folder = os.path.join(args.dataset_root, "npz_train_image_only")

    print(f"Root Dir:   {args.dataset_root}")
    print(f" -> Image:  {image_folder}")
    print(f" -> Output: {output_folder}")

    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"{image_folder} does not exist; run stage 6 (video_to_images) first.")

    os.makedirs(output_folder, exist_ok=True)
    device = "cuda"
    dtype = torch.float16

    rmbg_net = BriaRMBG.from_pretrained(args.rmbg_weights_dir).to(device)
    rmbg_net.eval()

    print("Loading TripoSG pipeline (DINOv2 image encoder only)...")
    pipe = TripoSGPipeline.from_pretrained(args.triposg_weights_dir).to(device, dtype)

    view_pairs = find_view_dirs(image_folder)
    view_pairs = [p for i, p in enumerate(view_pairs) if i % args.num_shards == args.shard]
    print(f"[INFO] shard {args.shard}/{args.num_shards}, will process {len(view_pairs)} sequences")

    for motion, view in tqdm(view_pairs, desc=f"Shard {args.shard}", unit="seq"):
        images_dir = os.path.join(image_folder, motion, view)
        out_file = os.path.join(output_folder, motion, view + ".npz")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        if os.path.exists(out_file):
            tqdm.write(f"[SKIP] {out_file} already exists, skipping.")
            continue

        img_files = sorted([
            os.path.join(images_dir, x)
            for x in os.listdir(images_dir)
            if x.lower().endswith((".jpg", ".png"))
        ])

        if not img_files:
            tqdm.write(f"[WARN] {images_dir} contains no images, skipping.")
            continue

        try:
            process_sample(img_files, pipe, rmbg_net,
                           device=device, dtype=dtype, out_file=out_file)
            tqdm.write(f"[DONE] {images_dir} -> {out_file}, {len(img_files)} frames")
        except Exception as e:
            tqdm.write(f"[ERROR] {images_dir} processing failed: {e}")

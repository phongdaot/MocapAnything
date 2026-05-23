"""
Encode per-frame images into image-only training npz (skips the mesh VAE).

Reads:
    {dataset_root}/image/{motion}/{view}/*.png      (one per frame)

Writes:
    {dataset_root}/npz_train_image_only/{motion}/{view}.npz  (image_embed)

Use this when you only need the DINOv2 image features and don't have (or
don't care about) the mesh latents. The full mesh+image variant lives in
preprocess_data.py.
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


def encode_image_folder(images_dir, pipe, rmbg_net, device, dtype, batch_size=512):
    """Background-remove + DINOv2-encode every image in `images_dir`.
    Returns image_embeds tensor of shape (T, 257, 1024) on CPU."""
    img_files = sorted([
        os.path.join(images_dir, x)
        for x in os.listdir(images_dir)
        if x.lower().endswith((".jpg", ".png"))
    ])
    if not img_files:
        return None

    img_pil_list = []
    for image_file in img_files:
        try:
            img_pil = prepare_image(
                image_file,
                bg_color=np.array([1.0, 1.0, 1.0]),
                rmbg_net=rmbg_net,
            )
            img_pil_list.append(img_pil)
        except Exception as e:
            print(f"Failed preprocessing {image_file}: {e}")

    with torch.no_grad():
        all_embeds = []
        for i in range(0, len(img_pil_list), batch_size):
            chunk = pipe.feature_extractor_dinov2(
                img_pil_list[i:i + batch_size], return_tensors="pt"
            ).pixel_values.to(device, dtype=dtype)
            embeds_chunk = pipe.image_encoder_dinov2(chunk).last_hidden_state
            all_embeds.append(embeds_chunk.cpu())
        return torch.cat(all_embeds, dim=0)


def find_all_view_dirs(image_root):
    """Yield (motion, view, abs_dir) for every {motion}/{view}/ folder."""
    for motion in sorted(os.listdir(image_root)):
        motion_dir = os.path.join(image_root, motion)
        if not os.path.isdir(motion_dir):
            continue
        for view in sorted(os.listdir(motion_dir)):
            view_dir = os.path.join(motion_dir, view)
            if os.path.isdir(view_dir):
                yield motion, view, view_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0,
                        help="current shard id (default 0)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="total number of shards (default 1, single GPU)")
    parser.add_argument("--dataset_root", type=str, default="zoo",
                        help="dataset root containing image/, npz_train_image_only/")
    parser.add_argument("--triposg_weights_dir", type=str, default="checkpoints/TripoSG")
    parser.add_argument("--rmbg_weights_dir", type=str, default="checkpoints/RMBG-1.4")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="images per DINOv2 batch; reduce if GPU memory is tight")
    args = parser.parse_args()

    image_folder = os.path.join(args.dataset_root, "image")
    output_folder = os.path.join(args.dataset_root, "npz_train_image_only")

    print(f"Root Dir:   {args.dataset_root}")
    print(f" -> Image:  {image_folder}")
    print(f" -> Output: {output_folder}")

    if not os.path.exists(image_folder):
        raise SystemExit(f"[ERROR] {image_folder} does not exist.")
    os.makedirs(output_folder, exist_ok=True)

    device = "cuda"
    dtype = torch.float16

    rmbg_net = BriaRMBG.from_pretrained(args.rmbg_weights_dir).to(device)
    rmbg_net.eval()

    print("Loading TripoSG pipeline (for DINOv2 encoder)...")
    pipe = TripoSGPipeline.from_pretrained(args.triposg_weights_dir).to(device, dtype)

    view_dirs = list(find_all_view_dirs(image_folder))
    view_dirs = [v for i, v in enumerate(view_dirs) if i % args.num_shards == args.shard]
    print(f"[INFO] shard {args.shard}/{args.num_shards}, "
          f"will process {len(view_dirs)} (motion, view) pairs")

    for motion, view, view_dir in tqdm(view_dirs, desc=f"Shard {args.shard}", unit="seq"):
        out_file = os.path.join(output_folder, motion, f"{view}.npz")
        if os.path.exists(out_file):
            tqdm.write(f"[SKIP] {out_file}")
            continue
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        try:
            image_embeds = encode_image_folder(
                view_dir, pipe, rmbg_net, device, dtype, batch_size=args.batch_size,
            )
            if image_embeds is None:
                tqdm.write(f"[WARN] {view_dir} contains no images, skipping.")
                continue

            np.savez_compressed(out_file, image_embed=image_embeds.numpy())
            tqdm.write(f"[DONE] {view_dir} -> {out_file}, {image_embeds.shape[0]} frames")
        except Exception as e:
            tqdm.write(f"[ERROR] {view_dir} processing failed: {e}")

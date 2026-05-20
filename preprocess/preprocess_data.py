"""
Encode normalized meshes + per-frame images into per-sequence training npz.

Reads:
    {dataset_root}/npz_mesh_normed/{motion}/{view}.npz   (vertices_normed, normals)
    {dataset_root}/image/{motion}/{view}/*.png            (one per frame)

Writes:
    {dataset_root}/npz_train/{motion}/{view}.npz         (latent, image_embed)
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TripoSG.triposg.models.autoencoders import TripoSGVAEModel

from models.v1.video2mesh.pipeline_triposg import TripoSGPipeline
from preprocess.briarmbg import BriaRMBG
from preprocess.image_process import prepare_image


def batched_encode(vae, surface, max_batch=4):
    """Encode `surface` ([T, N, C]) through the VAE in chunks to bound GPU memory."""
    latents = []
    for i in range(0, surface.shape[0], max_batch):
        with torch.no_grad():
            latent = vae.encode(surface[i:i + max_batch]).latent_dist.mean
        latents.append(latent.cpu())
    return torch.cat(latents, dim=0)


def process_sample(surface_npy, img_files, vae, pipe, rmbg_net, device, dtype, out_file):
    # 1. Load 3D surface and sampled points
    surface = np.load(surface_npy, allow_pickle=True)
    surface_pts = surface["vertices_normed"]  # [T, N, 3]
    normal_pts = surface["normals"]           # [T, N, 3]
    print('surface_pts.shape: ', surface_pts.shape)
    print('len(img_files): ', len(img_files))

    assert len(img_files) == surface_pts.shape[0], (
        "Number of frames does not match number of images!"
    )

    # Sample 8192 surface points independently per frame.
    num_pc = 8192
    rng = np.random.default_rng()
    surface_tensor_list = []
    normal_tensor_list = []
    for frame_idx in range(surface_pts.shape[0]):
        num_points = surface_pts.shape[1]
        if num_points >= num_pc:
            ind = rng.choice(num_points, num_pc, replace=False)
        else:
            print(f"[WARN] num_pc ({num_pc}) > num_points ({num_points}), sampling with replacement")
            ind = rng.choice(num_points, num_pc, replace=True)
        surface_tensor_list.append(torch.FloatTensor(surface_pts[frame_idx, ind]))
        normal_tensor_list.append(torch.FloatTensor(normal_pts[frame_idx, ind]))

    surface_tensor = torch.stack(surface_tensor_list, dim=0)            # [T, num_pc, 3]
    normal_tensor = torch.stack(normal_tensor_list, dim=0)              # [T, num_pc, 3]
    surface_cat = torch.cat([surface_tensor, normal_tensor], dim=-1)    # [T, num_pc, 6]
    surface_cat = surface_cat.to(device, dtype=dtype)

    # 2. VAE latent encode
    vae.eval()
    latents = batched_encode(vae, surface_cat, max_batch=64)

    # 3. Background-removal + DINOv2 image encode in batches
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

    assert latents.shape[0] == image_embeds.shape[0], "Frame counts do not match"

    np.savez_compressed(
        out_file,
        latent=latents.detach().cpu().numpy(),
        image_embed=image_embeds.cpu().numpy(),
    )


def find_all_npz(root):
    """Recursively find every .npz under `root`."""
    npz_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".npz"):
                npz_files.append(os.path.join(dirpath, f))
    return sorted(npz_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0,
                        help="current shard id (default 0)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="total number of shards (default 1, single GPU)")
    parser.add_argument("--dataset_root", type=str, default="zoo",
                        help="dataset root containing image/, npz_mesh_normed/, ...")
    parser.add_argument("--triposg_weights_dir", type=str, default="checkpoints/TripoSG")
    parser.add_argument("--rmbg_weights_dir", type=str, default="checkpoints/RMBG-1.4")
    args = parser.parse_args()

    image_folder = os.path.join(args.dataset_root, "image")
    npz_mesh_folder = os.path.join(args.dataset_root, "npz_mesh_normed")
    output_folder = os.path.join(args.dataset_root, "npz_train")

    print(f"Root Dir:   {args.dataset_root}")
    print(f" -> Image:  {image_folder}")
    print(f" -> Mesh:   {npz_mesh_folder}")
    print(f" -> Output: {output_folder}")

    if not os.path.exists(image_folder):
        print(f"Warning: {image_folder} does not exist!")

    os.makedirs(output_folder, exist_ok=True)
    device = "cuda"
    dtype = torch.float16

    vae: TripoSGVAEModel = TripoSGVAEModel.from_pretrained(
        args.triposg_weights_dir,
        subfolder="vae",
    ).to(device, dtype=dtype)

    rmbg_net = BriaRMBG.from_pretrained(args.rmbg_weights_dir).to(device)
    rmbg_net.eval()

    print("Loading TripoSG pipeline...")
    pipe = TripoSGPipeline.from_pretrained(args.triposg_weights_dir).to(device, dtype)

    npz_list = find_all_npz(npz_mesh_folder)
    npz_list = [x for i, x in enumerate(npz_list) if i % args.num_shards == args.shard]
    print(f"[INFO] shard {args.shard}/{args.num_shards}, will process {len(npz_list)} npz files")

    for surface_npy in tqdm(npz_list, desc=f"Shard {args.shard}", unit="file"):
        rel_path = os.path.relpath(surface_npy, npz_mesh_folder)
        obj_name = os.path.dirname(rel_path)
        angle_name = os.path.splitext(os.path.basename(rel_path))[0]

        out_file = os.path.join(output_folder, rel_path)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        images_dir = os.path.join(image_folder, obj_name, angle_name)

        if os.path.exists(out_file):
            tqdm.write(f"[SKIP] {out_file} already exists, skipping.")
            continue

        if not os.path.exists(images_dir):
            tqdm.write(f"[WARN] {images_dir} does not exist, skipping.")
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
            process_sample(surface_npy, img_files, vae, pipe, rmbg_net,
                           device=device, dtype=dtype, out_file=out_file)
            tqdm.write(f"[DONE] {surface_npy} -> {out_file}, {len(img_files)} frames")
        except Exception as e:
            tqdm.write(f"[ERROR] {surface_npy} processing failed: {e}")

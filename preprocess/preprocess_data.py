
### preprocess_data.py ###
# scripts/train_s1_preprocess_data_1002
import os
import sys
import time
from datetime import datetime
from PIL import Image
import math
import torch
import torch.nn.functional as F  
import trimesh
from tqdm import tqdm
import argparse
import numpy as np
import torch
import trimesh
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TripoSG.triposg.inference_utils import hierarchical_extract_geometry, flash_extract_geometry
# from triposg.models.autoencoders import TripoSGTVAEModel
from TripoSG.triposg.models.autoencoders import TripoSGVAEModel
from huggingface_hub import snapshot_download
sys.path.append('/home/ma-user/work/g00826954/projects/proj_4d/TripoSGbase')

# from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from ..models.v1.video2mesh.pipeline_triposg import TripoSGPipeline
from .image_process import prepare_image
from .briarmbg import BriaRMBG
import numpy as np



"""
0729: Found that using chunk processing causes different FPS within the same sequence,
      so slicing cannot be used here.
0801: Batch processing of a group of data.
0813: Crop is enabled by default, so the processed data is centered.
      In the future we should re-render and scale to a fixed bounding box
      and define a unified camera.
1002: Added support for handling npz files at arbitrary directory levels,
      and added a simple method for processing simultaneously on 8 GPUs.
"""

def batched_encode(vae, surface, max_batch=4):
    """
    surface: [T, N, C]  torch.Tensor
    vae: TripoSGTVAEModel
    max_batch: number of frames to encode each time (adjust based on GPU memory)
    return: [T, latent_dim] or [T, ...]
    """
    latents = []
    T = surface.shape[0]
    for i in range(0, T, max_batch):
        # slice batch
        chunk = surface[i:i+max_batch]
        with torch.no_grad():
            latent = vae.encode(chunk).latent_dist.mean   # [B, latent_dim]
        latents.append(latent.cpu())
    return torch.cat(latents, dim=0)


def process_sample(
    surface_npy, img_files, vae, pipe, device, dtype, out_dir
):
    # 1. Load 3D surface and sampled points
    surface = np.load(surface_npy, allow_pickle=True) #.tolist()
    surface_pts = surface["vertices_normed"]  # [T, N, 3]
    normal_pts = surface["normals"]  # [T, N, 3]
    print('surface_pts.shape: ', surface_pts.shape)
    print('len(img_files): ', len(img_files))

    assert len(img_files) == surface_pts.shape[0], "Number of frames does not match number of images!"

    # rng = np.random.default_rng()
    # ind = rng.choice(surface_pts.shape[1], 8192, replace=False)
    # For the same asset model, shold the same 8192 points also be shared 
    # surface_tensor = torch.FloatTensor(surface_pts[:, ind])
    # normal_tensor = torch.FloatTensor(normal_pts[:, ind])
    # surface_cat = torch.cat([surface_tensor, normal_tensor], dim=-1).to(device, dtype=dtype)

    # change to sampling independently for each frame
    rng = np.random.default_rng()
    surface_tensor_list = []
    normal_tensor_list = []
    for frame_idx in range(surface_pts.shape[0]):
        if 8192 <= surface_pts.shape[1]:
            ind = rng.choice(surface_pts.shape[1], 8192, replace=False)  # Independent random sampling per frame
        else:
            print(f"[WARN] num_pc ({8192}) > num_points ({surface_pts.shape[1]}), sampling with replacement")
            ind = rng.choice(surface_pts.shape[1], 8192, replace=True)
        surface_tensor_list.append(torch.FloatTensor(surface_pts[frame_idx, ind]))
        normal_tensor_list.append(torch.FloatTensor(normal_pts[frame_idx, ind]))
    # [T, 8192, 3]
    surface_tensor = torch.stack(surface_tensor_list, dim=0)
    normal_tensor = torch.stack(normal_tensor_list, dim=0)
    # 拼接 [T, 8192, 6]
    surface_cat = torch.cat([surface_tensor, normal_tensor], dim=-1).to(device, dtype=dtype)

    # 2. VAE latent encode
    vae.eval()
    # latents = vae.encode(surface_cat).latent_dist.mean # [T, latent_dim]
    latents = batched_encode(vae, surface_cat, max_batch=64)
    
    # 3. Load images and batch encode
    # Centralized loading and preprocessing
    img_pil_list = []

    for image_file in img_files:
        try:
            img_pil = prepare_image(image_file, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
            img_pil_list.append(img_pil)
        except Exception as e:
            print(f"Failed preprocessing {image_file}: {e}")

    image = img_pil_list
    
    # with torch.no_grad():
    #     image = pipe.feature_extractor_dinov2(image, return_tensors="pt").pixel_values  # torch.Size([B, 3, 224, 224])
    #     image_embeds = pipe.image_encoder_dinov2(image.to(device, dtype=dtype)).last_hidden_state   # torch.Size([B, 257, 1024])

    with torch.no_grad():
        batch_size = 512  # number of images per batch, can reduce to 16 or 8 if needed
        all_embeds = []

        for i in range(0, len(image), batch_size):
            image_chunk = image[i:i + batch_size]
            image_chunk = pipe.feature_extractor_dinov2(
                image_chunk, return_tensors="pt"
            ).pixel_values.to(device, dtype=dtype)

            embeds_chunk = pipe.image_encoder_dinov2(image_chunk).last_hidden_state
            all_embeds.append(embeds_chunk.cpu())  # move to CPU temporarily to avoid GPU memory

        image_embeds = torch.cat(all_embeds, dim=0)  # [3000, 257, 1024]

    assert latents.shape[0] == image_embeds.shape[0], "Frame counts do not match"
    
    # # 4. 保存每帧
    # for i in range(latents.shape[0]):
    #     np.savez_compressed(
    #         os.path.join(out_dir, f"{i:04d}.npz"),
    #         latent=latents[i].detach().cpu().numpy(),        # (latent_dim,)
    #         image_embed=image_embeds[i].cpu().numpy()  # (257, 1024)
    #     )

    # 保存一个整个npz
    np.savez_compressed(
        out_dir,
        latent=latents.detach().cpu().numpy(),        # (latent_dim,)
        image_embed=image_embeds.cpu().numpy()  # (257, 1024)
    )

    
def find_all_npz(root):
    """递归找到 root 下所有 .npz 文件"""
    npz_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".npz"):
                full_path = os.path.join(dirpath, f)
                npz_files.append(full_path)
    return sorted(npz_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="current shard id (default 0)")
    parser.add_argument("--num_shards", type=int, default=1, help="total number of shards (default 1, meaning single GPU)")
    
    # === new path parameter ===
    # === simplified path parameter: only pass root directory ===
    parser.add_argument("--dataset_root", type=str, 
                        default="./dataset/zoo1030", 
                        help="dataset root directory (will automatically locate image, npz_mesh_normed, etc. under it)")
    args = parser.parse_args()

    # === automatically construct paths ===
    # assuming the directory structure is fixed
    image_folder = os.path.join(args.dataset_root, "image")
    npz_mesh_folder = os.path.join(args.dataset_root, "npz_mesh_normed")
    output_folder = os.path.join(args.dataset_root, "npz_train")

    # print paths to verify correctness
    print(f"Root Dir:   {args.dataset_root}")
    print(f" -> Image:  {image_folder}")
    print(f" -> Mesh:   {npz_mesh_folder}")
    print(f" -> Output: {output_folder}")

    # check whether paths exist to avoid errors halfway through
    if not os.path.exists(image_folder):
        print(f"Warning: {image_folder} does not exist!")
    
    os.makedirs(output_folder, exist_ok=True)
    device = "cuda"
    dtype = torch.float16

    # load VAE
    triposg_weights_dir = "checkpoints/TripoSG"
    vae: TripoSGVAEModel = TripoSGVAEModel.from_pretrained(
        triposg_weights_dir,
        subfolder="vae",
    ).to(device, dtype=dtype)

    # load RMBG
    rmbg_weights_dir = "checkpoints/RMBG-1.4"
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval()

    # load pipeline
    print("Loading TripoSG pipeline...")
    pipe = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)

    # traverse all npz
    npz_list = find_all_npz(npz_mesh_folder)

    # shard split
    npz_list = [x for i, x in enumerate(npz_list) if i % args.num_shards == args.shard]
    print(f"[INFO] shard {args.shard}/{args.num_shards}, will process {len(npz_list)} npz files")

    # tqdm progress bar
    for surface_npy in tqdm(npz_list, desc=f"Shard {args.shard}", unit="file"):
        # relative path
        rel_path = os.path.relpath(surface_npy, npz_mesh_folder)

        # split object name and angle
        obj_name = os.path.dirname(rel_path)
        angle_name = os.path.splitext(os.path.basename(rel_path))[0]

        # output path (keep directory structure)
        out_file = os.path.join(output_folder, rel_path)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # corresponding image directory
        images_dir = os.path.join(image_folder, obj_name, angle_name)

        if os.path.exists(out_file):
            tqdm.write(f"[SKIP] {out_file} already exists, skipping.")
            continue

        if not os.path.exists(images_dir):
            tqdm.write(f"[WARN] {images_dir} does not exist, skipping.")
            continue

        # find images
        img_files = sorted([
            os.path.join(images_dir, x)
            for x in os.listdir(images_dir)
            if x.lower().endswith((".jpg", ".png"))
        ])

        if len(img_files) == 0:
            tqdm.write(f"[WARN] {images_dir} contains no images, skipping.")
            continue

        try:
            process_sample(surface_npy, img_files, vae, pipe, device=device, dtype=dtype, out_dir=out_file)
            tqdm.write(f"[DONE] {surface_npy} \n {images_dir} \n {out_file}, {len(img_files)} frames")
        except Exception as e:
            tqdm.write(f"[ERROR] {surface_npy} processing failed: {e}")
            
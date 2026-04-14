### video2mesh.py ###
# scripts/inference_triposg_kh0821b_batchframeprocess.py

import os
import sys
from PIL import Image
import torch
import trimesh
import numpy as np
import argparse
from utils.logger import logger
from utils.config_utils import load_yaml_config, get_obj_from_str

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.v1.video2mesh.pipeline_triposg import DiffusionPipeline
from preprocess.image_process import prepare_image
from preprocess.briarmbg import BriaRMBG

"""
0611: Batch process all folders inside selected_nbg, single-frame processing.
0627a: First generate results sequentially for 30 frames.
0627b: Then generate results in parallel for 30 frames and compare.
0722b: Try using a modified model to generate sequence results.
- Assume input images are from a folder, then process them into batches according to seq length, e.g. 16 together, shape becomes 1,16,S2,D2
- Latent is also 1,16,S1,D1.
- Replace model with modified model
0805a: Try a temporal pipeline that can automatically run the modified temporal model.
0805c: Full attention version
0821: Use original attention but add sliding window
0821a: Full window length
0821b: Adaptive — if short use all-in, if long split into segments
"""


def str2list(v):
    if isinstance(v, list):
        return v
    return [int(x) for x in v.split(",")]


def get_torch_dtype(dtype_str: str):
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype_map[dtype_str]


def video2mesh(cfg):

    attention_cfg = cfg["model"]["params"]["attention_kwargs"]

    # ========= Configuration ==========
    base_data_path = cfg["paths"]["base_data_path"]
    test_folders = cfg["paths"]["test_folder"]
    output_base = cfg["paths"]["output_base"]

    no_imgprepare_crop = cfg["preprocess"]["no_imgprepare_crop"]
    transformer_path = cfg["weights"]["transformer_path"]
    reverse = cfg["inference"]["reverse"]

    selfatt_temporal_layer_flag = str2list(attention_cfg["selfatt_temporal_layer_flag"])
    selfatt_temporal = attention_cfg["selfatt_temporal"]
    crossatt2_temporal = attention_cfg["crossatt2_temporal"]
    selfatt_slidwindow = attention_cfg["selfatt_slidwindow"]
    crossatt2_slidwindow = attention_cfg["crossatt2_slidwindow"]
    seq_len = attention_cfg["seq_len"]

    triposg_weights_dir = cfg["weights"]["triposg_weights_dir"]
    rmbg_weights_dir = cfg["weights"]["rmbg_weights_dir"]

    device = cfg["runtime"]["device"]
    dtype = get_torch_dtype(cfg["runtime"]["dtype"])
    seed = cfg["runtime"]["seed"]

    num_inference_steps = cfg["model"]["params"]["num_inference_steps"]
    guidance_scale = cfg["inference"]["guidance_scale"]
    faces = cfg["inference"]["faces"]

    bg_color = np.array(cfg["preprocess"]["bg_color"], dtype=np.float32)
    max_images = cfg["preprocess"]["max_images"]

    attention_design = {
        "seq_len": seq_len,
        "selfatt_temporal_layer_flag": selfatt_temporal_layer_flag,
        "selfatt_temporal": selfatt_temporal,
        "crossatt2_temporal": crossatt2_temporal,
        "selfatt_slidwindow": selfatt_slidwindow,
        "crossatt2_slidwindow": crossatt2_slidwindow,
    }

    logger.info("Loading RMBG...")
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval()

    logger.info("Loading TripoSG pipeline...")
    
    obj: DiffusionPipeline = get_obj_from_str(cfg["model"]["target"])
    pipe = obj.from_pretrained(triposg_weights_dir).to(device, dtype)

    MAX_SEQ_LEN = seq_len  # Maximum all-in length allowed by GPU memory

    for dataset_type in test_folders:
        dataset_path = os.path.join(base_data_path, dataset_type)
        if not os.path.exists(dataset_path):
            logger.info(f"dataset_path not exists: {dataset_path}")
            continue

        for sequence_name in sorted(os.listdir(dataset_path), reverse=reverse):
            sequence_path = os.path.join(dataset_path, sequence_name)
            if not os.path.isdir(sequence_path):
                logger.info(f"sequence_path is not dir: {sequence_path}")
                continue

            logger.info(f"Processing sequence: {dataset_type}/{sequence_name}")
            output_root = os.path.join(output_base, dataset_type, sequence_name)
            os.makedirs(output_root, exist_ok=True)

            image_files = sorted(
                [f for f in os.listdir(sequence_path) if f.endswith(".png") or f.endswith(".jpg")]
            )
            if max_images:
                image_files = image_files[:max_images]
            img_pil_list, image_indices = [], []

            for image_file in image_files:
                image_index = os.path.splitext(image_file)[0]
                image_path = os.path.join(sequence_path, image_file)
                try:
                    img_pil = prepare_image(
                        image_path,
                        bg_color=bg_color,
                        rmbg_net=rmbg_net,
                    )
                    img_pil_list.append(img_pil)
                    image_indices.append(image_index)
                except Exception as e:
                    logger.info(f"Failed preprocessing {image_file}: {e}")

            # Skip already processed sequences
            already_exist = True
            for idx in image_indices:
                glb_path = os.path.join(output_root, f"{idx}.glb")
                if not os.path.exists(glb_path):
                    already_exist = False
                    break

            if already_exist and len(os.listdir(output_root)) >= len(image_indices):
                logger.info(f"[{dataset_type}/{sequence_name}] Output exists and has all glb files, skip.")
                continue

            n = len(img_pil_list)
            logger.info(f"Running TripoSG pipeline on {n} images...")

            # ===================== Adaptive inference ======================
            if n <= MAX_SEQ_LEN:
                try:
                    outputs = pipe(
                        image=img_pil_list,
                        generator=torch.Generator(device=device).manual_seed(seed),
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        attention_kwargs={**attention_design, "seq_len": n},
                    ).samples
                except Exception as e:
                    logger.info(f"Pipeline inference failed: {e}")
                    continue
                all_outputs = outputs

            else:
                window_candidates = [selfatt_slidwindow, crossatt2_slidwindow]
                window_candidates = [v for v in window_candidates if v is not None]

                if len(window_candidates) == 0:
                    raise ValueError("When n > seq_len, at least one sliding window size must be set.")

                window = max(window_candidates)
                padding = (window - 1) // 2

                chunks, index_chunks, starts = [], [], []
                i = 0

                while i + seq_len <= n:
                    chunks.append(img_pil_list[i:i + seq_len])
                    index_chunks.append(image_indices[i:i + seq_len])
                    starts.append(i)
                    i += seq_len - 2 * padding

                if i < n:
                    chunks.append(img_pil_list[-seq_len:])
                    index_chunks.append(image_indices[-seq_len:])
                    starts.append(n - seq_len)

                all_outputs, last_kept = [], -1

                for batch_idx, (batch_imgs, batch_indices, start) in enumerate(zip(chunks, index_chunks, starts)):
                    try:
                        outputs = pipe(
                            image=batch_imgs,
                            generator=torch.Generator(device=device).manual_seed(seed),
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            attention_kwargs={**attention_design, "seq_len": seq_len},
                        ).samples
                    except Exception as e:
                        logger.info(f"Pipeline chunk inference failed: {e}")
                        continue

                    if batch_idx == 0:
                        keep_start = 0
                        keep_end = seq_len - padding
                    elif start + seq_len < n:
                        keep_start = padding
                        keep_end = seq_len - padding
                    else:
                        keep_start = max(0, last_kept - start + 1)
                        keep_end = seq_len

                    kept_outputs = outputs[keep_start:keep_end]
                    all_outputs.extend(kept_outputs)
                    last_kept = start + keep_end - 1

            logger.info(f"len(all_outputs)={len(all_outputs)}, len(img_pil_list)={len(img_pil_list)}")
            assert len(all_outputs) == n, f"Output length {len(all_outputs)} != original frame count {n}"

            for i, (output_verts, output_faces) in enumerate(all_outputs):
                try:
                    mesh = trimesh.Trimesh(output_verts.astype(np.float32), np.ascontiguousarray(output_faces))

                    if faces > 0 and mesh.faces.shape[0] > faces:
                        try:
                            import pymeshlab

                            def mesh_to_pymesh(vertices, faces):
                                mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
                                ms = pymeshlab.MeshSet()
                                ms.add_mesh(mesh)
                                return ms

                            def pymesh_to_trimesh(mesh):
                                verts = mesh.vertex_matrix()
                                faces = mesh.face_matrix()
                                return trimesh.Trimesh(vertices=verts, faces=faces)

                            ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
                            ms.meshing_merge_close_vertices()
                            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=faces)
                            mesh = pymesh_to_trimesh(ms.current_mesh())

                        except Exception as e:
                            logger.info(f"[Mesh simplify failed: {e}] ")

                    mesh_path = os.path.join(output_root, f"{image_indices[i]}.glb")
                    mesh.export(mesh_path)
                    logger.info(f"Saved: {mesh_path}")

                except Exception as e:
                    logger.info(f"Failed saving mesh for {image_indices[i]}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Video2Mesh")
    parser.add_argument("--config", type=str, default="configs/inference/inference_video2mesh.yaml", help="Path to the YAML config file")
    args = parser.parse_args()
    
    cfg = load_yaml_config(args.config)
    video2mesh(cfg)
### mesh2pose.py ###
import os
import random
import numpy as np
import torch
from utils.logger import logger
from utils.common import *
from utils.config_utils import load_yaml_config, instantiate_from_config
from preprocess.image_process import prepare_image
from preprocess.briarmbg import BriaRMBG
from TripoSG.triposg.pipelines.pipeline_triposg import TripoSGPipeline
from utils.common import plot_pose_compare_from_npy, load_surface_from_glb_folder, extract_and_compare_image_features_with_rmbg
from utils.common import find_all_valid_glb_sequences, find_image_folder
import argparse

def inference(cfg, device, attention_design, model, pipe, rmbg_net, seq_name, glb_folder, image_folder):
    """
    Run inference on a single sequence.
    All parameters are read from the config.
    """
    base_dir = cfg["data"]["base_dir"]
    ref_seq = cfg["data"]["retarget"]["ref_seq"]
    ref_idx = cfg["data"]["retarget"]["ref_idx"]
    save_dir_root = cfg["output"]["save_dir"]
    wild_flag = cfg["data"]["wild_flag"]
    bvh_roots = cfg["data"]["bvh_roots"]

    # === Load reference data ===
    pose_path = os.path.join(base_dir, "bvh_pose", f"{ref_seq}.npz")
    mesh_path = os.path.join(base_dir, "npz_mesh_normed", f"{ref_seq}.npz")
    train_path = os.path.join(base_dir, "npz_train", f"{ref_seq}.npz")
    species_info_path = os.path.join(base_dir, "species_info_dict.npy")

    for path in [pose_path, mesh_path, train_path, species_info_path]:
        if not os.path.exists(path):
            logger.warning(f"[SKIP] Required file not found: {path}")
            return

    pose_npz = np.load(pose_path)
    mesh_npz = np.load(mesh_path)
    train_npz = np.load(train_path)
    species_info_dict = np.load(species_info_path, allow_pickle=True).item()

    position = pose_npz["position"]
    ref_image_embed_all = train_npz["image_embed"]
    ref_vertices_all = mesh_npz["vertices_normed"]
    ref_normals_all = mesh_npz["normals"]

    species_name = ref_seq.split("#")[0]
    gscale = mesh_npz["global_scale"]

    graph_hop = torch.from_numpy(species_info_dict[species_name]["joints_distance"]).unsqueeze(0).to(torch.int64)
    graph_edge = torch.from_numpy(species_info_dict[species_name]["joint_relation"]).unsqueeze(0).to(torch.int64)
    joint_t5 = torch.from_numpy(species_info_dict[species_name]["t5_embedding"]).unsqueeze(0)

    # === Static joint mask ===
    static_joint_ids = species_info_dict.get(species_name, {}).get("static_joints", [])
    static_mask = np.zeros((graph_hop.shape[-1]), dtype=np.bool_)
    static_mask[static_joint_ids] = True
    static_mask = torch.from_numpy(static_mask).unsqueeze(0)

    pose_npz.close()
    mesh_npz.close()
    train_npz.close()

    # === Normalize pose ===
    position = (position - position[:, 0:1, :]) / gscale
    max_pts = 1024
    F_ref, V_ref = ref_vertices_all.shape[:2]

    # Adjust mesh point count
    if V_ref > max_pts:
        idx = np.random.choice(V_ref, max_pts, replace=False)
        ref_vertices_all = ref_vertices_all[:, idx]
        ref_normals_all = ref_normals_all[:, idx]
    elif V_ref < max_pts:
        pad = np.zeros((F_ref, max_pts - V_ref, 3), dtype=np.float32)
        ref_vertices_all = np.concatenate([ref_vertices_all, pad], axis=1)
        ref_normals_all = np.concatenate([ref_normals_all, pad], axis=1)

    # === Build reference inputs ===
    ref_pos = torch.from_numpy(position[ref_idx]).unsqueeze(0).float().to(device)
    ref_img = torch.from_numpy(ref_image_embed_all[ref_idx]).unsqueeze(0).float().to(device)
    ref_mesh = torch.from_numpy(ref_vertices_all[ref_idx]).unsqueeze(0).float().to(device)
    ref_normal = torch.from_numpy(ref_normals_all[ref_idx]).unsqueeze(0).float().to(device)
    position_seq = torch.from_numpy(position).unsqueeze(0).float().to(device)

    # === Replace with actual folders ===
    image_embed = extract_and_compare_image_features_with_rmbg(
        image_folder=image_folder, rmbg_net=rmbg_net, pipe=pipe
    )
    vertices, normals, _ = load_surface_from_glb_folder(glb_folder=glb_folder)

    image_seq = torch.from_numpy(image_embed).unsqueeze(0).float().to(device)
    mesh_seq = torch.from_numpy(vertices).unsqueeze(0).float().to(device)
    normal_seq = torch.from_numpy(normals).unsqueeze(0).float().to(device)

    joint_mask = torch.ones(position.shape[1], dtype=torch.bool).unsqueeze(0).to(device)
    graph_hop = graph_hop.to(device)
    graph_edge = graph_edge.to(device)
    static_mask = static_mask.to(device)
    joint_t5 = joint_t5.float().to(device)

    # === Handle sequence length mismatch ===
    min_seq_len = min(image_seq.shape[1], mesh_seq.shape[1], normal_seq.shape[1], position_seq.shape[1])
    image_seq = image_seq[:, :min_seq_len]
    mesh_seq = mesh_seq[:, :min_seq_len]
    normal_seq = normal_seq[:, :min_seq_len]
    position_seq = position_seq[:, :min_seq_len]

    attention_design = dict(attention_design)
    attention_design["seq_len"] = image_seq.shape[1]

    batch = {
        "ref_position": ref_pos,
        "ref_image_embed": ref_img,
        "ref_mesh": ref_mesh,
        "ref_surface_normal": ref_normal,
        "image_embed": image_seq,
        "mesh": mesh_seq,
        "surface_normal": normal_seq,
        "joint_mask": joint_mask,
        "static_mask": static_mask,
        "graph_hop": graph_hop,
        "graph_edge": graph_edge,
        "joint_t5embed": joint_t5,
    }

    # === Inference ===
    with torch.no_grad():
        pred = model(batch, attention_kwargs=attention_design)
        pred_np = pred.squeeze(0).cpu().numpy()

    if wild_flag:
        position_seq = pred

    # === Save results ===
    save_subfolder = os.path.relpath(glb_folder, cfg["data"]["glb_root_dir"])
    save_dir = os.path.join(save_dir_root, save_subfolder)
    os.makedirs(save_dir, exist_ok=True)

    species_actual = seq_name.split("#")[0]
    npy_pred_path = os.path.join(save_dir, f"{species_name}_{species_actual}_pred.npy")
    npy_gt_path = os.path.join(save_dir, f"{species_name}_gt.npy")

    np.save(npy_pred_path, pred_np)
    np.save(npy_gt_path, position_seq.squeeze(0).cpu().numpy())
    
    # print(ref_pos.shape, ref_img.shape, ref_mesh.shape, mesh_seq.shape, pred.shape, position_seq.shape)
    
    logger.info(f"[Complete] Saved predictions to {save_dir}")

    plot_pose_compare_from_npy(
        pred_npy_path=npy_pred_path,
        gt_npy_path=npy_gt_path,
        species_name=species_name,
        save_dir=save_dir,
        image_folder=image_folder,
        fps=15,
        wild_flag=wild_flag,
        species_actual=species_actual,
        bvh_roots=bvh_roots,
    )
    
    
    # === Evaluate MPJPE ===

    mpjpe = np.mean(np.linalg.norm(pred_np  - position_seq.squeeze(0).cpu().numpy(), axis=-1))

    logger.info(f"[METRIC] MPJPE (mm): {mpjpe:.6f}")


def mesh2pose(cfg):
    set_seed(cfg["runtime"]["seed"])

    device_str = cfg["runtime"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === Build preprocessing models ===
    rmbg_net = BriaRMBG.from_pretrained(cfg["weights"]["rmbg_weights_dir"]).to(device).eval()
    pipe = TripoSGPipeline.from_pretrained(cfg["weights"]["triposg_weights_dir"]).to(device, torch.float16)

    # === Build prediction model ===
    model: torch.nn.Module = instantiate_from_config(cfg["model"])
    model = model.float().to(device).eval()

    # Load checkpoint
    ckpt_path = os.path.join(cfg["weights"]["mesh2pose_ckpt_root"], cfg["experiment"]["exp"], "mesh2pose_ckpt_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    attention_design = cfg["model"]["attention_kwargs"]

    # === Batch inference over GLB sequences ===
    glb_seqs = find_all_valid_glb_sequences(cfg["data"]["glb_root_dir"])
    glb_seqs.sort()
    logger.info(f"Found {len(glb_seqs)} valid GLB sequences")

    for seq_name, glb_path in glb_seqs:
        image_folder = find_image_folder(seq_name, cfg["data"]["image_roots"])
        if image_folder is None:
            logger.warning(f"[SKIP] No image folder found for {seq_name} {glb_path}")
            continue

        if cfg["data"]["retarget"]["toggle"]:
            ref_seq = cfg["data"]["retarget"]["ref_seq"]
            cfg["data"]["wild_flag"] = True # Change to wild, expect no ground truth.
        else:
            
            npz_path, is_wild = resolve_npz_info(seq_name, cfg["data"]["base_dir"])
            if npz_path is None:
                logger.warning(f"[SKIP] No NPZ found for {seq_name}, and retargetting is disabled.")
                continue            
            ref_seq = cfg["data"]["retarget"]["ref_seq"] = os.path.splitext(os.path.relpath(npz_path, f"{cfg['data']['base_dir']}/npz_train"))[0]
            logger.info(f"[PAIR] {seq_name} | Wild={is_wild}")
        logger.info(f"  - image_folder: {image_folder}")
        logger.info(f"  - glb_folder:   {glb_path}")
        logger.info(f"  - seq_name:   {seq_name}\n")
        logger.info(f"  - ref_seq:   {ref_seq}\n")
        # args.wild_flag = is_wild

        inference(cfg, device, attention_design, model, pipe, rmbg_net, seq_name, glb_path, image_folder)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Inference script for Mesh2Pose")
    parser.add_argument("--config", type=str, default="configs/inference/inference_mesh2pose.yaml", help="Path to the YAML config file")
    args = parser.parse_args()
    
    cfg = load_yaml_config(args.config)
    mesh2pose(cfg)
    
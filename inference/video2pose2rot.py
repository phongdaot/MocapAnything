from math import e
import os
import shutil
import subprocess
import argparse
import pickle
import numpy as np
import torch

from utils.logger import logger
from utils.common import extract_and_compare_image_features_with_rmbg, find_all_valid_image_sequences, get_image_seq_relpath, resolve_npz_info, set_seed
from utils.visualization import plot_pose_compare_from_npy
from utils.npy2bvh import convert_npy_to_bvh
from utils.config_utils import load_yaml_config, instantiate_from_config
from preprocess.briarmbg import BriaRMBG
from TripoSG.triposg.pipelines.pipeline_triposg import TripoSGPipeline
from utils.bvh_reader import BVHReader
from utils.mesh import blender_visualize_character_motion

MAX_JOINTS=150

bvh_reader = BVHReader(
    max_num_joints=MAX_JOINTS,
    crop_size=600,
    no_pos=True,
    bvh_norm=False,
    reset_pose_prob=0.0,
)


def get_video_frame_count(video_path):
    """Use ffprobe to get frame count of a video. Returns -1 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=60,
        )
        return int(result.stdout.strip())
    except Exception as e:
        logger.warning(f"[ffprobe] Failed to read frame count for {video_path}: {e}")
        return -1


def check_and_clean_output_dir(save_dir, expected_frames):
    """
    Walk save_dir for all .mp4 files. If any has frame count != expected_frames,
    delete the ENTIRE save_dir (to avoid leftover mesh files causing GPU OOM)
    and return True. Otherwise return False.
    """
    if not os.path.isdir(save_dir):
        return False

    for root, dirs, files in os.walk(save_dir):
        for fname in files:
            if not fname.endswith(".mp4"):
                continue
            mp4_path = os.path.join(root, fname)
            fc = get_video_frame_count(mp4_path)
            if fc < 0:
                logger.warning(f"[CHECK] Cannot read {mp4_path}, deleting {save_dir} to be safe.")
                shutil.rmtree(save_dir)
                return True
            if fc != expected_frames:
                logger.warning(
                    f"[CHECK] Frame mismatch: {mp4_path} has {fc} frames, expected {expected_frames}. "
                    f"Deleting {save_dir} to regenerate."
                )
                shutil.rmtree(save_dir)
                return True

    return False


def build_inference_batch(
    cfg,
    image_embed_np,   # [W, D]
):
    """
    Build a batch matching collate_anyspecies_padded(batch) output for B=1,
    aligned with AnySpeciesPoseDataset + collate_anyspecies_padded.
    """

    data_cfg = cfg["data"]
    base_dir = data_cfg["base_dir"]
    memory_pkl_path = data_cfg["memory_pkl_path"]

    ref_seq = cfg["data"]["retarget"]["ref_seq"]
    ref_idx = cfg["data"]["retarget"]["ref_idx"]

    device = torch.device(
        cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )

    pose_path = os.path.join(base_dir, "bvh_pose", f"{ref_seq}.npz")
    train_path = os.path.join(base_dir, "npz_train_image_only", f"{ref_seq}.npz")
    species_info_path = os.path.join(base_dir, "species_info_dict.npy")
    bvh_path = os.path.join(base_dir, "bvh", f"{ref_seq}.bvh")

    # same cache path logic as dataset loader
    scale_dict_path = os.path.join(
        base_dir, "cache", "__mesh2pose1002_species_scale_cache.pkl"
    )

    pose_npz = np.load(pose_path, allow_pickle=False)
    train_npz = np.load(train_path, allow_pickle=False)
    species_info_dict = np.load(species_info_path, allow_pickle=True).item()

    with open(memory_pkl_path, "rb") as f:
        species_memory_dict = pickle.load(f)

    with open(scale_dict_path, "rb") as f:
        scale_dict = pickle.load(f)

    species_name = ref_seq.split("/")[0].split("#")[0]

    if species_name not in species_info_dict:
        raise KeyError(f"{species_name} missing in species_info_dict")

    if species_name not in scale_dict:
        raise KeyError(f"{species_name} missing in scale_dict: {scale_dict_path}")

    info = species_info_dict[species_name]
    gscale = np.float32(scale_dict[species_name]["global_scale"])

    # --------------------------------------------------
    # sequence data: follow dataset preload/_load_single_item_from_disk
    # --------------------------------------------------
    position = pose_npz["position"].astype(np.float32)   # [F, J, 3]
    rot6d = pose_npz["rot6d"].astype(np.float32)         # [F, J, 6]
    ref_image_embed_all = train_npz["image_embed"].astype(np.float32)

    # dataset trims pose / rot / image to same min length first
    F_pose = position.shape[0]
    F_img = ref_image_embed_all.shape[0]
    F_data = min(F_pose, F_img)

    position = position[:F_data]
    rot6d = rot6d[:F_data]
    ref_image_embed_all = ref_image_embed_all[:F_data]

    # normalize exactly like dataset
    position = (position - position[:, 0:1, :]) / gscale

    # --------------------------------------------------
    # BVH static info: same style as dataset
    # --------------------------------------------------
    res = {"motion_path": bvh_path}
    res["joint_rename"] = False
    res = bvh_reader(res)

    parents = np.array(res["parents"], dtype=np.int64)          # [J]
    rest_pose = np.array(res["rest_pose"], dtype=np.float32)    # [J, 3]
    rest_pose = (rest_pose - rest_pose[0:1, :]) / gscale

    J = position.shape[1]
    J_max = MAX_JOINTS

    if not cfg["data"]["wild_flag"]:
        # --------------------------------------------------
        # temporal align for inference
        # mimic dataset output W = image_embed length used by model,
        # while ensuring all modalities have same temporal length
        # --------------------------------------------------
        W_infer = image_embed_np.shape[0]
        F_final = min(W_infer, position.shape[0], rot6d.shape[0])

        if image_embed_np.shape[0] != F_final:
            logger.info(f"[Truncate] input image_seq {image_embed_np.shape[0]} -> {F_final}")
        if position.shape[0] != F_final:
            logger.info(f"[Truncate] position_seq {position.shape[0]} -> {F_final}")
        if rot6d.shape[0] != F_final:
            logger.info(f"[Truncate] rot6d_seq {rot6d.shape[0]} -> {F_final}")

        image_embed_np = image_embed_np[:F_final].astype(np.float32)
        position = position[:F_final]
        rot6d = rot6d[:F_final]
        cfg["model"]["attention_kwargs"]["seq_len"] = F_final
    else:
        cfg["model"]["attention_kwargs"]["seq_len"] = image_embed_np.shape[0]

    # --------------------------------------------------
    # Cap seq_len to MAX_SEQ_LEN and truncate data to match
    # --------------------------------------------------
    MAX_SEQ_LEN = 301
    seq_len = cfg["model"]["attention_kwargs"]["seq_len"]
    if seq_len > MAX_SEQ_LEN:
        logger.info(f"[Cap] seq_len {seq_len} -> {MAX_SEQ_LEN}")
        cfg["model"]["attention_kwargs"]["seq_len"] = MAX_SEQ_LEN
        image_embed_np = image_embed_np[:MAX_SEQ_LEN]
        position = position[:MAX_SEQ_LEN]
        rot6d = rot6d[:MAX_SEQ_LEN]

    ref_idx = min(ref_idx, 0)

    # dataset-like fields
    pos_win = position                          # [W, J, 3]
    rot_win_a = rot6d                           # [W, J, 6]
    img_win = image_embed_np                    # [W, D]

    ref_pos = position[ref_idx]                 # [J, 3]
    ref_rot_a = rot6d[ref_idx]                  # [J, 6]
    ref_img = ref_image_embed_all[ref_idx]      # [D]

    # --------------------------------------------------
    # species static cache fields: match dataset
    # --------------------------------------------------
    graph_hop = info["joints_distance"].astype(np.int64)
    graph_edge = info["joint_relation"].astype(np.int64)
    joint_t5embed = info["t5_embedding"].astype(np.float32)

    static_rot_joint_mask = np.zeros((J,), dtype=np.bool_)
    static_rot_joint_mask[info["static_rot_joints"]] = True

    static_pos_joint_mask = np.zeros((J,), dtype=np.bool_)
    static_pos_joint_mask[info["static_joints"]] = True

    # same fallback behavior as __getitem__
    memory_pose = None
    memory_rot6d = None
    if species_name in species_memory_dict:
        sp_mem = species_memory_dict[species_name]
        if ("pose_normed" in sp_mem) and ("rot6d" in sp_mem):
            memory_pose = sp_mem["pose_normed"].astype(np.float32)
            memory_rot6d = sp_mem["rot6d"].astype(np.float32)
            if memory_pose.shape[0] == 0 or memory_rot6d.shape[0] == 0:
                memory_pose = None
                memory_rot6d = None

    if (memory_pose is None) or (memory_rot6d is None):
        memory_pose = ref_pos[None, ...].astype(np.float32)      # [1, J, 3]
        memory_rot6d = ref_rot_a[None, ...].astype(np.float32)   # [1, J, 6]

    # --------------------------------------------------
    # collate_anyspecies_padded style padding
    # --------------------------------------------------
    def pad_joint_2d(x, c, fill=0):
        # [J, C] -> [J_max, C]
        out = np.full((J_max, c), fill, dtype=x.dtype)
        out[:min(x.shape[0], J_max)] = x[:J_max]
        return out

    def pad_joint_3d(x, c, fill=0):
        # [T, J, C] -> [T, J_max, C]
        T = x.shape[0]
        out = np.full((T, J_max, c), fill, dtype=x.dtype)
        out[:, :min(x.shape[1], J_max)] = x[:, :J_max]
        return out

    def pad_mem_3d(x, c, fill=0):
        # [N, J, C] -> [N, J_max, C]
        N = x.shape[0]
        out = np.full((N, J_max, c), fill, dtype=x.dtype)
        out[:, :min(x.shape[1], J_max)] = x[:, :J_max]
        return out

    def pad_vec(x, fill=0):
        # [J] -> [J_max]
        out = np.full((J_max,), fill, dtype=x.dtype)
        out[:min(x.shape[0], J_max)] = x[:J_max]
        return out

    joint_mask = np.zeros((J_max,), dtype=np.bool_)
    joint_mask[:min(J, J_max)] = True

    ancestor_mask = np.zeros((J_max, J_max), dtype=np.bool_)
    for i in range(min(J, J_max)):
        ancestor_mask[i, i] = True
        p = parents[i]
        while p != -1:
            ancestor_mask[i, p] = True
            p = parents[p]

    hop_pad = np.full((J_max, J_max), fill_value=5, dtype=np.int64)
    edge_pad = np.full((J_max, J_max), fill_value=4, dtype=np.int64)
    hop_pad[:J, :J] = graph_hop
    edge_pad[:J, :J] = graph_edge

    pos_win = pad_joint_3d(pos_win, 3, fill=0)
    rot_win_a = pad_joint_3d(rot_win_a, 6, fill=0)
    ref_pos = pad_joint_2d(ref_pos, 3, fill=0)
    ref_rot_a = pad_joint_2d(ref_rot_a, 6, fill=0)
    joint_t5embed = pad_joint_2d(joint_t5embed, joint_t5embed.shape[1], fill=0)
    static_rot_joint_mask = pad_vec(static_rot_joint_mask, fill=False)
    static_pos_joint_mask = pad_vec(static_pos_joint_mask, fill=False)
    memory_pose = pad_mem_3d(memory_pose, 3, fill=0)
    memory_rot6d = pad_mem_3d(memory_rot6d, 6, fill=0)
    parent_a = pad_vec(parents, fill=-1)
    offset_a = pad_joint_2d(rest_pose, 3, fill=0)

    pose_npz.close()
    train_npz.close()

    batch = {
        # ===== shared =====
        "position": torch.from_numpy(pos_win[None]).float().to(device),            # [1, W, J_max, 3]
        "ref_position": torch.from_numpy(ref_pos[None]).float().to(device),        # [1, J_max, 3]
        "joint_mask": torch.from_numpy(joint_mask[None]).to(device),               # [1, J_max]
        "ancestor_mask": torch.from_numpy(ancestor_mask[None]).to(device),         # [1, J_max, J_max]
        "J_valid": torch.tensor([J], dtype=torch.int32, device=device),
        "global_scale": torch.tensor([gscale], dtype=torch.float32, device=device),
        "species": [species_name],
        "graph_hop": torch.from_numpy(hop_pad[None]).to(torch.int64).to(device),
        "graph_edge": torch.from_numpy(edge_pad[None]).to(torch.int64).to(device),
        "joint_t5embed": torch.from_numpy(joint_t5embed[None]).float().to(device),
        "static_rot_joint_mask": torch.from_numpy(static_rot_joint_mask[None]).to(device),
        "static_pos_joint_mask": torch.from_numpy(static_pos_joint_mask[None]).to(device),

        # ===== memory =====
        "memory_pose": torch.from_numpy(memory_pose[None]).float().to(device),
        "memory_rot6d": torch.from_numpy(memory_rot6d[None]).float().to(device),

        # ===== view a =====
        "rot6d_a": torch.from_numpy(rot_win_a[None]).float().to(device),
        "ref_rot6d_a": torch.from_numpy(ref_rot_a[None]).float().to(device),
        "parent_a": torch.from_numpy(parent_a[None]).to(torch.int64).to(device),
        "offset_a": torch.from_numpy(offset_a[None]).float().to(device),

        # ===== video2pose =====
        "image_embed": torch.from_numpy(img_win[None]).float().to(device),
        "ref_image_embed": torch.from_numpy(ref_img[None]).float().to(device),
    }

    return batch
def inference(cfg, device, attention_design, model, pipe, rmbg_net, seq_name, image_folder):
    """
    Run inference on a single sequence using YAML config only.
    """
   
    batch = build_inference_batch(
        cfg=cfg,
        image_embed_np=extract_and_compare_image_features_with_rmbg(
            image_folder=image_folder, rmbg_net=rmbg_net, pipe=pipe
        ),
    )
    
    wild_flag = cfg["data"]["wild_flag"]
    bvh_roots = cfg["data"]["bvh_roots"]
    species_name = batch["species"][0]
    save_dir_root = cfg["output"]["save_dir"]
    expected_seq_len = cfg["model"]["attention_kwargs"]["seq_len"]

    # === Compute save_dir early for validation ===
    save_subfolder = get_image_seq_relpath(image_folder, cfg["data"]["image_roots"])
    test_name = image_folder.split("/")[-2]
    save_dir = os.path.join(save_dir_root, cfg["experiment"]["exp"], test_name, save_subfolder)

    # === Check existing mp4 frame counts; delete whole dir if mismatch ===
    deleted = check_and_clean_output_dir(save_dir, expected_seq_len)
    if deleted:
        logger.info(f"[RERUN] Stale outputs deleted for {seq_name}, regenerating.")

    # === If all outputs already valid, skip entirely ===
    all_videos_ok = True
    for subdir_name in ["camera", "front", "side"]:
        subdir_path = os.path.join(save_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            all_videos_ok = False
            break
        if not any(f.endswith(".mp4") for f in os.listdir(subdir_path)):
            all_videos_ok = False
            break

    if all_videos_ok and os.path.isdir(save_dir):
        npy_preds = [f for f in os.listdir(save_dir) if f.endswith("_pred.npy")]
        if len(npy_preds) >= 2:
            logger.info(f"[SKIP] All outputs valid for {seq_name}, skipping.")
            return

    os.makedirs(save_dir, exist_ok=True)

    # === Inference ===
    with torch.no_grad():
        model_out = model(batch, attention_kwargs=attention_design)
        pred_pos = model_out["pred_position"].squeeze(0).detach().cpu().numpy()
        gt_pos = batch["position"].squeeze(0).detach().cpu().numpy()

        pred_rot = model_out["pred_rot6d"].squeeze(0).detach().cpu().numpy()
        gt_rot = batch["rot6d_a"].squeeze(0).detach().cpu().numpy()
        
    logger.info(f"[Inference] Completed inference for {seq_name}")

    if wild_flag:
        gt_pos = pred_pos
        gt_rot = pred_rot

    # === Save results ===
    species_actual = seq_name.split("#")[0]
    camera_azim = 90
    npy_pose_pred_path = os.path.join(save_dir, f"{species_name}_{species_actual}_pose_pred.npy")
    npy_pose_gt_path = None if wild_flag else os.path.join(save_dir, f"{species_name}_pose_gt.npy")
    npy_rot_pred_path = os.path.join(save_dir, f"{species_name}_{species_actual}_rot6d_pred.npy")
    npy_rot_gt_path = None if wild_flag else os.path.join(save_dir, f"{species_name}_rot6d_gt.npy")

    np.save(npy_pose_pred_path, pred_pos)
    np.save(npy_rot_pred_path, pred_rot)
    if not wild_flag:
        np.save(npy_pose_gt_path, gt_pos)
        np.save(npy_rot_gt_path, gt_rot)

    logger.info(f"[Complete] Saved pose predictions to {save_dir}")

    plot_pose_compare_from_npy(
        pred_npy_path=npy_pose_pred_path,
        gt_npy_path=npy_pose_gt_path,
        species_name=species_name,
        species_actual=species_actual,
        save_dir=save_dir,
        image_folder=image_folder,
        fps=cfg["output"].get("fps", 15),
        wild_flag=wild_flag,
        bvh_roots=bvh_roots,
        camera_azim=camera_azim,
    )
    
    convert_npy_to_bvh(npy_path=npy_rot_pred_path, character_base_dir=cfg["data"]["character_dir"], species_name=species_name)
    if not wild_flag:
        convert_npy_to_bvh(npy_path=npy_rot_gt_path, character_base_dir=cfg["data"]["character_dir"], species_name=species_name)
    
    # plot_bvh_compare(
    #     pred_bvh_path=npy_rot_pred_path.replace(".npy", ".bvh"),
    #     gt_bvh_path=npy_rot_gt_path.replace(".npy", ".bvh") if not wild_flag else None,
    #     species_name=species_name,
    #     species_actual=species_actual,
    #     save_dir=save_dir,
    #     image_folder=image_folder,
    #     fps=cfg["output"].get("fps", 15),
    #     wild_flag=wild_flag,
    # )
    
    # Extract bvh to mesh
    character_base_dir = os.path.join(cfg["data"]["character_dir"], f"{species_name}")
    
    # Generate video from mesh
    if cfg["output"].get("blender_path", None) is not None and os.path.exists(cfg["output"]["blender_path"]):
        
        for azim in [(0, "camera"), (30, "front"), (-60, "side")]: # Need change

            output_dir = os.path.join(save_dir, azim[1])
            video_exists = any(
                f.endswith(".mp4")
                for f in os.listdir(output_dir)
            ) if os.path.isdir(output_dir) else False

            if video_exists:
                logger.info(f"[SKIP] Video already exists in {output_dir}, skipping Blender export.")
            else:
                blender_visualize_character_motion(
                    blender_path=cfg["output"]["blender_path"],
                    output_dir=output_dir,
                    scene="blank",
                    motion_path=npy_rot_pred_path.replace(".npy", ".bvh"),
                    character_folder=character_base_dir,
                    view_scale=1.5,  # 1.8
                    object_position=0.4,  # 0.05
                    camera_trace=True,
                    traj_smooth=0.0,
                    fps=cfg["output"].get("fps", 15),
                    auto_scale="bvh",
                    bg_color=(255, 255, 255),
                    azim=azim[0],
                )
        
            # if cfg["output"].get("export_gt_video", True) and not wild_flag:
            #     blender_visualize_character_motion(
            #         blender_path=cfg["output"]["blender_path"],
            #         output_dir=save_dir,
            #         scene="blank",
            #         motion_path=npy_rot_gt_path.replace(".npy", ".bvh"),
            #         character_folder=character_base_dir,
            #         view_scale=1.8,
            #         object_position=0.5,
            #         camera_trace=True,
            #         traj_smooth=0.0,
            #         fps=cfg["output"].get("fps", 15),
            #         auto_scale="bvh",
            #         bg_color=(255, 255, 255),
            #         azim=zim,
            #     )
    else:
        logger.warning(f"Blender path {cfg['output']['blender_path']} does not exist. Skipping video export.")
    
    mpjpe_pos = np.mean(np.linalg.norm(pred_pos - gt_pos, axis=-1))
    logger.info(f"[METRIC] MPJPE (mm): {mpjpe_pos:.6f}")

    mpjpe_rot = np.mean(np.linalg.norm(pred_rot - gt_rot, axis=-1))
    logger.info(f"[METRIC] MPJPE (mm): {mpjpe_rot:.6f}")


def video2pose2rot(cfg):
    set_seed(cfg["runtime"]["seed"])

    device_str = cfg["runtime"]["device"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === Build preprocessing models ===
    rmbg_net = BriaRMBG.from_pretrained(
        cfg["weights"]["rmbg_weights_dir"]
    ).to(device).eval()

    pipe = TripoSGPipeline.from_pretrained(
        cfg["weights"]["triposg_weights_dir"]
    ).to(device, torch.float16)

    # === Build prediction model ===
    model: torch.nn.Module = instantiate_from_config(cfg["model"])
    model = model.float().to(device).eval()

    ckpt_path = os.path.join(
        cfg["weights"]["video2pose_ckpt_root"],
        cfg["experiment"]["exp"],
        cfg["weights"].get("ckpt_name", "video2pose2rot_ckpt_best.pt"),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    attention_design = cfg["model"]["attention_kwargs"]

    # === Batch inference over IMAGE sequences ===
    image_seqs = find_all_valid_image_sequences(cfg["data"]["image_roots"])
    # image_seqs.sort()
    logger.info(f"Found {len(image_seqs)} valid image sequences")

    for seq_name, image_folder in image_seqs:
        
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
        logger.info(f"  - seq_name:   {seq_name}\n")
        logger.info(f"  - ref_seq:   {ref_seq}\n")

        inference(
            cfg=cfg,
            device=device,
            attention_design=attention_design,
            model=model,
            pipe=pipe,
            rmbg_net=rmbg_net,
            seq_name=seq_name,
            image_folder=image_folder,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Video2Pose")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference/inference_video2pose2rot.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    video2pose2rot(cfg)
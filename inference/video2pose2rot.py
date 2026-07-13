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
try:
    from TripoSG.triposg.pipelines.pipeline_triposg import TripoSGPipeline  # 沙盒不用,可选
except Exception:
    TripoSGPipeline = None
from utils.bvh_reader import BVHReader
from utils.mesh import blender_visualize_character_motion, extract_mesh_from_bvh

# ===== 沙盒改造:独立 DINOv2(绕开缺失的 TripoSG_temporal)+ 直接读 mp4(内部透明抽帧)=====
import glob, tempfile, re
from transformers import AutoImageProcessor, Dinov2Model
# 沙盒:环境无 ffmpeg CLI,用 imageio-ffmpeg 自带二进制喂给 matplotlib(pose 对比 mp4 用它)
try:
    import matplotlib, imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

def derive_ref_seq(seq_name, base_dir):
    """自驱动:输入 seq_name(如 Hamster#Hamster-RollAttack_y60)→ ref_seq(Hamster#Hamster-RollAttack/y60)。
    绕开 resolve_npz_info(它找的是不存在的 npz_train/);按 bvh_pose 存在性校验,优先输入视角、回退 y0。"""
    m = re.match(r"^(.*)_(y\d+)$", seq_name)
    stem, view = (m.group(1), m.group(2)) if m else (seq_name, "y0")
    for v in [view, "y0"]:
        ref = f"{stem}/{v}"
        if os.path.exists(os.path.join(base_dir, "bvh_pose", ref + ".npz")):
            return ref
    return None

def derive_wild_ref(seq_name, base_dir):
    """野外/跨骨架:输入 seq_name(如 Lion#Lion_Act1)→ 取该物种在 base_dir 里任一序列做参考骨架(y0)。
    无 GT(wild_flag=True)。物种名对不上则返回 None 跳过。"""
    species = seq_name.split("#")[0]
    pose_root = os.path.join(base_dir, "bvh_pose")
    if not os.path.isdir(pose_root):
        return None
    for d in sorted(os.listdir(pose_root)):
        if d.split("#")[0] == species:
            for v in ["y0", "y90", "y30"]:
                ref = f"{d}/{v}"
                if os.path.exists(os.path.join(pose_root, ref + ".npz")):
                    return ref
    return None

class DinoPipe:
    """只提供 v2p2r 推理需要的 DINOv2 接口(feature_extractor_dinov2 / image_encoder_dinov2),
    从 facebook/dinov2-large 本地缓存加载,避免依赖缺失的 TripoSG 几何权重。"""
    def __init__(self, device, dtype=torch.float16, model_id="facebook/dinov2-large"):
        self.feature_extractor_dinov2 = AutoImageProcessor.from_pretrained(model_id)
        self.image_encoder_dinov2 = Dinov2Model.from_pretrained(model_id).to(device, dtype).eval()

def video_to_frames(video_path, out_dir):
    """用 cv2 把 mp4 抽帧到 out_dir(环境无 ffmpeg CLI),返回帧数。"""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.png"), frame)
        i += 1
    cap.release()
    return i

def find_all_videos(video_roots):
    """返回 [(seq_name, mp4_path)],seq_name = 去扩展名的文件名。"""
    out = []
    for root in video_roots:
        if not os.path.isdir(root):
            continue
        for f in sorted(os.listdir(root)):
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                out.append((os.path.splitext(f)[0], os.path.join(root, f)))
    return out

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

    # 沙盒:noT5 无 memory bank 训练,memory_pkl 缺失时用空 dict → 自动回退到 ref 帧(见下方 memory fallback)
    if memory_pkl_path and os.path.exists(memory_pkl_path):
        with open(memory_pkl_path, "rb") as f:
            species_memory_dict = pickle.load(f)
    else:
        species_memory_dict = {}

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

    # 解耦:推理时 ref_image_embed_all 只取第 0 帧(ref_idx=0)当参考图,
    # 它的长度不该限制 pose/输入视频的长度。只有复现 eval(use_stored,把存好的、
    # 与 pose 逐帧对齐的特征当输入)时才需要 pose 与 image 逐帧对齐 → 取 min。
    # 这样 npz_train_image_only 可只存 1 帧(demo mini-dataset),自驱动输入仍走全长。
    if cfg.get("use_stored_image_embed", False):
        F_data = min(F_pose, F_img)
        position = position[:F_data]
        rot6d = rot6d[:F_data]
        ref_image_embed_all = ref_image_embed_all[:F_data]
        image_embed_np = ref_image_embed_all.copy()
    else:
        position = position[:F_pose]
        rot6d = rot6d[:F_pose]
        # ref_image_embed_all 保持原样,后面只用 ref_image_embed_all[ref_idx=0]

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

    # 沙盒:对齐 eval 的窗口长度(eval 用 seq_len 帧窗口;设 data.eval_seq_len=32 截断成同样窗口)
    # eval_win_mode: "first"(默认,前 _cap 帧)| "middle"(序列中段窗口,验证冷启动假设)
    _cap = cfg["data"].get("eval_seq_len")
    if _cap and cfg["model"]["attention_kwargs"]["seq_len"] > _cap:
        cfg["model"]["attention_kwargs"]["seq_len"] = _cap
        _wmode = cfg["data"].get("eval_win_mode", "first")
        _F = image_embed_np.shape[0]
        if _wmode == "middle":
            _s = max(0, (_F - _cap) // 2)
        else:
            _s = 0
        logger.info(f"[Window] mode={_wmode} start={_s} cap={_cap} (F={_F})")
        image_embed_np = image_embed_np[_s:_s + _cap]
        position = position[_s:_s + _cap]
        rot6d = rot6d[_s:_s + _cap]

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
def inference(cfg, device, attention_design, model, pipe, rmbg_net, seq_name, image_folder, stage_cb=None):
    """
    Run inference on a single sequence using YAML config only.
    stage_cb(name): 可选阶段回调,name ∈ {"dino","v2p","p2r","render","export"},供 Web 端实时显示进度。
    """
    if stage_cb is not None:
        stage_cb("dino")
    batch = build_inference_batch(
        cfg=cfg,
        image_embed_np=extract_and_compare_image_features_with_rmbg(
            image_folder=image_folder, rmbg_net=rmbg_net, pipe=pipe
        ),
    )
    # 沙盒:noT5 ckpt 训练时 joint_t5embed 被置零(--ablate_no_t5),推理必须同样置零才匹配
    if cfg.get("ablate_no_t5", False):
        batch["joint_t5embed"] = torch.zeros_like(batch["joint_t5embed"])

    wild_flag = cfg["data"]["wild_flag"]
    bvh_roots = cfg["data"]["bvh_roots"]
    species_name = batch["species"][0]
    save_dir_root = cfg["output"]["save_dir"]
    expected_seq_len = cfg["model"]["attention_kwargs"]["seq_len"]

    # === Compute save_dir early for validation ===
    save_subfolder = get_image_seq_relpath(image_folder, cfg["data"]["image_roots"])
    test_name = image_folder.split("/")[-2]
    # 沙盒:输出结构 = infer_outputs/{output_tag=方法_ckptstep}/{nbg_split}/{sample}/(与 ckpt 路径的 exp 解耦)
    _otag = cfg["output"].get("output_tag") or cfg["experiment"]["exp"]
    save_dir = os.path.join(save_dir_root, _otag, test_name, save_subfolder)

    # === Check existing mp4 frame counts; delete whole dir if mismatch ===
    deleted = check_and_clean_output_dir(save_dir, expected_seq_len)
    if deleted:
        logger.info(f"[RERUN] Stale outputs deleted for {seq_name}, regenerating.")

    # === If all outputs already valid, skip entirely ===
    all_videos_ok = True
    for subdir_name in ["camera", "side"]:  # 只渲 cam+side,skip 判据同步(去掉 front)
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
        model_out = model(batch, attention_kwargs=attention_design, stage_cb=stage_cb)
        # 沙盒:用 eval 同一套 evaluate_joint_metrics 算 masked 指标,直接和训练 eval 对齐
        try:
            from train.video2pose2rot import evaluate_joint_metrics as _ejm
            _em = _ejm(model_out, batch)
            logger.info(f"[EVALMETRIC] {seq_name} pose_mpjpe={float(_em['pose_mpjpe']):.6f} "
                        f"pose_mpjve={float(_em['pose_mpjve']):.6f} rot_l1={float(_em['rot_l1']):.6f} "
                        f"rot_l2={float(_em['rot_l2']):.6f} angle_l1={float(_em['angle_l1']):.4f} "
                        f"fk_l1={float(_em['fk_l1']):.6f}")
        except Exception as _e:
            logger.warning(f"[EVALMETRIC fail] {seq_name}: {_e}")
        if cfg.get("metric_only"):
            return   # 对齐测试:只要指标,跳过慢的 npy/plot/bvh
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

    if stage_cb is not None:
        stage_cb("plot")   # 骨架视频绘制(matplotlib,较慢);与真正的 p2r 前向区分开

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
        front_azim=None,   # 只要 camera + side,跳过 front(省时间)
    )

    # 沙盒:主骨架 compare = 输入视频 | 骨架camera | 骨架side 横拼(web demo 布局:input|skeleton;
    # 视角只用 cam+side)。移到父目录集中浏览。
    try:
        import re as _re, shutil as _sh, subprocess as _sp
        import imageio_ffmpeg as _iio
        _ff = _iio.get_ffmpeg_exe()
        _parent = os.path.dirname(save_dir)
        _sample = os.path.basename(save_dir)
        _cam = _side = _image = None
        for _f in os.listdir(save_dir):
            if _re.search(r"_pose_compare_camera\.mp4$", _f):
                _cam = os.path.join(save_dir, _f)
            elif _re.search(r"_pose_compare_side\.mp4$", _f):
                _side = os.path.join(save_dir, _f)
            elif _re.search(r"_pose_compare_image\.mp4$", _f):
                _image = os.path.join(save_dir, _f)
        _dst = os.path.join(_parent, f"{_sample}_pose_compare.mp4")
        # 顺序:输入 | 骨架cam | 骨架side(缺输入则退回 cam|side)
        _cols = [p for p in [_image, _cam, _side] if p]
        if len(_cols) >= 2:
            _cmd = [_ff, "-y"]
            for _p in _cols:
                _cmd += ["-i", _p]
            _cmd += ["-filter_complex", f"hstack=inputs={len(_cols)}", _dst]
            _sp.run(_cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, check=False)
        elif len(_cols) == 1:
            _sh.copy(_cols[0], _dst)
    except Exception as _ce:
        logger.warning(f"[compare hstack skip] {save_dir}: {_ce}")

    # 沙盒:缺 *_ffs.bvh 模板时 BVH 转换会失败;包起来不影响已保存的 npy + pose 对比 mp4
    try:
        convert_npy_to_bvh(npy_path=npy_rot_pred_path, character_base_dir=cfg["data"]["character_dir"], species_name=species_name)
        if not wild_flag:
            convert_npy_to_bvh(npy_path=npy_rot_gt_path, character_base_dir=cfg["data"]["character_dir"], species_name=species_name)
    except Exception as _bvh_e:
        logger.warning(f"[BVH skip] {species_name}: {_bvh_e}")
        # 不 return:继续到下方 MPJPE metric 日志(渲染块因 blender_path=null 自动跳过)
    
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

    # === 交互式 3D:从预测 bvh + 角色 mesh 导出 skeleton + mesh glb(供 Web gr.Model3D 旋转/播放)===
    glb_paths = {}
    if cfg.get("export_glb", False):
        if stage_cb is not None:
            stage_cb("export")
        try:
            # skinned glTF(纯 python,与 LBS 数值一致;morph-target 在 three.js 会爆炸,勿用)
            from utils.glb_export import export_skinned_glb
            _bvh_pred = npy_rot_pred_path.replace(".npy", ".bvh")
            _mesh_glb = os.path.join(save_dir, f"{species_name}_mesh.glb")
            _skel_glb = os.path.join(save_dir, f"{species_name}_skeleton.glb")
            export_skinned_glb(_bvh_pred, character_base_dir, _mesh_glb, _skel_glb,
                               fps=cfg["output"].get("fps", 15), validate=False)
            glb_paths = {"mesh": _mesh_glb, "skeleton": _skel_glb}
            logger.info(f"[glb] exported skinned mesh+skeleton glb → {save_dir}")
        except Exception as _ge:
            import traceback as _tbg
            logger.warning(f"[glb export skip] {species_name}: {_ge}\n{_tbg.format_exc()[-400:]}")

    # Generate video from mesh
    if cfg["output"].get("blender_path", None) is not None and os.path.exists(cfg["output"]["blender_path"]):
        if stage_cb is not None:
            stage_cb("render")
        
        for azim in [(0, "camera"), (-60, "side")]:  # 只渲 camera+side(省 33% 渲染)

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
                    object_position=0.25,  # 0.4偏下→0.25上抬(framing solve_heights居中值~0.35,取其下)
                    camera_trace=True,
                    traj_smooth=0.0,
                    fps=cfg["output"].get("fps", 15),
                    auto_scale="bvh",
                    bg_color=(255, 255, 255),
                    azim=azim[0],
                    render_samples=32,   # 快模式:128->32
                    fast_render=True,    # OptiX 降噪 + GPU-only
                    render_resolution=480,  # 720->480(拼图缩到400高,无损提速~2x)
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

        # 沙盒:mesh 渲完 → 一步到位拼最终布局(web demo):
        #   输入 | 骨架cam | mesh_cam | 骨架side | mesh_side(统一高 400 横拼)
        try:
            import imageio_ffmpeg as _iio2, subprocess as _sp2
            _ff2 = _iio2.get_ffmpeg_exe()
            _parent2 = os.path.dirname(save_dir)
            _sample2 = os.path.basename(save_dir)
            _img = _skc = _sks = None
            for _f in os.listdir(save_dir):
                if _f.endswith("_pose_compare_image.mp4"):
                    _img = os.path.join(save_dir, _f)
                elif _f.endswith("_pose_compare_camera.mp4"):
                    _skc = os.path.join(save_dir, _f)
                elif _f.endswith("_pose_compare_side.mp4"):
                    _sks = os.path.join(save_dir, _f)

            def _first_pred(_sub):
                _p = os.path.join(save_dir, _sub)
                if os.path.isdir(_p):
                    for _f in os.listdir(_p):
                        if _f.endswith("rot6d_pred.mp4"):
                            return os.path.join(_p, _f)
                return None
            _mc = _first_pred("camera")
            _ms = _first_pred("side")
            # 顺序:输入 | 骨架cam | mesh_cam | 骨架side | mesh_side
            _seq = [p for p in [_img, _skc, _mc, _sks, _ms] if p]
            if len(_seq) >= 2:
                _cmd = [_ff2, "-y"]
                for _p in _seq:
                    _cmd += ["-i", _p]
                _fc = ";".join(f"[{_i}:v]scale=-1:400[v{_i}]" for _i in range(len(_seq))) + ";"
                _fc += "".join(f"[v{_i}]" for _i in range(len(_seq))) + f"hstack=inputs={len(_seq)}"
                _finalp = os.path.join(_parent2, f"{_sample2}_final.mp4")
                _cmd += ["-filter_complex", _fc, _finalp]
                _sp2.run(_cmd, stdout=_sp2.DEVNULL, stderr=_sp2.DEVNULL, check=False)
                # 有 _final 后即删中间产物 _pose_compare.mp4(冗余,省空间)
                if os.path.exists(_finalp):
                    _pcp = os.path.join(_parent2, f"{_sample2}_pose_compare.mp4")
                    if os.path.exists(_pcp):
                        try: os.remove(_pcp)
                        except OSError: pass
        except Exception as _fe:
            logger.warning(f"[final compose skip] {save_dir}: {_fe}")
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

    # 沙盒:用独立 DINOv2 替身,绕开缺失的 TripoSG_temporal 几何权重(v2p2r 只需 DINO 特征)
    pipe = DinoPipe(device)

    # === Build prediction model ===
    model = instantiate_from_config(cfg["model"])
    model = model.float().to(device).eval()

    ckpt_path = os.path.join(
        cfg["weights"]["video2pose_ckpt_root"],
        cfg["experiment"]["exp"],
        cfg["weights"].get("ckpt_name", "video2pose2rot_ckpt_best.pt"),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    # 沙盒:兼容不同存 key(release=model_state,lab=model),并剥 module. 前缀
    _sd = None
    for _k in ("model_state", "model", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and _k in ckpt:
            _sd = ckpt[_k]; break
    if _sd is None:
        _sd = ckpt
    _sd = {(k[7:] if k.startswith("module.") else k): v for k, v in _sd.items()}
    model.load_state_dict(_sd)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    attention_design = cfg["model"]["attention_kwargs"]

    # === 沙盒:直接遍历 mp4 视频(内部透明抽帧),逐个 try/except 保证一个失败不断批 ===
    import shutil as _shutil, traceback as _tb
    FRAMES_ROOT = cfg["data"].get("frames_tmp_root", "/tmp/v2p2r_frames")
    video_roots = cfg["data"].get("video_roots") or cfg["data"].get("image_roots", [])
    n_ok = n_fail = 0
    # 多卡:每卡起一个进程,SHARD_ID/SHARD_COUNT 把视频列表分片(全局 index % N == id),
    # 各进程只跑自己那份 → N 卡近 N 倍速(渲染是 CPU 密集,分进程也并行)
    _shard_id = int(os.environ.get("SHARD_ID", "0"))
    _shard_n = max(1, int(os.environ.get("SHARD_COUNT", "1")))
    _gidx = -1
    for vroot in video_roots:
        split_tag = os.path.basename(os.path.normpath(vroot))
        videos = find_all_videos([vroot])
        logger.info(f"[{split_tag}] found {len(videos)} videos (shard {_shard_id}/{_shard_n})")
        split_frames_root = os.path.join(FRAMES_ROOT, split_tag)
        cfg["data"]["image_roots"] = [split_frames_root]   # 使 test_name/relpath 逻辑不变
        for seq_name, mp4_path in videos:
            _gidx += 1
            if _gidx % _shard_n != _shard_id:
                continue   # 不属于本 shard,跳过
            frames_dir = os.path.join(split_frames_root, seq_name)
            try:
                nfr = video_to_frames(mp4_path, frames_dir)
                logger.info(f"[{split_tag}] {seq_name}: {nfr} frames -> 推理")
                if cfg["data"].get("wild_mode"):
                    ref = derive_wild_ref(seq_name, cfg["data"]["base_dir"])
                    if ref is None:
                        logger.warning(f"[SKIP] wild 无匹配物种 for {seq_name}")
                        continue
                    cfg["data"]["retarget"]["ref_seq"] = ref
                    cfg["data"]["wild_flag"] = True
                elif cfg["data"]["retarget"]["toggle"]:
                    cfg["data"]["retarget"]["ref_seq"] = cfg["data"]["retarget"].get("ref_seq_fixed", cfg["data"]["retarget"]["ref_seq"])
                    cfg["data"]["wild_flag"] = True
                else:
                    ref = derive_ref_seq(seq_name, cfg["data"]["base_dir"])
                    if ref is None:
                        logger.warning(f"[SKIP] 无 ref bvh_pose for {seq_name}")
                        continue
                    cfg["data"]["retarget"]["ref_seq"] = ref
                    cfg["data"]["wild_flag"] = False
                inference(cfg=cfg, device=device, attention_design=attention_design,
                          model=model, pipe=pipe, rmbg_net=rmbg_net,
                          seq_name=seq_name, image_folder=frames_dir)
                n_ok += 1
            except Exception as e:
                n_fail += 1
                logger.warning(f"[FAIL] {seq_name}: {e}\n{_tb.format_exc()}")
            finally:
                _shutil.rmtree(frames_dir, ignore_errors=True)
    logger.info(f"[DONE] 成功 {n_ok} / 失败 {n_fail}")

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
### sam_utils.py ###
"""Dance Anything 抠人:SAM2 分割 + 用户勾选图层 + 全片跟踪。
- candidate_layers(frame0): SAM2 自动分割 → 候选蒙版(供用户勾选)
- track_to_rgba(...):用选中蒙版做视频跟踪 → 逐帧 RGBA(原图 RGB + 蒙版 alpha)
   → 直接喂 MocapAnything(load_image 见 4 通道有效 alpha 会跳过 RMBG,用我们的蒙版)
权重走 HF 自动下载(facebook/sam2-hiera-large),终端用户首次运行自动拉;
可用环境变量 SAM2_MODEL 换模型。任何异常 → 返回 None,调用方回退 RMBG。"""
import os
import numpy as np
from PIL import Image

SAM2_MODEL = os.environ.get("SAM2_MODEL", "facebook/sam2-hiera-large")
_DEVICE = os.environ.get("APP_DEVICE", "cuda:0")

_amg = None   # SAM2AutomaticMaskGenerator(image)
_vp = None    # video predictor
_load_err = None


def available():
    """SAM2 是否可用(能导入 + 构建成功)。首次调用会加载模型。"""
    return _ensure() is None


def _ensure():
    """懒加载 SAM2(image AMG + video predictor)。成功返回 None,失败返回错误串。"""
    global _amg, _vp, _load_err
    if _amg is not None and _vp is not None:
        return None
    if _load_err is not None:
        return _load_err
    try:
        import torch
        from sam2.build_sam import build_sam2_hf, build_sam2_video_predictor_hf
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        img_model = build_sam2_hf(SAM2_MODEL, device=_DEVICE)
        _amg = SAM2AutomaticMaskGenerator(
            img_model, points_per_side=24, pred_iou_thresh=0.8,
            stability_score_thresh=0.9, min_mask_region_area=2000)
        _vp = build_sam2_video_predictor_hf(SAM2_MODEL, device=_DEVICE)
        return None
    except Exception as e:
        import traceback
        _load_err = f"{e}"
        traceback.print_exc()
        return _load_err


_PALETTE = [(255, 80, 80), (80, 180, 255), (120, 230, 120), (255, 200, 60),
            (200, 120, 255), (80, 230, 230), (255, 140, 200), (170, 220, 80)]


def candidate_layers(frame_path, out_dir, topk=8):
    """对第 0 帧自动分割,返回按面积排序的候选:(overlay 预览图路径, 蒙版 bool 数组) 列表。
    失败返回 ([], [])(调用方回退 RMBG)。"""
    if _ensure() is not None:
        return [], []
    import torch
    os.makedirs(out_dir, exist_ok=True)
    img = np.array(Image.open(frame_path).convert("RGB"))
    try:
        with torch.inference_mode(), torch.autocast(_DEVICE.split(":")[0], dtype=torch.bfloat16):
            masks = _amg.generate(img)
    except Exception:
        import traceback; traceback.print_exc()
        return [], []
    # 过滤过大(整背景)/过小,按面积降序,取 topk
    H, W = img.shape[:2]
    total = H * W
    masks = [m for m in masks if 0.008 * total < m["area"] < 0.9 * total]
    masks.sort(key=lambda m: (-m["predicted_iou"], -m["area"]))
    masks = masks[:topk]
    previews, segs = [], []
    for i, m in enumerate(masks):
        seg = m["segmentation"].astype(bool)
        color = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)
        ov = img.astype(np.float32).copy()
        ov[seg] = 0.45 * ov[seg] + 0.55 * color
        p = os.path.join(out_dir, f"cand_{i}.png")
        Image.fromarray(ov.astype(np.uint8)).save(p)
        previews.append(p)
        segs.append(seg)
    return previews, segs


def track_to_rgba(sam_frames_dir, seed_mask, orig_frames_dir, out_rgba_dir):
    """用 seed_mask(第0帧)在 sam_frames_dir(jpg 帧)上跟踪全片,
    输出逐帧 RGBA(RGB=orig_frames_dir 原图, A=跟踪蒙版)到 out_rgba_dir。
    返回帧数;失败返回 0(调用方回退 RMBG)。"""
    if _ensure() is not None or seed_mask is None:
        return 0
    import torch
    os.makedirs(out_rgba_dir, exist_ok=True)
    orig = sorted(f for f in os.listdir(orig_frames_dir) if f.lower().endswith((".png", ".jpg")))
    try:
        with torch.inference_mode(), torch.autocast(_DEVICE.split(":")[0], dtype=torch.bfloat16):
            state = _vp.init_state(video_path=sam_frames_dir)
            _vp.reset_state(state)
            _vp.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.as_tensor(seed_mask, dtype=torch.bool))
            frame_masks = {}
            for fidx, obj_ids, mask_logits in _vp.propagate_in_video(state):
                frame_masks[fidx] = (mask_logits[0] > 0.0).squeeze().cpu().numpy()
    except Exception:
        import traceback; traceback.print_exc()
        return 0
    n = 0
    for fidx, fname in enumerate(orig):
        seg = frame_masks.get(fidx)
        if seg is None:
            continue
        rgb = np.array(Image.open(os.path.join(orig_frames_dir, fname)).convert("RGB"))
        if seg.shape != rgb.shape[:2]:
            seg = np.array(Image.fromarray(seg.astype(np.uint8) * 255).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
        alpha = (seg.astype(np.uint8) * 255)
        rgba = np.dstack([rgb, alpha])
        Image.fromarray(rgba, "RGBA").save(os.path.join(out_rgba_dir, f"{fidx:05d}.png"))
        n += 1
    return n

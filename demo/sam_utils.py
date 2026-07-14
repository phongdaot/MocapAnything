### sam_utils.py ###
"""Dance Anything 抠人:SAM2 分割 + 用户勾选图层 + 全片跟踪(transformers 实现)。
- candidate_layers(frame0): 网格点自动分割 → 候选蒙版(供用户勾选)
- track_to_rgba(...):用选中蒙版做视频跟踪 → 逐帧 RGBA(原图 RGB + 蒙版 alpha)
   → 直接喂 MocapAnything(load_image 见 4 通道有效 alpha 会跳过 RMBG,用我们的蒙版)

权重走 HF transformers 自动下载(facebook/sam2.1-hiera-large),首次运行自动拉;
可用环境变量 SAM2_MODEL 换模型。任何异常 → 返回空/0,调用方回退 RMBG。

改用 transformers 的 Sam2/Sam2Video(无需 facebook sam2 git 包,免 CUDA 构建)。"""
import os
import numpy as np
from PIL import Image

# base-plus:质量/体积平衡(~320MB),足够跟踪单个舞者;可用 SAM2_MODEL 换 large。
SAM2_MODEL = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-base-plus")
_DEVICE = os.environ.get("APP_DEVICE", "cuda:0")
_GRID = int(os.environ.get("SAM2_POINTS_PER_SIDE", "16"))   # 自动分割网格密度

_img_model = _img_proc = None      # 图像分割(候选层)
_vid_model = _vid_proc = None      # 视频跟踪
_load_err = None


def available():
    """SAM2 是否可用(权重下载 + 构建成功)。首次调用会在 CPU 上加载。"""
    return _ensure() is None


def preload():
    """在 import 阶段(ZeroGPU 无 CUDA)于 CPU 预载 + 预下载权重,避免 GPU 窗口里现下。"""
    return _ensure()


def _ensure():
    """懒加载 SAM2(image + video)到 CPU。成功返回 None,失败返回错误串。
    ZeroGPU 规则:import 时无 GPU → 一律先加载到 CPU,GPU 函数内再 _to(cuda)。"""
    global _img_model, _img_proc, _vid_model, _vid_proc, _load_err
    if _img_model is not None and _vid_model is not None:
        return None
    if _load_err is not None:
        return _load_err
    try:
        from transformers import (Sam2Model, Sam2Processor,
                                  Sam2VideoModel, Sam2VideoProcessor)
        _img_model = Sam2Model.from_pretrained(SAM2_MODEL).eval()
        _img_proc = Sam2Processor.from_pretrained(SAM2_MODEL)
        _vid_model = Sam2VideoModel.from_pretrained(SAM2_MODEL).eval()
        _vid_proc = Sam2VideoProcessor.from_pretrained(SAM2_MODEL)
        return None
    except Exception as e:
        import traceback
        _load_err = f"{e}"
        traceback.print_exc()
        return _load_err


def _dev():
    import torch
    return _DEVICE if torch.cuda.is_available() else "cpu"


def _to(dev):
    """把两个模型移到目标设备(GPU 函数内调用)。"""
    if _img_model is not None:
        _img_model.to(dev)
    if _vid_model is not None:
        _vid_model.to(dev)


_PALETTE = [(255, 80, 80), (80, 180, 255), (120, 230, 120), (255, 200, 60),
            (200, 120, 255), (80, 230, 230), (255, 140, 200), (170, 220, 80)]


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def candidate_layers(frame_path, out_dir, topk=8):
    """对第 0 帧网格点自动分割,返回按分数排序的候选:
    (overlay 预览图路径, 蒙版 bool 数组) 列表。失败返回 ([], [])(调用方回退 RMBG)。"""
    if _ensure() is not None:
        return [], []
    import torch
    os.makedirs(out_dir, exist_ok=True)
    dev = _dev()
    _to(dev)
    img = np.array(Image.open(frame_path).convert("RGB"))
    H, W = img.shape[:2]
    total = H * W

    # 均匀网格采样点(每点作为一个独立 object,一次前向拿多蒙版)
    ys = np.linspace(0, H - 1, _GRID + 2)[1:-1]
    xs = np.linspace(0, W - 1, _GRID + 2)[1:-1]
    pts = [[float(x), float(y)] for y in ys for x in xs]

    cand = []   # (score, area, seg_bool)
    try:
        CHUNK = 64
        for i in range(0, len(pts), CHUNK):
            sub = pts[i:i + CHUNK]
            input_points = [[[p] for p in sub]]           # [1, n_obj, 1, 2]
            input_labels = [[[1] for _ in sub]]           # [1, n_obj, 1]
            inputs = _img_proc(images=img, input_points=input_points,
                               input_labels=input_labels, return_tensors="pt").to(dev)
            with torch.inference_mode():
                out = _img_model(**inputs, multimask_output=True)
            masks = _img_proc.post_process_masks(
                out.pred_masks, inputs["original_sizes"])[0]   # [n_obj, 3, H, W] bool
            scores = out.iou_scores[0].detach().cpu().numpy()  # [n_obj, 3]
            masks = masks.detach().cpu().numpy().astype(bool)
            for j in range(masks.shape[0]):
                best = int(np.argmax(scores[j]))
                seg = masks[j, best]
                area = int(seg.sum())
                if 0.008 * total < area < 0.9 * total:
                    cand.append((float(scores[j, best]), area, seg))
    except Exception:
        import traceback
        traceback.print_exc()
        return [], []

    # 按 (分数, 面积) 降序,贪心 NMS 去重(IoU>0.7 视为同一物体)
    cand.sort(key=lambda c: (-c[0], -c[1]))
    kept = []
    for sc, ar, seg in cand:
        if all(_iou(seg, k[2]) < 0.7 for k in kept):
            kept.append((sc, ar, seg))
        if len(kept) >= topk:
            break

    previews, segs = [], []
    for i, (sc, ar, seg) in enumerate(kept):
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
    sam_files = sorted(f for f in os.listdir(sam_frames_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
    orig = sorted(f for f in os.listdir(orig_frames_dir)
                  if f.lower().endswith((".png", ".jpg")))
    if not sam_files:
        return 0
    dev = _dev()
    _to(dev)
    frames = [Image.open(os.path.join(sam_frames_dir, f)).convert("RGB") for f in sam_files]

    frame_masks = {}
    try:
        session = _vid_proc.init_video_session(video=frames, inference_device=dev)
        _vid_proc.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=1,
            input_masks=torch.as_tensor(seed_mask, dtype=torch.bool))
        with torch.inference_mode():
            _vid_model(inference_session=session, frame_idx=0)
            for out in _vid_model.propagate_in_video_iterator(session):
                m = _vid_proc.post_process_masks(
                    [out.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                    binarize=True)[0]
                frame_masks[out.frame_idx] = m[0].detach().cpu().numpy().astype(bool).squeeze()
    except Exception:
        import traceback
        traceback.print_exc()
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

#!/usr/bin/env python3
"""MocapAnything V2 — unified demo app (local GPU & HF ZeroGPU, one file).

Runs identically in two environments:
  · local   : `python demo/app.py` — uses local datasets/ + checkpoints/, direct CUDA,
              Blender render preferred when available (BLENDER_BIN / blender on PATH).
  · HF Space: the Space's thin app.py clones this repo and imports this module —
              `spaces.GPU` windows, weights/data auto-download from the Hub,
              character meshes lazy-fetched per species.

Tabs: 🎯 Mocap · Retarget  |  💃 Dance Anything (dance video + music → character dances).
Outputs: interactive 3D skeleton + mesh (glb), synced input|skeleton|mesh share clip
(camera view), npy/BVH/glb download.
"""
import os
import sys
import glob
import time
import shutil
import zipfile

# Silence the harmless asyncio event-loop GC noise on ZeroGPU / gradio SSR
# ("Exception ignored in BaseEventLoop.__del__ … ValueError: Invalid file descriptor: -1").
# It fires during garbage collection of an already-closed loop, after requests finish —
# it does NOT affect functionality; it just spams the logs and looks alarming.
_orig_unraisablehook = sys.unraisablehook


def _quiet_unraisablehook(unraisable):
    exc = unraisable.exc_value
    if isinstance(exc, ValueError) and "Invalid file descriptor" in str(exc):
        return
    _orig_unraisablehook(unraisable)


sys.unraisablehook = _quiet_unraisablehook

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "demo"))
os.chdir(REPO)

import torch
import gradio as gr

try:
    import spaces  # ZeroGPU
    ZERO = True
except ImportError:
    ZERO = False
    class _NS:
        @staticmethod
        def GPU(*a, **k):
            if len(a) == 1 and callable(a[0]):
                return a[0]
            return lambda f: f
    spaces = _NS()

from utils.config_utils import load_yaml_config, instantiate_from_config
from inference.video2pose2rot import DinoPipe, video_to_frames, inference
from utils.glb_export import export_skinned_glb
from preprocess.briarmbg import BriaRMBG
import sam_utils as su      # SAM2 subject isolation (transformers)
import dance_utils as du    # ffmpeg audio/video helpers (imageio-ffmpeg)
try:
    from utils.composite_render import render_composite   # input|skeleton|mesh clip
except Exception as _ce:
    print(f"[app] composite render unavailable: {_ce}")
    render_composite = None

BASE_CONFIG = "demo/configs/demo_zoo.yaml"
OUT = os.path.join(REPO, "demo_outputs/_space" if ZERO else "demo_outputs/_app")
os.makedirs(OUT, exist_ok=True)
DATA_REPO = "kehong/MoCapAnythingV2-data-sample"
WEIGHTS_REPO = "kehong/MoCapAnythingV2-weights"

# Blender is optional; when present the Mocap result additionally gets the
# photorealistic render (the composite clip is produced either way).
_BL_WRAP = os.path.join(REPO, "blender_mocapanything.sh")
BLENDER = (not ZERO and os.path.exists(_BL_WRAP)
           and bool(os.environ.get("BLENDER_BIN") or shutil.which("blender")))
print(f"[app] mode={'zerogpu' if ZERO else 'local'} blender={BLENDER}")


def _dset_dir(name):
    """prefer the full local dataset; fall back to demo/data (HF download target)."""
    full = os.path.join(REPO, "datasets", name)
    return full if os.path.isdir(os.path.join(full, "bvh_pose")) else os.path.join(REPO, "demo/data", name)


# ---- weights + demo data (download only what's missing → no-op on a local box) ----
_cfgP = load_yaml_config(BASE_CONFIG)
_ckpt = os.path.join(_cfgP["weights"]["video2pose_ckpt_root"], _cfgP["experiment"]["exp"],
                     _cfgP["weights"].get("ckpt_name", "video2pos2rot_ckpt_best.pt"))
if not os.path.exists(_ckpt):
    from huggingface_hub import snapshot_download
    snapshot_download(WEIGHTS_REPO, local_dir="checkpoints")
if not os.path.isdir(os.path.join(_dset_dir("zoo1030"), "bvh_pose")):
    from huggingface_hub import snapshot_download
    snapshot_download(DATA_REPO, repo_type="dataset", local_dir="demo/data",
                      ignore_patterns=["*/characters/*", "*/characters_fix_facezplus/*"])


def _ensure_character(base, char_sub, sp):
    """character mesh dir must hold base_mesh + skinning + *_ffs.bvh; on the Space the
    startup snapshot skips meshes → lazy-fetch per species (re-fetch if _ffs missing)."""
    d = os.path.join(base, char_sub, sp)
    ok = (os.path.exists(os.path.join(d, "base_mesh.obj"))
          and glob.glob(os.path.join(d, "*_ffs.bvh")))
    if not ok and base.startswith(os.path.join(REPO, "demo/data")):
        from huggingface_hub import snapshot_download
        name = os.path.basename(base)
        snapshot_download(DATA_REPO, repo_type="dataset", local_dir="demo/data",
                          allow_patterns=[f"{name}/{char_sub}/{sp}/*"])
        ok = os.path.exists(os.path.join(d, "base_mesh.obj"))
    return d if ok else None


def _ensure_ref_bvh(base, ref_seq):
    p = os.path.join(base, "bvh", ref_seq + ".bvh")
    if not os.path.exists(p) and base.startswith(os.path.join(REPO, "demo/data")):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(DATA_REPO, repo_type="dataset", local_dir="demo/data",
                              allow_patterns=[f"{os.path.basename(base)}/bvh/{ref_seq}.bvh"])
        except Exception:
            pass
    return p


# ---- models on CPU (ZeroGPU: no CUDA at import; local: moved to cuda on first run) ----
print("[app] loading models on CPU …")
_cfg0 = load_yaml_config(BASE_CONFIG)
_rmbg = BriaRMBG.from_pretrained(_cfg0["weights"]["rmbg_weights_dir"]).eval()
_pipe = DinoPipe(torch.device("cpu"))
_model = instantiate_from_config(_cfg0["model"]).float().eval()
_ck = torch.load(_ckpt, map_location="cpu")
_sd = next((_ck[k] for k in ("model_state", "model", "model_state_dict", "state_dict")
            if isinstance(_ck, dict) and k in _ck), _ck)
_model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in _sd.items()})
try:
    print("[app] preloading SAM2 (transformers) on CPU …")
    print("[app] SAM2 ready" if su.preload() is None else "[app] SAM2 unavailable → RMBG fallback")
except Exception as _se:
    print(f"[app] SAM2 preload skipped: {_se}")
print("[app] models ready")

_moved = {"done": False}


def _to_cuda():
    dev = torch.device((os.environ.get("APP_DEVICE", "cuda:0") if not ZERO else "cuda")
                       if torch.cuda.is_available() else "cpu")
    if not _moved["done"]:
        _model.to(dev); _rmbg.to(dev)
        _pipe.image_encoder_dinov2.to(dev)
        _moved["done"] = True
    return dev


# ---- target refs ----
# obj:只保留老本地 demo 的精选角色(可运行∩有ref图,去掉1个渲染坏的;用户要求不要太多)
OBJ_KEEP = {
    "0698819d-b40e-555d-9ef4-139bbc839933", "0a06d19c-d326-53a4-9d72-a9600deaa733",
    "0a763d17-417d-5473-a41a-7d1a796544d2", "0b8ed01e-5a6d-503f-bc59-24a2eb4e0951",
    "0df893d8-8774-5cc1-ac74-8111a1c0ae08", "0ff3240c-32e5-5ef3-89ff-e4c30ad80c54",
    "1a963917-a097-5856-b445-024ebabf7a78", "22160d82-24bc-573c-922e-da11786b5ea2",
    "2fcedbc3-d899-516b-b339-4f671945e329", "315eddf6-32b7-53e1-8106-c270c726dca8",
    "530a6b1a-5be2-5d3d-aa68-60ca61cb0974", "5ecda513-bc1b-5d76-849a-beace54e80c5",
    "6babbd9f-307b-5db5-aabc-35196095cbaa", "82d18eaf-00de-5fd8-91d0-cb2152ae4629",
    "97e2fc5c-50e4-5199-8632-99fa7160b126", "99d7a77d-61de-5d65-8764-e15a2480cc82",
    "9a089094-d277-52e2-b00c-4fe2e7c4a717", "9f41c79f-b2d7-5c38-93e2-29cc1470357c",
    "a6d3cfe3-ac09-5a09-b902-0baf55e5533e",
}

DSET = {
    "zoo": {"base": _dset_dir("zoo1030"), "char_sub": "characters_fix_facezplus",
            "label": "🐾 zoo (animal · front)"},
    "zoo_side": {"base": _dset_dir("zoo1030"), "char_sub": "characters_fix_facezplus",
                 "label": "🐾 zoo (animal · side)", "force_view": "y90", "ref_img_sub": "ref_images_y90"},
    "obj": {"base": _dset_dir("obj1k"), "char_sub": "characters", "label": "📦 obj (objects)"},
}


def _species_map(base, force_view=None):
    pose_root, train_root = f"{base}/bvh_pose", f"{base}/npz_train_image_only"
    views = [force_view] if force_view else ("y0", "y90", "y30")
    out = {}
    for d in sorted(os.listdir(pose_root)) if os.path.isdir(pose_root) else []:
        sp = d.split("#")[0]
        if sp in out:
            continue
        for v in views:
            if os.path.exists(f"{pose_root}/{d}/{v}.npz") and os.path.exists(f"{train_root}/{d}/{v}.npz"):
                out[sp] = f"{d}/{v}"
                break
    return out


_THUMB = os.path.join(REPO, "demo/assets/thumbs")


def _ref_thumb(base, sp, sub="ref_images"):
    p = os.path.join(base, sub, f"{sp}.jpg")
    if os.path.exists(p):
        return p
    if sub != "ref_images":
        p = os.path.join(base, "ref_images", f"{sp}.jpg")
        if os.path.exists(p):
            return p
    cand = sorted(glob.glob(os.path.join(_THUMB, f"{sp}_{sp}_*.jpg")))
    if cand:
        return cand[0]
    tex = sorted(glob.glob(os.path.join(base, "characters", sp, "*.png")))
    return tex[0] if tex else None


for d in DSET.values():
    d["sp_map"] = _species_map(d["base"], d.get("force_view"))
    sub = d.get("ref_img_sub", "ref_images")
    keep = OBJ_KEEP if "obj1k" in d["base"] else None
    gal, sps = [], []
    for sp in sorted(d["sp_map"]):
        if keep is not None and sp not in keep:
            continue
        im = _ref_thumb(d["base"], sp, sub)
        if im:
            gal.append((im, sp[:14] if len(sp) > 16 else sp))
            sps.append(sp)
    d["gallery"] = gal
    d["species"] = sps
print("[app] refs: " + " ".join(f"{k}={len(v['gallery'])}" for k, v in DSET.items()))

# ---- sample inputs ----
EXSET = {
    "wild":  {"dir": "demo/data/inputs/nbg_wild",  "label": "🦁 wild (real footage)"},
    "human": {"dir": "demo/data/inputs/nbg_human", "label": "🕺 human (motion)"},
    "dance": {"dir": "demo/data/inputs/dance",     "label": "💃 dance (with music)"},
    "zoo":   {"dir": "demo/data/inputs/nbg_zoo",   "label": "🐾 zoo (synthetic)"},
    "obj":   {"dir": "demo/data/inputs/nbg_obj",   "label": "📦 obj (objects)"},
}
EX_FRONT = {
    "wild": ["Eagle#Eagle_Act2", "Jaguar#Jaguar_Act2", "Chicken#Chicken_Act2",
             "Leapord#Leapord_Act4", "Dog#Dog_Act2"],
    "zoo":  ["Hamster#Hamster-RollAttack_y60", "Jaguar#Jaguar-Run_y60",
             "Eagle#Eagle-Landing_y15", "Dog-2#DOG-SwimIdle_y30", "Horse#HorseALL-FeetUp_y30"],
    "obj":  ["6babbd9f-307b-5db5-aabc-35196095cbaa#6babbd9f-307b-5db5-aabc-35196095cbaa_y15",
             "0a763d17-417d-5473-a41a-7d1a796544d2#0a763d17-417d-5473-a41a-7d1a796544d2_y0",
             "97e2fc5c-50e4-5199-8632-99fa7160b126#97e2fc5c-50e4-5199-8632-99fa7160b126_y30",
             "1a963917-a097-5856-b445-024ebabf7a78#1a963917-a097-5856-b445-024ebabf7a78_y45",
             "22160d82-24bc-573c-922e-da11786b5ea2#22160d82-24bc-573c-922e-da11786b5ea2_y45"],
}


def _ex_items(name):
    vids = sorted(glob.glob(os.path.join(EXSET[name]["dir"], "*.mp4")))
    front = EX_FRONT.get(name, [])
    fpos = {s: i for i, s in enumerate(front)}
    vids.sort(key=lambda p: (fpos.get(os.path.splitext(os.path.basename(p))[0], len(front)), p))
    items = []
    for mp4 in vids:
        stem = os.path.splitext(os.path.basename(mp4))[0]
        th = os.path.join(_THUMB, stem.replace("#", "_") + ".jpg")
        items.append((th if os.path.exists(th) else mp4, stem.split("#")[0][:14]))
    return vids, items


for name in EXSET:
    EXSET[name]["videos"], EXSET[name]["items"] = _ex_items(name)
print("[app] samples: " + " ".join(f"{k}={len(v['videos'])}" for k, v in EXSET.items()))


# ---------- SAM subject isolation (uploads) ----------
def _fg_prior(img_path):
    """RMBG 前景 bool mask,作为 SAM 候选排序的语义先验(主体排最前)。失败返回 None。
    预处理与 preprocess/image_process.load_image 的 RMBG 分支一致。"""
    try:
        import cv2
        import numpy as np
        import torchvision.transforms.functional as TF
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        dev = next(_rmbg.parameters()).device
        t = torch.from_numpy(img).to(dev).float().permute(2, 0, 1) / 255.0
        t1k = torch.nn.functional.interpolate(t.unsqueeze(0), size=(1024, 1024),
                                              mode="bilinear", antialias=True)[0]
        inp = TF.normalize(t1k, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]).unsqueeze(0)
        with torch.no_grad():
            r = _rmbg(inp)[0][0]
        m = torch.nn.functional.interpolate(r, size=(H, W), mode="bilinear")[0, 0]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return (m > 0.5).cpu().numpy()
    except Exception as e:
        print(f"[app] fg prior failed: {e}")
        return None


@spaces.GPU(duration=90)
def segment_upload(video_path):
    if not video_path:
        return None, gr.update(value=[], visible=False), "Upload a video, then pick the subject to track."
    if not su.available():
        return ({"video": video_path, "segs": None}, gr.update(value=[], visible=False),
                "SAM2 unavailable — RMBG will auto-matte on Run.")
    _to_cuda()
    stamp = str(int(time.time() * 1000))[-9:]
    work = os.path.join(OUT, "seg" + stamp)
    orig = os.path.join(work, "orig"); sam = os.path.join(work, "sam")
    os.makedirs(orig, exist_ok=True); os.makedirs(sam, exist_ok=True)
    video_to_frames(video_path, orig)
    from PIL import Image
    for f in sorted(os.listdir(orig)):
        if f.endswith(".png"):
            Image.open(os.path.join(orig, f)).convert("RGB").save(os.path.join(sam, f[:-4] + ".jpg"))
    f0 = sorted(glob.glob(os.path.join(orig, "*.png")))
    if not f0:
        return ({"video": video_path, "segs": None}, gr.update(value=[], visible=False),
                "Could not decode the video.")
    seed_idx = len(f0) // 2      # 中段帧:避开片头标题卡/黑场
    prevs, segs = su.candidate_layers(f0[seed_idx], os.path.join(work, "cand"), topk=8,
                                      prior=_fg_prior(f0[seed_idx]))
    ctx = {"video": video_path, "work": work, "orig": orig, "sam": sam, "segs": segs,
           "seed_idx": seed_idx}
    if prevs:
        return (ctx, gr.update(value=prevs, visible=True),
                f"✅ SAM found **{len(prevs)}** subjects — click the one to track, then Run.")
    return ctx, gr.update(value=[], visible=False), "SAM found no subject — RMBG will auto-matte on Run."


def pick_subject(evt: gr.SelectData):
    return evt.index, f"✅ subject **#{evt.index + 1}** selected — pick a target, then Run."


# ---------- gallery plumbing ----------
GAL_DEFAULT = 30


def on_dset(name):
    d = DSET[name]
    show = [im for im, _ in d["gallery"][:GAL_DEFAULT]]
    more = len(d["gallery"]) > GAL_DEFAULT
    return (gr.update(value=show), None, "No target selected",
            gr.update(visible=more, value=f"▾ Load all {len(d['gallery'])} targets"))


def load_more(name):
    d = DSET[name]
    return gr.update(value=[im for im, _ in d["gallery"]]), gr.update(visible=False)


def on_species(name, evt: gr.SelectData):
    d = DSET[name]
    if evt.index is None or evt.index >= len(d["species"]):
        return None, "No target selected"
    sp = d["species"][evt.index]
    return sp, f"🎯 target: **{sp[:16]}**"


def _safe_copy(src):
    """'#' in a filename breaks gradio's /file= URL (fragment) → copy to a safe name."""
    safe = os.path.join(OUT, "_ex_" + os.path.basename(src).replace("#", "_"))
    if not os.path.exists(safe):
        shutil.copy2(src, safe)
    return safe


def on_exset(name):
    return gr.update(value=EXSET[name]["items"])


def load_example(exname, active, evt: gr.SelectData):
    """route a sample click to the active tab: mocap input, or the dance flow."""
    vids = EXSET[exname]["videos"]
    if evt.index is None or evt.index >= len(vids):
        return gr.update(), gr.update(), gr.update(), "No sample", gr.update()
    safe = _safe_copy(vids[evt.index])
    if active == "dance":
        return gr.update(), gr.update(), gr.update(), "Dance sample loaded — segmenting…", safe
    return safe, None, None, "Example loaded — pick a target, then Run.", gr.update()


# ---------- Mocap / Retarget ----------
@spaces.GPU(duration=120)
def run(video_path, seg_ctx, subject_idx, dset, species, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Pick or upload a video first.")
    d = DSET[dset]
    if not species or species not in d["sp_map"]:
        raise gr.Error("Pick a target species / object from the gallery.")
    dev = _to_cuda()

    stamp = str(int(time.time() * 1000))[-9:]
    seq_name = f"{species}#req{stamp}"
    work = os.path.join(OUT, stamp)
    frames_dir = os.path.join(work, "frames", seq_name)
    os.makedirs(frames_dir, exist_ok=True)

    if seg_ctx and subject_idx is not None and seg_ctx.get("segs") and su.available():
        progress(0.1, desc="SAM tracking subject")
        try:
            n = su.track_to_rgba(seg_ctx["sam"], seg_ctx["segs"][subject_idx], seg_ctx["orig"],
                                 frames_dir, seed_idx=seg_ctx.get("seed_idx", 0))
            if n == 0:
                video_to_frames(video_path, frames_dir)
        except Exception:
            video_to_frames(video_path, frames_dir)
    else:
        progress(0.1, desc="extracting frames")
        video_to_frames(video_path, frames_dir)

    progress(0.15, desc="fetching character")
    base = d["base"]
    if _ensure_character(base, d["char_sub"], species) is None:
        raise gr.Error("Character assets for this target are unavailable — pick another target.")
    if not os.path.exists(_ensure_ref_bvh(base, d["sp_map"][species])):
        raise gr.Error("Reference motion for this target isn't available yet — pick another target.")

    cfg = load_yaml_config(BASE_CONFIG)
    cfg["data"].update(base_dir=base,
                       scale_dict_path=f"{base}/cache/__mesh2pose1002_species_scale_cache.pkl",
                       character_dir=os.path.join(base, d["char_sub"]),
                       bvh_roots=[f"{base}/bvh"],
                       image_roots=[os.path.join(work, "frames")], wild_flag=True)
    cfg["data"]["retarget"].update(ref_seq=d["sp_map"][species], ref_idx=0)
    cfg["output"].update(save_dir=os.path.join(work, "out"), output_tag="app",
                         blender_path=_BL_WRAP if BLENDER else None)
    cfg["export_glb"] = True

    steps = {"dino": 0.25, "v2p": 0.45, "p2r": 0.6, "plot": 0.7, "export": 0.78, "render": 0.82}
    inference(cfg=cfg, device=dev, attention_design=cfg["model"]["attention_kwargs"],
              model=_model, pipe=_pipe, rmbg_net=_rmbg, seq_name=seq_name,
              image_folder=frames_dir,
              stage_cb=lambda nm: progress(steps.get(nm, 0.6), desc=nm))

    out_root = os.path.join(work, "out")
    bvh = next(iter(glob.glob(os.path.join(out_root, "**", "*_rot6d_pred.bvh"), recursive=True)), None)
    if not bvh:
        raise gr.Error("Inference produced no motion — try another video.")
    char = os.path.join(base, d["char_sub"], species)

    progress(0.85, desc="building interactive 3D")
    mesh_glb = next(iter(glob.glob(os.path.join(out_root, "**", "*_mesh.glb"), recursive=True)), None)
    skel_glb = next(iter(glob.glob(os.path.join(out_root, "**", "*_skeleton.glb"), recursive=True)), None)
    if not (mesh_glb and skel_glb):
        mesh_glb = os.path.join(work, f"{species[:16]}_mesh.glb")
        skel_glb = os.path.join(work, f"{species[:16]}_skeleton.glb")
        export_skinned_glb(bvh, char, mesh_glb, skel_glb, fps=15, validate=False)

    # share clip: Blender _final.mp4 when available, else the synced composite
    progress(0.9, desc="rendering shareable clip")
    clip = None
    finals = glob.glob(os.path.join(out_root, "**", "*_final.mp4"), recursive=True)
    if finals:
        clip = os.path.join(work, os.path.basename(finals[0]).replace("#", "_"))
        shutil.copy2(finals[0], clip)
    elif render_composite is not None:
        img_panel = next(iter(glob.glob(os.path.join(out_root, "**", "*_pose_compare_image.mp4"),
                                        recursive=True)), None)
        if img_panel:
            rc = render_composite(bvh, char, img_panel, os.path.join(work, f"composite_{species[:16]}.mp4"), fps=15)
            clip = rc[0] if rc else None
    if clip is None:
        pc = next(iter(glob.glob(os.path.join(out_root, "**", "*_pose_compare.mp4"), recursive=True)), None)
        if pc:
            clip = os.path.join(work, "pose_compare.mp4"); shutil.copy2(pc, clip)

    zpath = os.path.join(work, f"mocapanything_{species[:16]}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in glob.glob(os.path.join(out_root, "**", "*_pred.npy"), recursive=True):
            z.write(f, os.path.basename(f))
        z.write(bvh, os.path.basename(bvh))
        for g in (mesh_glb, skel_glb):
            if g and os.path.exists(g):
                z.write(g, os.path.basename(g))
        if clip:
            z.write(clip, os.path.basename(clip))
    progress(1.0, desc="done")
    return (skel_glb, mesh_glb, gr.update(value=zpath, visible=True),
            gr.update(value=clip, visible=bool(clip)),
            gr.update(value=_share_html(species), visible=True))


# ---------- 💃 Dance Anything ----------
DANCE_FPS = 30


def _dance_segment_impl(video_path, max_sec):
    if not video_path:
        return None, [], "upload / pick a dance sample first"
    _to_cuda()
    stamp = str(int(time.time() * 1000))[-9:]
    work = os.path.join(OUT, "dance" + stamp)
    os.makedirs(work, exist_ok=True)
    std = du.standardize_30fps(video_path, os.path.join(work, "std.mp4"),
                               max_seconds=float(max_sec), fps=DANCE_FPS)
    orig = os.path.join(work, "orig"); du.frames_from_video(std, orig, ext="png")
    sam = os.path.join(work, "sam"); du.frames_from_video(std, sam, ext="jpg")
    audio = du.extract_audio(video_path, os.path.join(work, "audio.m4a"))
    ofrs = sorted(glob.glob(os.path.join(orig, "*.png")))
    seed_idx = len(ofrs) // 2    # 中段帧:避开片头标题卡
    prevs, segs = su.candidate_layers(ofrs[seed_idx], os.path.join(work, "cand"), topk=8,
                                      prior=_fg_prior(ofrs[seed_idx]))
    ctx = {"work": work, "std": std, "orig": orig, "sam": sam, "audio": audio, "segs": segs,
           "seed_idx": seed_idx}
    if prevs:
        return ctx, prevs, f"✅ SAM found **{len(prevs)}** layers — click the person, pick a character, Run."
    return ctx, [], "SAM found no layer — RMBG will auto-matte. Pick a character, then Run."


@spaces.GPU(duration=90)
def dance_segment(video_path, max_sec):
    return _dance_segment_impl(video_path, max_sec)


def dance_pick(evt: gr.SelectData):
    return evt.index, f"✅ person layer **#{evt.index + 1}** — pick a character, then Run."


@spaces.GPU(duration=180)
def run_dance(ctx, pick_idx, dset, species, progress=gr.Progress()):
    if not ctx or not ctx.get("orig"):
        raise gr.Error("Upload / pick a dance video first (it auto-segments).")
    d = DSET[dset]
    if not species or species not in d["sp_map"]:
        raise gr.Error("Pick a target character from the gallery below.")
    dev = _to_cuda()
    work = ctx["work"]
    seq_name = f"{species}#d{os.path.basename(work)}"
    frames_dir = os.path.join(work, "frames", seq_name)
    os.makedirs(frames_dir, exist_ok=True)

    progress(0.05, desc="SAM matting")
    used = "RMBG"
    if pick_idx is not None and ctx.get("segs") and 0 <= pick_idx < len(ctx["segs"]):
        try:
            if su.track_to_rgba(ctx["sam"], ctx["segs"][pick_idx], ctx["orig"], frames_dir,
                                seed_idx=ctx.get("seed_idx", 0)) > 0:
                used = "SAM"
        except Exception:
            pass
    if used != "SAM":
        for f in sorted(os.listdir(ctx["orig"])):
            shutil.copy2(os.path.join(ctx["orig"], f), os.path.join(frames_dir, f))

    progress(0.15, desc="fetching character")
    base = d["base"]
    if _ensure_character(base, d["char_sub"], species) is None:
        raise gr.Error("Character assets for this target are unavailable — pick another target.")
    if not os.path.exists(_ensure_ref_bvh(base, d["sp_map"][species])):
        raise gr.Error("Reference motion for this target isn't available yet — pick another target.")

    cfg = load_yaml_config(BASE_CONFIG)
    cfg["data"].update(base_dir=base,
                       scale_dict_path=f"{base}/cache/__mesh2pose1002_species_scale_cache.pkl",
                       character_dir=os.path.join(base, d["char_sub"]),
                       bvh_roots=[f"{base}/bvh"],
                       image_roots=[os.path.join(work, "frames")], wild_flag=True)
    cfg["data"]["retarget"].update(ref_seq=d["sp_map"][species], ref_idx=0)
    cfg["output"].update(save_dir=os.path.join(work, "out"), output_tag="dance",
                         blender_path=None, fps=DANCE_FPS)
    cfg["export_glb"] = False

    steps = {"dino": 0.3, "v2p": 0.5, "p2r": 0.65, "plot": 0.75}
    inference(cfg=cfg, device=dev, attention_design=cfg["model"]["attention_kwargs"],
              model=_model, pipe=_pipe, rmbg_net=_rmbg, seq_name=seq_name,
              image_folder=frames_dir,
              stage_cb=lambda nm: progress(steps.get(nm, 0.6), desc=nm))

    progress(0.85, desc="rendering character + music")
    bvh = next(iter(glob.glob(os.path.join(work, "out", "**", "*_rot6d_pred.bvh"), recursive=True)), None)
    if not bvh or render_composite is None:
        raise gr.Error("Inference produced no motion — try another video.")
    char = os.path.join(base, d["char_sub"], species)
    rc = render_composite(bvh, char, ctx["std"], os.path.join(work, "dance_comp.mp4"), fps=DANCE_FPS)
    if not rc:
        raise gr.Error("Render failed — please retry.")
    outp = os.path.join(work, f"dance_{species[:16]}.mp4")
    du.mux_audio(rc[0], ctx["audio"], outp)
    progress(1.0, desc="done")
    return outp, gr.update(value=outp, visible=True), gr.update(value=_share_html(species, "dresmp4"), visible=True)


# ---------- share ----------
SPACE_URL = "https://huggingface.co/spaces/kehong/MoCapAnythingV2"


def _share_html(species, vid_elem="resmp4"):
    """mobile: navigator.share attaches the video (IG/小红书/…); desktop: auto-download
    the clip + open the X compose window. gradio 6 strips <script> from gr.HTML → the JS
    lives inline in onclick. JS uses ONLY single quotes; the whole handler is html-escaped
    and wrapped in a double-quoted onclick so inner quotes/& don't break the attribute."""
    import urllib.parse as up
    import html as _html
    text = f"I just animated a {species} from a video with MoCapAnything V2! #MoCapAnythingV2"
    u = up.quote(SPACE_URL); t = up.quote(text)
    x = f"https://twitter.com/intent/tweet?text={t}&url={u}&hashtags=MoCapAnythingV2"
    fb = f"https://www.facebook.com/sharer/sharer.php?u={u}"
    btn = ("display:inline-flex;align-items:center;gap:6px;padding:8px 16px;margin:4px 6px 0 0;border:0;"
           "border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;color:#fff;cursor:pointer;")
    share_js = _html.escape(
        "(async()=>{"
        f"const v=document.querySelector('#{vid_elem} video');"
        f"const msg='{text}';const url='{SPACE_URL}';"
        "try{if(v&&v.src){const r=await fetch(v.src);const b=await r.blob();"
        "const f=new File([b],'mocapanything.mp4',{type:'video/mp4'});"
        "if(navigator.canShare&&navigator.canShare({files:[f]})){"
        "await navigator.share({files:[f],text:msg,url:url});return;}"
        "const a=document.createElement('a');a.href=URL.createObjectURL(b);"
        "a.download='mocapanything.mp4';a.click();"
        "setTimeout(()=>URL.revokeObjectURL(a.href),4000);}}catch(e){}"
        f"window.open('{x}','_blank');" + "})()", quote=True)
    copy_js = _html.escape(
        f"navigator.clipboard.writeText('{SPACE_URL}');this.textContent='✓ Link copied';", quote=True)
    return ('<div style="padding:8px 2px">'
            '<span style="font-weight:600;margin-right:8px">✨ Like it? Share your result:</span><br>'
            f'<button onclick="{share_js}" style="{btn}background:linear-gradient(90deg,#833ab4,#fd1d1d,#fcb045)">'
            '📲 Share result video</button>'
            f'<a href="{x}" target="_blank" style="{btn}background:#000">𝕏 / Twitter</a>'
            f'<a href="{fb}" target="_blank" style="{btn}background:#1877f2">Facebook</a>'
            f'<button onclick="{copy_js}" style="{btn}background:#e75480">📋 Copy link</button>'
            '<div style="font-size:12px;opacity:.72;margin-top:8px;line-height:1.5">'
            '📱 <b>Phone</b>: “Share result video” attaches the clip to Instagram / 小红书 / X directly.<br>'
            '💻 <b>Desktop</b>: social sites can’t auto-attach video, so “Share result video” '
            '<b>auto-downloads the clip</b> and opens the X post box — just <b>drag the downloaded '
            'mocapanything.mp4 into the post</b> (or use ⬇ Download above).</div>'
            '</div>')


# ---------- UI ----------
CSS = ".model3d-h{height:340px!important}"

with gr.Blocks(title="MoCapAnything V2 — End-to-End Motion Capture & Retargeting for Arbitrary Skeletons") as demo:
    gr.Markdown(
        "# 🐾 MoCapAnything V2 — End-to-End Motion Capture & Retargeting for Arbitrary Skeletons\n"
        "Pick a video (or **upload your own** — SAM2 isolates the subject), pick a **target "
        "species/object** (can differ from the input → retargeting), hit **Run** — you get an "
        "**interactive 3D pose skeleton** and **3D mesh** you can rotate, plus `.npy` / BVH.  \n"
        "[Paper](https://huggingface.co/papers/2604.28130) · "
        "[Code](https://github.com/phongdaot/MocapAnything) · "
        "[Weights](https://huggingface.co/kehong/MoCapAnythingV2-weights)"
        + ("" if BLENDER else " — install Blender for the photorealistic render.")
    )
    seg_ctx = gr.State(None)
    subject_idx = gr.State(None)
    species_st = gr.State(None)
    dance_ctx = gr.State(None)
    dance_pick_idx = gr.State(None)
    active_tab = gr.State("mocap")

    with gr.Tabs():
      with gr.Tab("🎯 Mocap · Retarget") as tab_mocap:
        with gr.Row():
            video_in = gr.Video(label="① Input video (upload / drag / pick a sample)", height=340,
                                autoplay=True, loop=True)
            skel_out = gr.Model3D(label="② 3D pose skeleton (drag to rotate)", elem_classes="model3d-h",
                                  clear_color=[1.0, 1.0, 1.0, 1.0])
            mesh_out = gr.Model3D(label="③ 3D mesh result (drag to rotate)", elem_classes="model3d-h",
                                  clear_color=[1.0, 1.0, 1.0, 1.0])
        subj_gallery = gr.Gallery(label="① b — SAM subject layers: click the subject to track (uploads)",
                                  columns=8, height=150, object_fit="contain",
                                  allow_preview=False, visible=False)
        with gr.Row():
            run_btn = gr.Button("🚀 Run", variant="primary", scale=2)
            sel_md = gr.Markdown("No target selected")
            dl_btn = gr.DownloadButton("⬇ Download (npy + BVH + glb)", visible=False, scale=2)
        with gr.Row():
            result_mp4 = gr.Video(label="🎬 Shareable clip — input | skeleton | mesh (synced, camera view)",
                                  elem_id="resmp4", height=240, autoplay=True, loop=True, visible=False, scale=1)
            share_html = gr.HTML(visible=False)

      with gr.Tab("💃 Dance Anything") as tab_dance:
        gr.Markdown("Dance video + music → **an animal / object performs the dance**. "
                    "Upload (or pick a **dance sample below** ⬇) → SAM finds the dancer → click the "
                    "person layer → pick a target character below → **Run dance**.")
        with gr.Row():
            dance_video = gr.Video(label="① Dance video (with audio)", height=300, scale=1)
            dcand_gallery = gr.Gallery(label="② Person layers — click the dancer", columns=2, height=300,
                                       object_fit="contain", allow_preview=False, scale=1)
            dance_out = gr.Video(label="🎬 Result: input | skeleton | character (+music)", height=300,
                                 autoplay=True, loop=True, scale=2, elem_id="dresmp4")
        with gr.Row():
            dmax = gr.Slider(3, 10, value=6, step=1, label="Max seconds", scale=1)
            drun_btn = gr.Button("🚀 Run dance", variant="primary", scale=2)
            dstatus = gr.Markdown("pick a dance sample below ⬇ (or upload) → click person layer → pick character → Run")
            ddl_btn = gr.DownloadButton("⬇ Download", visible=False, scale=1)
        dshare_html = gr.HTML(visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Sample videos** — pick a source, click a thumbnail")
            ex_ds = gr.Radio(choices=[(v["label"], k) for k, v in EXSET.items()],
                             value="wild", label=None)
            ex_gallery = gr.Gallery(value=EXSET["wild"]["items"], columns=3, height=220,
                                    object_fit="cover", show_label=False, allow_preview=False)
        with gr.Column(scale=1):
            dset = gr.Radio(choices=[(v["label"], k) for k, v in DSET.items()],
                            value="zoo", label="Target set")
            species_gallery = gr.Gallery(value=[im for im, _ in DSET["zoo"]["gallery"][:GAL_DEFAULT]], columns=6,
                                         height=380, object_fit="cover", show_label=False, allow_preview=False)
            more_btn = gr.Button(f"▾ Load all {len(DSET['zoo']['gallery'])} targets", size="sm",
                                 visible=len(DSET["zoo"]["gallery"]) > GAL_DEFAULT)

    status = gr.Markdown("Pick or upload a video, then a target species. Uploads run SAM2 to isolate the subject.")

    # tab switch re-targets the shared sample area
    tab_mocap.select(lambda: ("mocap", gr.update(value="wild")), outputs=[active_tab, ex_ds])
    tab_dance.select(lambda: ("dance", gr.update(value="dance")), outputs=[active_tab, ex_ds])

    ex_ds.change(on_exset, inputs=[ex_ds], outputs=[ex_gallery])
    ex_gallery.select(load_example, inputs=[ex_ds, active_tab],
                      outputs=[video_in, seg_ctx, subject_idx, status, dance_video])
    video_in.upload(segment_upload, inputs=[video_in], outputs=[seg_ctx, subj_gallery, status])
    subj_gallery.select(pick_subject, outputs=[subject_idx, status])
    dset.change(on_dset, inputs=[dset], outputs=[species_gallery, species_st, sel_md, more_btn])
    more_btn.click(load_more, inputs=[dset], outputs=[species_gallery, more_btn])
    species_gallery.select(on_species, inputs=[dset], outputs=[species_st, sel_md])
    run_btn.click(run, inputs=[video_in, seg_ctx, subject_idx, dset, species_st],
                  outputs=[skel_out, mesh_out, dl_btn, result_mp4, share_html])

    dance_video.change(dance_segment, inputs=[dance_video, dmax],
                       outputs=[dance_ctx, dcand_gallery, dstatus])
    dcand_gallery.select(dance_pick, outputs=[dance_pick_idx, dstatus])
    drun_btn.click(run_dance, inputs=[dance_ctx, dance_pick_idx, dset, species_st],
                   outputs=[dance_out, ddl_btn, dshare_html])


def main():
    demo.queue(max_size=12).launch(
        server_name=os.environ.get("APP_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("APP_PORT", "7860")) if not ZERO else None,
        allowed_paths=[REPO, OUT, os.path.join(REPO, "demo"),
                       DSET["zoo"]["base"], DSET["obj"]["base"],
                       os.path.realpath(DSET["zoo"]["base"]), os.path.realpath(DSET["obj"]["base"])])


if __name__ == "__main__":
    main()

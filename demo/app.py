#!/usr/bin/env python3
"""
MocapAnything V2 — 交互式 Web Demo (Gradio)。HY-Motion 风格 3D + 科幻终端。

布局:
  第 1 排: [ 输入视频 ]  [ 3D pose 骨架(可旋转) ]  [ 3D mesh 结果(可旋转) ]
  第 2 排: [ 样例视频(hover 预览/点击载入) ]  [ 目标物种画廊(点选) ]  [ 运行终端(流式) ]
顶部 zoo / obj 数据集切换。运行时终端实时打印阶段(DINO → v2p → p2r → 导出 3D)。

启动: PYTHONPATH=$PWD:$PWD/TripoSG python demo/app.py  →  http://localhost:7860
"""
import os, sys, glob, time, threading, html, urllib.parse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
import gradio as gr
from utils.config_utils import load_yaml_config, instantiate_from_config
from inference.video2pose2rot import DinoPipe, video_to_frames, inference
from preprocess.briarmbg import BriaRMBG
sys.path.insert(0, os.path.join(REPO, "demo"))
import dance_utils as du      # noqa: E402  ffmpeg 音视频工具
import sam_utils as su        # noqa: E402  SAM2 抠人

BASE_CONFIG = os.path.join(REPO, "demo/configs/demo_zoo.yaml")
DEVICE = os.environ.get("APP_DEVICE", "cuda:0")
APP_OUT = os.path.join(REPO, "demo_outputs/_app"); os.makedirs(APP_OUT, exist_ok=True)


def _dset_dir(name):
    full = os.path.join(REPO, "datasets", name)
    demo = os.path.join(REPO, "demo/data", name)
    return full if os.path.isdir(os.path.join(full, "bvh_pose")) else demo

# ref 数据集(目标物种/物体):zoo(动物骨架)/ obj(物体骨架)
DSET = {
    # zoo 非人形 mesh 必须用 face-Z+ canonical rest(characters_fix_facezplus),
    # 否则蒙皮时 mesh 朝向与 _ffs 骨架 rest 不匹配 → 渲染爆炸(蜘蛛腿)。obj 无此处理,用 characters。
    "zoo": {"base": _dset_dir("zoo1030"), "char_sub": "characters_fix_facezplus"},
    # zoo 侧面:同一 zoo 数据/角色,ref 强制 y90(侧视参考帧),与正面共享缩略图
    "zoo_side": {"base": _dset_dir("zoo1030"), "char_sub": "characters_fix_facezplus", "force_view": "y90",
                 "ref_img_sub": "ref_images_y90"},
    "obj": {"base": _dset_dir("obj1k"), "char_sub": "characters"},
}
# 样例视频来源(3 类,独立于 ref):wild=真实野外动物 / zoo=渲染合成 / obj=物体
EXSET = {
    "wild":  os.path.join(REPO, "demo/data/inputs/nbg_wild"),
    "human": os.path.join(REPO, "demo/data/inputs/nbg_human"),
    "zoo":   os.path.join(REPO, "demo/data/inputs/nbg_zoo"),
    "obj":   os.path.join(REPO, "demo/data/inputs/nbg_obj"),
    "dance": os.path.join(REPO, "demo/data/inputs/dance"),   # Dance Anything 样例(带音频)
}

# 各来源置顶样例(用户挑选,保持在画廊最前;其余按字母序)
PINNED = {
    "wild": ["Chicken#Chicken_Act2", "Dog#Dog_Act2", "Eagle#Eagle_Act2",
             "Jaguar#Jaguar_Act2", "Leapord#Leapord_Act4"],
}

# ---- 一次性加载模型 ----
print(f"[app] device={DEVICE} 加载模型…")
_cfg0 = load_yaml_config(BASE_CONFIG)
_device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
_rmbg = BriaRMBG.from_pretrained(_cfg0["weights"]["rmbg_weights_dir"]).to(_device).eval()
_pipe = DinoPipe(_device)
_model = instantiate_from_config(_cfg0["model"]).float().to(_device).eval()
_ckpt = os.path.join(_cfg0["weights"]["video2pose_ckpt_root"], _cfg0["experiment"]["exp"],
                     _cfg0["weights"].get("ckpt_name", "video2pos2rot_ckpt_best.pt"))
_ck = torch.load(_ckpt, map_location=_device)
_sd = next((_ck[k] for k in ("model_state", "model", "model_state_dict", "state_dict") if isinstance(_ck, dict) and k in _ck), _ck)
_model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in _sd.items()})
print(f"[app] 模型就绪 {_ckpt}")


def _species_map(base, force_view=None):
    """物种 → ref_seq(Species#Motion/view)。force_view 指定角度(如 'y90' 侧视),
    否则按 y0(正面)→y90→y30 取第一个可用。同一物种取字母序第一个 motion,
    正面/侧面用同一 motion,只换视角。"""
    pose_root = os.path.join(base, "bvh_pose"); train_root = os.path.join(base, "npz_train_image_only")
    views = [force_view] if force_view else ["y0", "y90", "y30"]
    out = {}
    for d in sorted(os.listdir(pose_root)) if os.path.isdir(pose_root) else []:
        sp = d.split("#")[0]
        if sp in out:
            continue
        for v in views:
            ref = f"{d}/{v}"
            if os.path.exists(os.path.join(pose_root, ref + ".npz")) and os.path.exists(os.path.join(train_root, ref + ".npz")):
                out[sp] = ref; break
    return out


THUMB_DIR = os.path.join(REPO, "demo/assets/thumbs")


def _build_ex(name):
    """样例来源:(首帧缩略图, 视频路径) 列表。"""
    mp4s = sorted(glob.glob(os.path.join(EXSET[name], "*.mp4")))
    pin = PINNED.get(name, [])
    rank = {s: i for i, s in enumerate(pin)}
    mp4s.sort(key=lambda p: (rank.get(os.path.splitext(os.path.basename(p))[0], len(pin)),
                             os.path.basename(p)))
    ex_gallery, ex_videos = [], []
    for mp4 in mp4s:
        stem = os.path.splitext(os.path.basename(mp4))[0]
        th = os.path.join(THUMB_DIR, f"{stem.replace('#', '_')}.jpg")
        ex_gallery.append((th if os.path.exists(th) else mp4, stem.split("#")[0].split("_")[0]))
        ex_videos.append(mp4)
    return {"gallery": ex_gallery, "videos": ex_videos}


def _runnable(base, sp, char_sub="characters"):
    """能否完整出结果:需 base_mesh.obj + skinning_weights.npy + *_ffs.bvh(否则 bvh/glb 失败)。
    char_sub 与渲染用的角色目录一致(zoo=characters_fix_facezplus, obj=characters)。"""
    cdir = os.path.join(base, char_sub, sp)
    return (os.path.exists(os.path.join(cdir, "base_mesh.obj")) and
            os.path.exists(os.path.join(cdir, "skinning_weights.npy")) and
            bool(glob.glob(os.path.join(cdir, "*_ffs.bvh"))))


# 手动屏蔽的问题 ref(渲染/参考帧有问题)
BLOCK_REF = {
    "obj": {"12b981a9-d0ad-5abb-84ce-0ebffe25dd48"},   # obj 原第11个,渲染有问题
}


def _build_ref(name, prio_species, cap=72):
    """ref 数据集:物种 → ref_seq;画廊(ref 图, 物种名)。只保留可运行物种,示例排最前,总数封顶。"""
    d = DSET[name]; base = d["base"]
    sp_map = _species_map(base, d.get("force_view"))
    img_dir = os.path.join(base, d.get("ref_img_sub", "ref_images"))
    blocked = BLOCK_REF.get(name, set()) | BLOCK_REF.get(name.replace("_side", ""), set())
    def ok(sp):
        return (sp not in blocked and sp in sp_map and os.path.exists(os.path.join(img_dir, f"{sp}.jpg"))
                and _runnable(base, sp, d.get("char_sub", "characters")))
    prio = [sp for sp in prio_species if ok(sp)]
    rest = [sp for sp in sorted(sp_map) if sp not in prio and ok(sp)]
    ordered = (prio + rest)[:cap]
    gallery = [(os.path.join(img_dir, f"{sp}.jpg"), sp) for sp in ordered]
    d.update(sp_map=sp_map, gallery=gallery, gspecies=[s for _, s in gallery])
    print(f"[app] ref {name}: 可运行物种画廊={len(gallery)}(示例{len(prio)}在前,封顶{cap})")

# 先建样例,再用样例物种给 ref 画廊排优先序
EX = {name: _build_ex(name) for name in EXSET}
def _ex_species(*ex_names):
    out = []
    for en in ex_names:
        for mp4 in EX[en]["videos"]:
            sp = os.path.basename(mp4).split("#")[0].split("_")[0]
            if sp not in out:
                out.append(sp)
    return out
_build_ref("zoo", _ex_species("wild", "zoo"), cap=72)       # zoo 正面(y0):wild+zoo 示例物种在前
_build_ref("zoo_side", _ex_species("wild", "zoo"), cap=72)  # zoo 侧面(y90):同序,共享缩略图
_build_ref("obj", _ex_species("obj"), cap=60)              # obj ref:obj 示例(用户的5个)在前,封顶60
print(f"[app] 样例: " + " ".join(f"{k}={len(v['videos'])}" for k, v in EX.items()))

# ---- 双语文案 ----
LANG = {"cur": "en"}   # 当前语言(默认英文;语言开关会改它;各 handler 读它)

TXT = {
    "zh": {
        "title": "# 🐾 MocapAnything V2\n**MocapAnything · RetargetingAnything · DanceAnything**",
        "lang": "🌐 语言 / Language",
        "in_video": "① 输入视频(上传 / 拖拽 / 选样例)",
        "ref": "② 参考 ref(点右下画廊选)",
        "result": "🎬 pose 结果(输入 | 骨架 camera | 骨架 side)",
        "run": "🚀 运行推理",
        "download": "⬇ 下载结果(mp4 + npy)",
        "ex_label": "样例视频来源(点缩略图载入到①)",
        "ref_label": "目标物种/物体(点图选 · 可与输入不同集 → retarget)",
        "term_hdr": "**运行终端**",
        "idle": "idle — 选视频 + 目标物种后点运行",
        "no_sel": "未选目标物种",
        "sel": "✅ 目标物种:",
        "none": "未选",
        "booting": "启动中…",
        "frames": "抽取帧数",
        "done": "完成",
        "ex_choices": [("🦁 wild(真实野外)", "wild"), ("🕺 human(人体舞蹈)", "human"),
                       ("🐾 zoo(合成)", "zoo"), ("📦 obj(物体)", "obj")],
        "ref_choices": [("🐾 zoo(动物 正面)", "zoo"), ("🐾 zoo(动物 侧面)", "zoo_side"),
                        ("📦 obj(物体)", "obj")],
        "stage": {"dino": "① 提取 DINO 视频特征", "v2p": "② v2p:视频 → 3D pose",
                  "p2r": "③ p2r:pose → joint rotation", "plot": "④ 保存 npy + 绘制骨架视频",
                  "export": "④ 导出交互式 3D", "render": "⑤ blender 渲染 mesh"},
        # Dance Anything
        "d_video": "① 舞蹈视频(带音乐)",
        "d_layers": "② 人物候选层 — 点选「人」那一层",
        "d_result": "🎬 结果:输入 | 角色(带配乐)",
        "d_run": "🚀 运行",
        "d_download": "⬇ 下载",
        "d_maxsec": "最长秒数 (≤10)",
        "d_samples": "**样例舞蹈(带音频)**",
        "d_target": "目标角色",
        "d_termhdr": "**运行终端**",
        "d_status0": "上传/点样例 → 自动分割 → 勾人物层 + 选角色 → 运行",
        "d_idle": "idle — dance anything",
        "share_note": "💚 觉得有意思就分享一下吧!  #MocapAnythingV2",
    },
    "en": {
        "title": "# 🐾 MocapAnything V2\n**MocapAnything · RetargetingAnything · DanceAnything**",
        "lang": "🌐 语言 / Language",
        "in_video": "① Input video (upload / drag / pick a sample)",
        "ref": "② Reference (pick from gallery ↘)",
        "result": "🎬 Pose result (input | skeleton camera | skeleton side)",
        "run": "🚀 Run inference",
        "download": "⬇ Download (mp4 + npy)",
        "ex_label": "Sample video source (click thumbnail → loads into ①)",
        "ref_label": "Target species/object (click to pick · can differ from input → retarget)",
        "term_hdr": "**Run terminal**",
        "idle": "idle — pick a video + target species, then run",
        "no_sel": "No target selected",
        "sel": "✅ Target: ",
        "none": "none",
        "booting": "booting…",
        "frames": "frames extracted",
        "done": "done",
        "ex_choices": [("🦁 wild (real footage)", "wild"), ("🕺 human (dance)", "human"),
                       ("🐾 zoo (synthetic)", "zoo"), ("📦 obj (objects)", "obj")],
        "ref_choices": [("🐾 zoo (animal front)", "zoo"), ("🐾 zoo (animal side)", "zoo_side"),
                        ("📦 obj (objects)", "obj")],
        "stage": {"dino": "① Extract DINO video features", "v2p": "② v2p: video → 3D pose",
                  "p2r": "③ p2r: pose → joint rotation", "plot": "④ Save npy + plot skeleton video",
                  "export": "④ Export interactive 3D", "render": "⑤ Blender mesh render"},
        # Dance Anything
        "d_video": "① Dance video (with audio)",
        "d_layers": "② Person layers — click the person",
        "d_result": "🎬 Result: input | character (+audio)",
        "d_run": "🚀 Run dance",
        "d_download": "⬇ Download",
        "d_maxsec": "Max sec (≤10)",
        "d_samples": "**Sample dance videos (with audio)**",
        "d_target": "Target character",
        "d_termhdr": "**Run terminal**",
        "d_status0": "upload/pick a sample → auto-segment → pick person + character → Run",
        "d_idle": "idle — dance anything",
        "share_note": "💚 Please share if you find it interesting!  #MocapAnythingV2",
    },
}


def T():
    return TXT[LANG["cur"]]


def _fp(path):
    return "/gradio_api/file=" + urllib.parse.quote(os.path.abspath(path))


# 终端配色:全部内联 style(gradio 6 会清洗 <style> 块,故不用 class)
# 绿色系为主,每行循环换色 + 辉光;时间 count 亮蓝,报错亮红
_TERM_BOX = ("background:#050a05;border:1px solid #2c9a5a;border-radius:8px;padding:12px 16px;"
             "font-family:'DejaVu Sans Mono',ui-monospace,Menlo,Consolas,monospace;font-size:13.5px;font-weight:500;"
             "color:#7dffb0;line-height:1.9;overflow:auto;min-height:452px;max-height:452px;"
             "box-shadow:inset 0 0 14px #1c4a2c55")
_TERM_HDR = ("color:#4dffa0;font-weight:700;opacity:.85;font-size:12px;letter-spacing:.08em;"
             "border-bottom:1px solid #1c4a2c;padding-bottom:6px;margin-bottom:8px")
_S_GRN = "color:#4dffa0;font-weight:600;text-shadow:0 0 3px #00ff8844"      # 提示符 绿
_S_BLUE = "color:#5fb8ff;font-weight:600;text-shadow:0 0 3px #2b8fff44"     # 时间 count 蓝
_S_ERR = "color:#ff6b6b;font-weight:600;text-shadow:0 0 3px #ff000033"      # 报错红
# 每行循环配色(绿→青→黄绿→薄荷,彼此有别但柔和)
_PALETTE = ["#7dffb0", "#b6f56a", "#5fe8d0", "#d0f56a", "#4fd6a8", "#a6f58a", "#6fd8ff", "#c4f58a"]


def term_html(lines, running=False):
    cur = f'<span style="{_S_GRN}">█</span>' if running else ""
    rows = []
    for i, ln in enumerate(lines):
        c = _PALETTE[i % len(_PALETTE)]
        rows.append(f'<span style="color:{c};font-weight:500;text-shadow:0 0 3px {c}44">{ln}</span>')
    body = "<br>".join(rows)
    return (f'<div style="{_TERM_BOX}">'
            f'<div style="{_TERM_HDR}">▮ RUN TERMINAL</div>'
            f'<span style="{_S_GRN}">mocap@v2</span>:~$ {body} {cur}</div>')


def run_inference(video_path, species, dset):
    d = DSET[dset]; sp_map = d["sp_map"]
    if not video_path:
        raise gr.Error("请先选择或上传一个视频")
    if not species or species not in sp_map:
        raise gr.Error("请在下方画廊点选一个目标物种")
    ref_seq = sp_map[species]
    stamp = str(int(time.time() * 1000))[-9:]
    seq_name = f"{species}#req{stamp}"
    work = os.path.join(APP_OUT, stamp)
    frames_dir = os.path.join(work, "frames", seq_name); os.makedirs(frames_dir, exist_ok=True)

    lines = [f'run --dataset {dset} --species <b>{html.escape(species[:20])}</b>']
    t0 = time.time()
    yield term_html(lines + [T()["booting"]], True), None, gr.update()

    nfr = video_to_frames(video_path, frames_dir)
    lines.append(f"{T()['frames']}: {nfr}")

    base = d["base"]
    cfg = load_yaml_config(BASE_CONFIG)
    cfg["data"].update(base_dir=base,
                       scale_dict_path=os.path.join(base, "cache/__mesh2pose1002_species_scale_cache.pkl"),
                       character_dir=os.path.join(base, d.get("char_sub", "characters")),
                       bvh_roots=[os.path.join(base, "bvh")],
                       image_roots=[os.path.join(work, "frames")], wild_flag=True)
    cfg["data"]["retarget"].update(ref_seq=ref_seq, ref_idx=0)
    # 和命令行推理一致:跑 blender 渲染出 _final.mp4(输入|骨架cam|mesh_cam|骨架side|mesh_side)。慢 ~4min。
    cfg["output"].update(save_dir=os.path.join(work, "out"), output_tag="app",
                         blender_path=os.path.join(REPO, "blender_mocapanything.sh"))
    cfg["export_glb"] = False

    # 每阶段记录 (名称, 起始时间);展示时算每阶段耗时(下一阶段起始 - 本阶段起始;最后阶段用当前时间)
    state = {"log": [], "done": False, "err": None}
    def _cb(nm): state["log"].append([nm, time.time()])   # 存 key,渲染时按当前语言翻译
    def _worker():
        try:
            inference(cfg=cfg, device=_device, attention_design=cfg["model"]["attention_kwargs"],
                      model=_model, pipe=_pipe, rmbg_net=_rmbg, seq_name=seq_name,
                      image_folder=frames_dir, stage_cb=_cb)
        except Exception as e:
            import traceback; state["err"] = f"{e}\n{traceback.format_exc()[-300:]}"
        finally:
            state["done"] = True
    th = threading.Thread(target=_worker, daemon=True); th.start()

    def _stage_lines(running):
        lg = state["log"]; out = []
        for i, (nm, ts) in enumerate(lg):
            end = lg[i + 1][1] if i + 1 < len(lg) else time.time()
            dur = end - ts
            spin = ' ⏳' if (running and i == len(lg) - 1 and not state["done"]) else ""
            label = T()["stage"].get(nm, nm)
            out.append(f'{label} <span style="{_S_BLUE}">({dur:.1f}s)</span>{spin}')
        return out

    while not state["done"]:
        yield term_html(lines + _stage_lines(True), True), None, gr.update()
        time.sleep(0.2)
    if state["err"]:
        lines += _stage_lines(False)
        lines.append(f'<span style="{_S_ERR}">ERROR: {html.escape(state["err"][:180])}</span>')
        yield term_html(lines), None, gr.update(); return
    lines += _stage_lines(False)

    # blender 渲染结果 _final.mp4(输入|骨架cam|mesh_cam|骨架side|mesh_side)
    finals = glob.glob(os.path.join(work, "out", "**", "*_final.mp4"), recursive=True)
    npys = glob.glob(os.path.join(work, "out", "**", "*_pred.npy"), recursive=True)
    if not finals:
        lines.append(f'<span style="{_S_ERR}">未生成渲染 mp4(检查 BLENDER_BIN / ffmpeg)</span>')
        yield term_html(lines), None, gr.update(); return
    import shutil, zipfile
    safe_mp4 = os.path.join(work, os.path.basename(finals[0]).replace("#", "_"))
    shutil.copy2(finals[0], safe_mp4)
    zpath = os.path.join(work, f"{species[:16]}_result.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(safe_mp4, os.path.basename(safe_mp4))
        for f in npys:
            if f.endswith("_pred.npy"):
                zf.write(f, os.path.basename(f))
    lines.append(f'<span style="{_S_GRN}">✔ {T()["done"]} {time.time()-t0:0.1f}s · {nfr} frames · pose mp4 + npy</span>')
    yield term_html(lines), safe_mp4, gr.update(value=zpath, visible=True)


def on_species_select(dset, evt: gr.SelectData):
    d = DSET[dset]
    i = evt.index
    if i is None or i >= len(d["gspecies"]):
        return None, T()["none"], None
    sp = d["gspecies"][i]
    ref_img = d["gallery"][i][0]
    return sp, f'{T()["sel"]}{sp[:24]}', ref_img


def on_example_select(name, evt: gr.SelectData):
    vids = EX[name]["videos"]
    if evt.index is None or evt.index >= len(vids):
        return None
    src = vids[evt.index]
    # 拷成无 # 的安全名(gr.Video 服务 URL 遇 # 会截断 → 显示失败)
    import shutil
    safe = os.path.join(APP_OUT, "_ex_" + os.path.basename(src).replace("#", "_"))
    try:
        if not os.path.exists(safe):
            shutil.copy2(src, safe)
        return safe
    except Exception:
        return src


def on_ref_change(name):
    d = DSET[name]
    return gr.update(value=[i for i, _ in d["gallery"]]), None, T()["no_sel"], None

def on_ex_change(name):
    return gr.update(value=[t for t, _ in EX[name]["gallery"]])


def on_lang_change(lang):
    LANG["cur"] = lang if lang in TXT else "zh"
    t = T()
    return (
        gr.update(value=t["title"]),                              # title_md
        gr.update(label=t["in_video"]),                           # video_in
        gr.update(label=t["ref"]),                                # ref_img
        gr.update(label=t["result"]),                             # result_out
        gr.update(value=t["run"]),                                # run_btn
        gr.update(value=t["no_sel"]),                             # sel_label(切换语言重置选择提示)
        gr.update(label=t["download"]),                           # dl_btn
        gr.update(label=t["ex_label"], choices=t["ex_choices"]),  # ex_ds
        gr.update(label=t["ref_label"], choices=t["ref_choices"]),# ref_ds
        term_html([t["idle"]]),                                   # term
        gr.update(value=t["share_note"]),                         # tabnote_md
        # ---- Dance tab ----
        gr.update(label=t["d_video"]),                            # dance_video
        gr.update(label=t["d_layers"]),                           # dcand_gallery
        gr.update(label=t["d_result"]),                           # dance_out
        gr.update(label=t["d_maxsec"]),                           # dmax
        gr.update(value=t["d_run"]),                              # drun_btn
        gr.update(value=t["d_status0"]),                          # dstatus
        gr.update(label=t["d_download"]),                         # ddl_btn
        gr.update(value=t["d_samples"]),                          # dsamp_md
        gr.update(label=t["d_target"], choices=t["ref_choices"]), # dref_ds
        term_html([t["d_idle"]]),                                 # dterm
    )


# ================= Dance Anything =================
DANCE_FPS = 30


def _cfg_and_infer(work, frames_dir, seq_name, d, ref_seq, lines, t0, tag="dance"):
    """公用:建 cfg(fps=30)+ 起推理线程 + 阶段流。yield (term_html, None, update) 直到完成;
    返回时 lines 已含各阶段。出错则最后一次 yield 后调用方应 return。"""
    base = d["base"]
    cfg = load_yaml_config(BASE_CONFIG)
    cfg["data"].update(base_dir=base,
                       scale_dict_path=os.path.join(base, "cache/__mesh2pose1002_species_scale_cache.pkl"),
                       character_dir=os.path.join(base, d.get("char_sub", "characters")),
                       bvh_roots=[os.path.join(base, "bvh")],
                       image_roots=[os.path.join(work, "frames")], wild_flag=True)
    cfg["data"]["retarget"].update(ref_seq=ref_seq, ref_idx=0)
    cfg["output"].update(save_dir=os.path.join(work, "out"), output_tag=tag,
                         blender_path=os.path.join(REPO, "blender_mocapanything.sh"), fps=DANCE_FPS)
    cfg["export_glb"] = False
    state = {"log": [], "done": False, "err": None}
    def _cb(nm): state["log"].append([nm, time.time()])
    def _worker():
        try:
            inference(cfg=cfg, device=_device, attention_design=cfg["model"]["attention_kwargs"],
                      model=_model, pipe=_pipe, rmbg_net=_rmbg, seq_name=seq_name,
                      image_folder=frames_dir, stage_cb=_cb)
        except Exception as e:
            import traceback; state["err"] = f"{e}\n{traceback.format_exc()[-300:]}"
        finally:
            state["done"] = True
    threading.Thread(target=_worker, daemon=True).start()
    def _stage_lines(running):
        lg = state["log"]; out = []
        for i, (nm, ts) in enumerate(lg):
            end = lg[i + 1][1] if i + 1 < len(lg) else time.time()
            spin = ' ⏳' if (running and i == len(lg) - 1 and not state["done"]) else ""
            out.append(f'{T()["stage"].get(nm, nm)} <span style="{_S_BLUE}">({end-ts:.1f}s)</span>{spin}')
        return out
    while not state["done"]:
        yield term_html(lines + _stage_lines(True), True), None, gr.update()
        time.sleep(0.2)
    lines += _stage_lines(False)
    if state["err"]:
        lines.append(f'<span style="{_S_ERR}">ERROR: {html.escape(state["err"][:180])}</span>')
        state["failed"] = True
    yield ("__DONE__", state.get("failed", False), None)


def dance_segment(video_path, max_sec):
    """上传舞蹈视频 → 标准化 30fps + 抽帧 + 抽音轨 + SAM 候选图层。"""
    if not video_path:
        raise gr.Error("请先上传舞蹈视频 / upload a dance video first")
    stamp = str(int(time.time() * 1000))[-9:]
    work = os.path.join(APP_OUT, "dance" + stamp); os.makedirs(work, exist_ok=True)
    std = du.standardize_30fps(video_path, os.path.join(work, "std.mp4"), max_seconds=float(max_sec), fps=DANCE_FPS)
    orig_dir = os.path.join(work, "orig"); du.frames_from_video(std, orig_dir, ext="png")
    sam_dir = os.path.join(work, "sam"); du.frames_from_video(std, sam_dir, ext="jpg")
    audio = du.extract_audio(video_path, os.path.join(work, "audio.m4a"))
    prevs, segs = su.candidate_layers(os.path.join(orig_dir, "00000.png"), os.path.join(work, "cand"), topk=8)
    ctx = {"work": work, "std": std, "orig_dir": orig_dir, "sam_dir": sam_dir, "audio": audio, "segs": segs}
    if prevs:
        status = f"✅ SAM found **{len(prevs)}** layers — click the person layer / 点选「人」那一层"
    else:
        status = "⚠ SAM found no layer — will fall back to RMBG auto-matting / 回退 RMBG 自动抠人"
    return ctx, prevs, status, None


def dance_pick(evt: gr.SelectData):
    return evt.index, f"✅ picked person layer **#{evt.index + 1}** / 选中第 {evt.index + 1} 层 — pick a character then Run"


def dance_example_load(max_sec, evt: gr.SelectData):
    """点样例 dance 视频(带音频):载入 ① 并自动 SAM 分割 → 候选人物图层。"""
    vids = EX["dance"]["videos"]
    if evt.index is None or evt.index >= len(vids):
        return gr.update(), None, [], "…", None
    import shutil
    src = vids[evt.index]
    safe = os.path.join(APP_OUT, "_dex_" + os.path.basename(src))
    if not os.path.exists(safe):
        shutil.copy2(src, safe)
    ctx, prevs, status, pick = dance_segment(safe, max_sec)
    return safe, ctx, prevs, status, pick


def d_ref_change(name):
    d = DSET[name]
    return gr.update(value=[i for i, _ in d["gallery"]]), None, "Pick a target character / 选一个目标角色"


def d_species_select(dset, evt: gr.SelectData):
    d = DSET[dset]; i = evt.index
    if i is None or i >= len(d["gspecies"]):
        return None, "none"
    sp = d["gspecies"][i]
    return sp, f"✅ character: **{sp[:24]}** — click Run / 已选角色,点 Run"


def run_dance(ctx, pick_idx, species, dset):
    if not ctx or not ctx.get("orig_dir"):
        raise gr.Error("请先上传并点 Segment / upload & click Segment first")
    d = DSET[dset]; sp_map = d["sp_map"]
    if not species or species not in sp_map:
        raise gr.Error("请点选目标角色 / pick a target character")
    ref_seq = sp_map[species]
    work = ctx["work"]; seq_name = f"{species}#d{os.path.basename(work)}"
    frames_dir = os.path.join(work, "frames", seq_name); os.makedirs(frames_dir, exist_ok=True)
    lines = [f'dance --char <b>{html.escape(species[:20])}</b> --fps 30 --max {len(os.listdir(ctx["orig_dir"]))}f']
    t0 = time.time()
    yield term_html(lines + ["SAM matting…"], True), None, gr.update()

    # 1) 抠人 → RGBA 帧(SAM 选中层跟踪;失败/未选 → 回退原图,推理内部 RMBG)
    used = "RMBG"
    if pick_idx is not None and ctx.get("segs") and 0 <= pick_idx < len(ctx["segs"]):
        n = su.track_to_rgba(ctx["sam_dir"], ctx["segs"][pick_idx], ctx["orig_dir"], frames_dir)
        if n > 0:
            used = "SAM"
    if used != "SAM":
        import shutil
        for f in sorted(os.listdir(ctx["orig_dir"])):
            shutil.copy2(os.path.join(ctx["orig_dir"], f), os.path.join(frames_dir, f))
    lines.append(f'matting: <b>{used}</b> · {len(os.listdir(frames_dir))} frames')

    # 2) 推理(fps=30)+ 渲染
    failed = False
    for out in _cfg_and_infer(work, frames_dir, seq_name, d, ref_seq, lines, t0):
        if out[0] == "__DONE__":
            failed = out[1]; break
        yield out
    if failed:
        yield term_html(lines), None, gr.update(); return

    # 3) 合成:原图 | 角色 mesh(camera)+ 配乐回填
    mesh_cam = glob.glob(os.path.join(work, "out", "**", "camera", "*rot6d_pred.mp4"), recursive=True)
    if not mesh_cam:
        lines.append(f'<span style="{_S_ERR}">no character render (blender)</span>')
        yield term_html(lines), None, gr.update(); return
    comp = os.path.join(work, "dance_side.mp4")
    du.compose_side_by_side([ctx["std"], mesh_cam[0]], comp, height=480, fps=DANCE_FPS)
    outp = os.path.join(work, f"dance_{species[:16]}.mp4")
    du.mux_audio(comp, ctx["audio"], outp)
    tag_audio = "+audio" if ctx.get("audio") else "no-audio"
    lines.append(f'<span style="{_S_GRN}">✔ {T()["done"]} {time.time()-t0:0.1f}s · {used} · 30fps · {tag_audio}</span>')
    yield term_html(lines), outp, gr.update(value=outp, visible=True)


_CSS = (
    # 整页居中
    ".gradio-container{max-width:1520px!important;margin:0 auto!important}"
    # 大标题 + 副标题
    "#hdr{align-items:center;margin-bottom:8px}"
    "#hdr h1{font-size:2rem;margin:0;line-height:1.2;font-weight:800}"
    "#title p{font-size:1.02rem;opacity:.72;margin:3px 0 0;letter-spacing:.02em}"
    "#lang{max-width:260px}"
    # selbar 一排:顶部三项等高等宽、垂直居中
    "#selbar{align-items:stretch!important}"
    "#selbar>*{align-self:stretch!important}"
    "#selbar button,#sel,#dsel{min-height:56px!important}"
    # 状态框(Target/status):与按钮同高,内容居中,超长最多 2 行省略
    "#sel,#dsel{display:flex!important;align-items:center;justify-content:center}"
    "#sel p,#dsel p{text-align:center;font-size:.98rem;margin:0;opacity:.9;"
    "display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}"
    # 语言开关:横排一行
    "#lang .wrap{display:flex!important;flex-flow:row wrap;align-items:center;gap:6px 16px}"
    # 样例来源 / 目标角色 radio:纵列(一行一个)
    "#exds .wrap,#refds .wrap,#drefds .wrap{display:flex!important;flex-flow:column;"
    "align-items:flex-start;gap:8px}"
    # 圆圈与文字垂直居中对齐
    "#lang .wrap label,#exds .wrap label,#refds .wrap label,#drefds .wrap label"
    "{display:inline-flex;align-items:center;gap:6px;margin:0}"
    "#lang .wrap input,#exds .wrap input,#refds .wrap input,#drefds .wrap input{margin:0;flex:none}"
    # tab 突出:更大 + 阴影 + 边框 + 选中高亮
    ".tab-nav{gap:8px;border:0!important}"
    ".tab-nav button,button[role=tab]{font-size:1.2rem!important;font-weight:700!important;"
    "padding:10px 22px!important;border-radius:12px 12px 0 0!important;border:1px solid rgba(120,120,120,.25)!important;"
    "box-shadow:0 3px 12px rgba(0,0,0,.16)!important}"
    ".tab-nav button.selected,button[role=tab][aria-selected=true]{"
    "box-shadow:0 6px 18px rgba(60,175,106,.38)!important;border-color:#3caf6a!important;"
    "border-bottom:3px solid #3caf6a!important}"
    # 各方块:阴影 + 圆角(边缘更清晰)
    ".gradio-container .block{box-shadow:0 2px 12px rgba(0,0,0,.12)!important;border-radius:12px!important}"
    # tab 行右侧的分享提示语(与 tab 标签同一排,负边距压到 tab-nav 行)
    "#tabnote{box-shadow:none!important;background:transparent!important;border:0!important;"
    "margin:0 4px -42px 0!important;text-align:right;position:relative;z-index:5;pointer-events:none}"
    "#tabnote p{display:inline-block;margin:8px 0;color:#2c9a5a;font-weight:700;font-size:.98rem}"
    # 「样例舞蹈」标题上色(与其它 label 一致的强调色)
    "#dsamphdr{box-shadow:none!important;background:transparent!important;border:0!important;padding:0!important}"
    "#dsamphdr p,#dsamphdr strong{color:#7c5cff;font-weight:700}"
)
_z = TXT["en"]   # 默认英文
with gr.Blocks(title="MocapAnything V2 Demo", theme=gr.themes.Soft(), css=_CSS) as demo:
    with gr.Row(elem_id="hdr"):
        title_md = gr.Markdown(_z["title"], elem_id="title")
        lang_ds = gr.Radio(choices=[("中文", "zh"), ("English", "en")], value="en",
                           label=_z["lang"], interactive=True, scale=0, min_width=260, elem_id="lang")
    species_state = gr.State(None)

    tabnote_md = gr.Markdown(_z["share_note"], elem_id="tabnote")
    with gr.Tabs():
      # ========== Tab 1:Mocap / Retarget ==========
      with gr.Tab("🎯 Mocap · Retarget"):
        # 第 1 排:输入视频 | 参考 ref | 结果渲染
        with gr.Row(equal_height=True):
            video_in = gr.Video(label=_z["in_video"], height=280, scale=1)
            ref_img = gr.Image(label=_z["ref"], height=280, interactive=False, scale=1)
            result_out = gr.Video(label=_z["result"], height=280, autoplay=True, scale=3)

        with gr.Row(elem_id="selbar", equal_height=False):
            run_btn = gr.Button(_z["run"], variant="primary", size="lg", scale=2)
            sel_label = gr.Markdown(_z["no_sel"], elem_id="sel")
            dl_btn = gr.DownloadButton(_z["download"], size="lg", scale=2)

        # 第 2 排:样例视频 | 目标物种画廊 | 运行终端
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                ex_ds = gr.Radio(choices=_z["ex_choices"], value="wild", label=_z["ex_label"], interactive=True, elem_id="exds")
                ex_gallery = gr.Gallery(value=[t for t, _ in EX["wild"]["gallery"]], columns=3, height=280,
                                        object_fit="cover", show_label=False, allow_preview=False)
            with gr.Column(scale=4):
                ref_ds = gr.Radio(choices=_z["ref_choices"], value="zoo", label=_z["ref_label"], interactive=True, elem_id="refds")
                species_gallery = gr.Gallery(value=[i for i, _ in DSET["zoo"]["gallery"]], columns=8, height=280,
                                             object_fit="cover", show_label=False, allow_preview=False)
            with gr.Column(scale=3):
                term = gr.HTML(term_html([_z["idle"]]))

      # ========== Tab 2:Dance Anything(布局与 Mocap 一致)==========
      with gr.Tab("💃 Dance Anything"):
        dance_ctx = gr.State(None)          # 分割上下文(帧目录/音轨/候选蒙版)
        dance_pick_idx = gr.State(None)     # 选中的人物图层 index
        dspecies_state = gr.State(None)     # 选中的目标角色
        # 第 1 排:① 舞蹈视频 | ② 人物候选层(SAM,点选) | 🎬 结果(输入|角色 + 配乐)
        with gr.Row(equal_height=True):
            dance_video = gr.Video(label=_z["d_video"], height=280, scale=1)
            dcand_gallery = gr.Gallery(label=_z["d_layers"], columns=2, height=280,
                                       object_fit="contain", allow_preview=False, scale=1)
            dance_out = gr.Video(label=_z["d_result"], height=280, autoplay=False, scale=3)
        with gr.Row(elem_id="selbar", equal_height=False):
            dmax = gr.Slider(3, 10, value=10, step=1, label=_z["d_maxsec"], scale=1)
            drun_btn = gr.Button(_z["d_run"], variant="primary", size="lg", scale=2)
            dstatus = gr.Markdown(_z["d_status0"], elem_id="dsel")
            ddl_btn = gr.DownloadButton(_z["d_download"], size="lg", scale=2)
        # 第 2 排:样例舞蹈(带音频) | 目标角色画廊 | 运行终端
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                dsamp_md = gr.Markdown(_z["d_samples"], elem_id="dsamphdr")
                dex_gallery = gr.Gallery(value=[t for t, _ in EX["dance"]["gallery"]], columns=3, height=280,
                                         object_fit="cover", show_label=False, allow_preview=False)
            with gr.Column(scale=4):
                dref_ds = gr.Radio(choices=_z["ref_choices"], value="zoo",
                                   label=_z["d_target"], interactive=True, elem_id="drefds")
                dspecies_gallery = gr.Gallery(value=[i for i, _ in DSET["zoo"]["gallery"]], columns=8, height=280,
                                              object_fit="cover", show_label=False, allow_preview=False)
            with gr.Column(scale=3):
                dterm = gr.HTML(term_html([_z["d_idle"]]))

    # ---- Tab1 交互 ----
    lang_ds.change(on_lang_change, inputs=[lang_ds],
                   outputs=[title_md, video_in, ref_img, result_out, run_btn, sel_label,
                            dl_btn, ex_ds, ref_ds, term, tabnote_md,
                            dance_video, dcand_gallery, dance_out, dmax, drun_btn, dstatus,
                            ddl_btn, dsamp_md, dref_ds, dterm])
    ex_ds.change(on_ex_change, inputs=[ex_ds], outputs=[ex_gallery])
    ref_ds.change(on_ref_change, inputs=[ref_ds], outputs=[species_gallery, species_state, sel_label, ref_img])
    ex_gallery.select(on_example_select, inputs=[ex_ds], outputs=[video_in])
    species_gallery.select(on_species_select, inputs=[ref_ds], outputs=[species_state, sel_label, ref_img])
    run_btn.click(run_inference, inputs=[video_in, species_state, ref_ds],
                  outputs=[term, result_out, dl_btn])

    # ---- Tab2(Dance)交互:上传/点样例 → 自动 SAM 分割 → 点人物层 + 选角色 → Run ----
    dance_video.upload(dance_segment, inputs=[dance_video, dmax],
                       outputs=[dance_ctx, dcand_gallery, dstatus, dance_pick_idx])
    dex_gallery.select(dance_example_load, inputs=[dmax],
                       outputs=[dance_video, dance_ctx, dcand_gallery, dstatus, dance_pick_idx])
    dcand_gallery.select(dance_pick, inputs=None, outputs=[dance_pick_idx, dstatus])
    dref_ds.change(d_ref_change, inputs=[dref_ds], outputs=[dspecies_gallery, dspecies_state, dstatus])
    dspecies_gallery.select(d_species_select, inputs=[dref_ds], outputs=[dspecies_state, dstatus])
    drun_btn.click(run_dance, inputs=[dance_ctx, dance_pick_idx, dspecies_state, dref_ds],
                   outputs=[dterm, dance_out, ddl_btn])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name=os.environ.get("APP_HOST", "0.0.0.0"),
        server_port=int(os.environ.get("APP_PORT", "7860")),
        allowed_paths=[os.path.join(REPO, "demo"), os.path.join(REPO, "demo_outputs"),
                       DSET["zoo"]["base"], DSET["obj"]["base"],
                       os.path.realpath(DSET["zoo"]["base"]), os.path.realpath(DSET["obj"]["base"])],
        share=False)

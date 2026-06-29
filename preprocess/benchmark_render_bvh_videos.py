import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.mesh import BVH, blender_visualize_character_motion


def _choose_bvhs(bvh_root, motions, view, motion_filter=None):
    selected = []
    for motion_dir in sorted(glob(os.path.join(bvh_root, "*"))):
        if not os.path.isdir(motion_dir):
            continue
        motion_name = os.path.basename(motion_dir)
        if motion_filter and motion_filter not in motion_name:
            continue
        path = os.path.join(motion_dir, f"{view}.bvh")
        if os.path.exists(path):
            selected.append(path)
        if len(selected) >= motions:
            break
    return selected


def _limited_bvh(src_path, dst_root, max_frames):
    motion = os.path.basename(os.path.dirname(src_path))
    view = Path(src_path).stem
    dst = os.path.join(dst_root, motion, f"{view}.bvh")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if max_frames:
        anim, names, frametime = BVH.load(src_path)
        anim.positions = anim.positions[:max_frames]
        anim.rotations = anim.rotations[:max_frames]
        BVH.save(dst, anim, names, frametime)
    else:
        shutil.copy2(src_path, dst)
    return dst


def _run_original(args, bvhs):
    t0 = time.time()
    outputs = []
    for path in bvhs:
        motion = os.path.basename(os.path.dirname(path))
        character = motion.split("#")[0]
        out_dir = os.path.join(args.benchmark_root, "old", motion)
        os.makedirs(out_dir, exist_ok=True)
        blender_visualize_character_motion(
            out_dir,
            path,
            character_folder=os.path.join(args.character_root, character),
            auto_scale="bvh",
            blender_path=args.blender,
            hdri_path=args.hdri,
            view_scale=args.view_scale,
            object_position=args.object_position,
            camera_trace=False,
            fps=args.fps,
        )
        outputs.append(os.path.join(out_dir, f"{Path(path).stem}.mp4"))
        for name in os.listdir(out_dir):
            full = os.path.join(out_dir, name)
            if os.path.isdir(full):
                shutil.rmtree(full)
            elif not name.endswith(".mp4"):
                os.remove(full)
    return {"seconds": time.time() - t0, "outputs": outputs}


def _run_fast(args, bvh_root):
    t0 = time.time()
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "render_bvh_videos_fast.py"),
        "--zoo-root",
        args.zoo_root,
        "--bvh-root",
        bvh_root,
        "--character-root",
        args.character_root,
        "--output-root",
        os.path.join(args.benchmark_root, "fast"),
        "--work-root",
        os.path.join(args.benchmark_root, "fast_work"),
        "--blender",
        args.blender,
        "--scene",
        args.scene,
        "--hdri",
        args.hdri,
        "--workers",
        str(args.fast_workers),
        "--views",
        args.view,
        "--fps",
        str(args.fps),
        "--view-scale",
        str(args.view_scale),
        "--object-position",
        str(args.object_position),
        "--eevee-samples",
        str(args.eevee_samples),
        "--image-format",
        args.image_format,
    ]
    subprocess.run(cmd, check=True)
    return {"seconds": time.time() - t0, "command": cmd}


def main():
    parser = argparse.ArgumentParser(description="Compare original MocapAnything renderer with fast batched renderer.")
    parser.add_argument("--zoo-root", default="zoo")
    parser.add_argument("--bvh-root")
    parser.add_argument("--character-root")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--motions", type=int, default=10)
    parser.add_argument("--motion-filter")
    parser.add_argument("--view", default="y0")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    parser.add_argument("--scene", default=os.path.join("preprocess", "blank.blend"))
    parser.add_argument("--hdri", default="transparent")
    parser.add_argument("--fast-workers", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--view-scale", type=float, default=1.3)
    parser.add_argument("--object-position", type=float, default=1.0)
    parser.add_argument("--eevee-samples", type=int, default=64)
    parser.add_argument("--image-format", choices=["PNG", "JPEG"], default="PNG")
    args = parser.parse_args()

    args.bvh_root = args.bvh_root or os.path.join(args.zoo_root, "bvh")
    args.character_root = args.character_root or os.path.join(args.zoo_root, "characters_fix_facezplus")
    os.makedirs(args.benchmark_root, exist_ok=True)

    selected = _choose_bvhs(args.bvh_root, args.motions, args.view, args.motion_filter)
    if len(selected) < args.motions:
        raise SystemExit(f"Only found {len(selected)} BVHs for view {args.view}; requested {args.motions}.")

    limited_root = os.path.join(args.benchmark_root, "limited_bvh")
    limited = [_limited_bvh(path, limited_root, args.max_frames) for path in selected]
    manifest = {
        "motions": [os.path.basename(os.path.dirname(path)) for path in limited],
        "motion_filter": args.motion_filter,
        "view": args.view,
        "max_frames": args.max_frames,
    }
    with open(os.path.join(args.benchmark_root, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    old = _run_original(args, limited)
    fast = _run_fast(args, limited_root)
    summary = {
        **manifest,
        "original_seconds": old["seconds"],
        "fast_seconds": fast["seconds"],
        "speedup": old["seconds"] / fast["seconds"] if fast["seconds"] > 0 else None,
        "old_outputs": old["outputs"],
        "fast_output_root": os.path.join(args.benchmark_root, "fast"),
    }
    with open(os.path.join(args.benchmark_root, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.mesh import BVH, extract_mesh_from_bvh
from utils.common import get_diameter, interchange_y_z_axis, sm_loop
from utils.visualization import convert_images_to_video


def _split_even(items, chunks):
    chunks = max(1, min(chunks, len(items)))
    return [items[index::chunks] for index in range(chunks) if items[index::chunks]]


def _select_bvhs(bvh_root, limit=None, motion_filter=None, exact_motion_filter=None, views=None):
    paths = sorted(glob(os.path.join(bvh_root, "*", "*.bvh")))
    if motion_filter:
        paths = [path for path in paths if motion_filter in os.path.basename(os.path.dirname(path))]
    if exact_motion_filter:
        keep = set(exact_motion_filter)
        paths = [path for path in paths if os.path.basename(os.path.dirname(path)) in keep]
    if views:
        keep_views = {view[:-4] if view.endswith(".bvh") else view for view in views}
        paths = [path for path in paths if Path(path).stem in keep_views]
    if limit:
        paths = paths[:limit]
    return paths


def _save_limited_bvh(src_path, dst_path, max_frames):
    if not max_frames:
        return src_path
    anim, names, frametime = BVH.load(src_path)
    if len(anim.positions) <= max_frames:
        return src_path
    anim.positions = anim.positions[:max_frames]
    anim.rotations = anim.rotations[:max_frames]
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    BVH.save(dst_path, anim, names, frametime)
    return dst_path


def _auto_scale_from_bvh(motion_path):
    anim, _, _ = BVH.load(motion_path)
    bone_length = np.linalg.norm(anim.offsets, axis=1)
    diameter, _ = get_diameter(anim.parents, bone_length)
    return 1 / diameter


def _prepare_job(args, motion_path):
    motion_name = os.path.basename(os.path.dirname(motion_path))
    view = Path(motion_path).stem
    character = motion_name.split("#")[0]
    character_folder = os.path.join(args.character_root, character)
    output_dir = os.path.join(args.output_root, motion_name)
    video_path = os.path.join(output_dir, f"{view}.mp4")
    if args.skip_existing and os.path.exists(video_path):
        return {"status": "skipped", "video": video_path}

    job_root = os.path.join(args.work_root, "jobs", motion_name, view)
    mesh_dir = os.path.join(job_root, "mesh")
    image_dir = os.path.join(job_root, "images")
    camera_path = os.path.join(job_root, "camera_trajectory.npy")
    limited_bvh = os.path.join(args.work_root, "limited_bvh", motion_name, f"{view}.bvh")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    render_motion = _save_limited_bvh(motion_path, limited_bvh, args.max_frames)
    scale = _auto_scale_from_bvh(render_motion) if args.auto_scale == "bvh" else 1.0
    skin_weight_path = os.path.join(character_folder, "skinning_weights.npy")
    if not os.path.exists(skin_weight_path):
        skin_weight_path = os.path.join(character_folder, "skin_weights.npy")

    root_pos = extract_mesh_from_bvh(
        render_motion,
        os.path.join(character_folder, "base_mesh.obj"),
        mesh_dir,
        skin_weight_path,
        unicube=False,
        vis_skeleton=False,
        return_root_pos=True,
        mesh_format="obj",
        scale=scale,
        mesh_scale=args.mesh_scale,
        azim=args.azim,
    )

    root_pos_sm = sm_loop(root_pos, args.traj_smooth)
    if args.camera_trace:
        root_pos_sm[:, 2] = -root_pos_sm[:, 2]
        camera_traj = root_pos_sm + np.array([0, args.object_position, -3.8 / args.view_scale])
    else:
        camera_traj = np.tile(
            np.array([0, args.object_position, -3.8 / args.view_scale]),
            (root_pos_sm.shape[0], 1),
        )
    np.save(camera_path, interchange_y_z_axis(camera_traj))

    return {
        "status": "ready",
        "motion": motion_name,
        "view": view,
        "character": character,
        "character_folder": character_folder,
        "mesh_folder": mesh_dir,
        "image_dir": image_dir,
        "camera_traj": camera_path,
        "video": video_path,
    }


def _write_jobs(path, jobs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mesh_folder", "output_dir", "camera_traj"],
            delimiter="\t",
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "mesh_folder": job["mesh_folder"],
                    "output_dir": job["image_dir"],
                    "camera_traj": job["camera_traj"],
                }
            )


def _run_blender(args, jobs, worker_id):
    jobs_file = os.path.join(args.work_root, "job_files", f"worker_{worker_id:03d}.tsv")
    log_file = os.path.join(args.work_root, "logs", f"blender_worker_{worker_id:03d}.log")
    _write_jobs(jobs_file, jobs)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    script = os.path.join(os.path.dirname(__file__), "blender_render_bvh_batch.py")
    cmd = [
        args.blender,
        "-b",
        "--python",
        script,
        "--",
        "--jobs-file",
        jobs_file,
        "--scene",
        args.scene,
        "--character-folder",
        jobs[0]["character_folder"],
        "--hdri",
        args.hdri,
        "--engine",
        args.engine,
        "--samples",
        str(args.eevee_samples),
        "--image-format",
        args.image_format,
        "--jpeg-quality",
        str(args.jpeg_quality),
    ]
    with open(log_file, "w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=log, check=True)


def _encode_job(args, job):
    convert_images_to_video(job["image_dir"], job["video"], fps=args.fps, crf=args.crf, preset=args.preset)
    return job["video"]


def main():
    parser = argparse.ArgumentParser(description="Fast batched BVH video renderer for MocapAnything preprocess.")
    parser.add_argument("--zoo-root", default="zoo")
    parser.add_argument("--bvh-root")
    parser.add_argument("--character-root")
    parser.add_argument("--output-root")
    parser.add_argument("--work-root")
    parser.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    parser.add_argument("--scene", default=os.path.join("preprocess", "blank.blend"))
    parser.add_argument("--hdri", default="transparent")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--motion-filter")
    parser.add_argument("--exact-motion-filter", action="append")
    parser.add_argument("--views", help="Comma-separated view names, e.g. y0,y30")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--engine", default=os.environ.get("BLENDER_RENDER_ENGINE", "BLENDER_EEVEE_NEXT"))
    parser.add_argument("--eevee-samples", type=int, default=int(os.environ.get("BLENDER_EEVEE_SAMPLES", "64")))
    parser.add_argument("--image-format", choices=["PNG", "JPEG"], default=os.environ.get("BLENDER_IMAGE_FORMAT", "PNG"))
    parser.add_argument("--jpeg-quality", type=int, default=int(os.environ.get("BLENDER_JPEG_QUALITY", "95")))
    parser.add_argument("--view-scale", type=float, default=1.3)
    parser.add_argument("--object-position", type=float, default=1.0)
    parser.add_argument("--camera-trace", action="store_true")
    parser.add_argument("--traj-smooth", type=float, default=0.8)
    parser.add_argument("--auto-scale", choices=["none", "bvh"], default="bvh")
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--azim", type=float, default=0.0)
    parser.add_argument("--cleanup-work", action="store_true")
    args = parser.parse_args()

    args.bvh_root = args.bvh_root or os.path.join(args.zoo_root, "bvh")
    args.character_root = args.character_root or os.path.join(args.zoo_root, "characters_fix_facezplus")
    args.output_root = args.output_root or os.path.join(args.zoo_root, "video")
    args.work_root = args.work_root or os.path.join(args.zoo_root, "video_fast_work")
    if args.auto_scale == "none":
        args.auto_scale = None
    if args.views:
        args.views = [item.strip() for item in args.views.split(",") if item.strip()]
    if not os.path.exists(args.scene):
        raise SystemExit(f"Blender scene file not found: {args.scene}")

    t0 = time.time()
    selected = _select_bvhs(
        args.bvh_root,
        limit=args.limit,
        motion_filter=args.motion_filter,
        exact_motion_filter=args.exact_motion_filter,
        views=args.views,
    )
    if not selected:
        raise SystemExit("No BVH files matched the requested filters.")

    prepared = []
    skipped = 0
    for path in tqdm(selected, desc="Preparing mesh sequences"):
        job = _prepare_job(args, path)
        if job["status"] == "skipped":
            skipped += 1
        else:
            prepared.append(job)

    prepare_seconds = time.time() - t0
    render_seconds = 0.0
    if prepared:
        by_character = defaultdict(list)
        for job in prepared:
            by_character[job["character"]].append(job)

        worker_jobs = []
        worker_id = 0
        for jobs in by_character.values():
            for part in _split_even(jobs, args.workers):
                worker_jobs.append((worker_id, part))
                worker_id += 1

        t_render = time.time()
        with ThreadPoolExecutor(max_workers=min(args.workers, len(worker_jobs))) as executor:
            futures = [
                executor.submit(_run_blender, args, jobs, worker_id)
                for worker_id, jobs in worker_jobs
            ]
            for future in as_completed(futures):
                future.result()
        render_seconds = time.time() - t_render

        t_encode = time.time()
        for job in tqdm(prepared, desc="Encoding videos"):
            _encode_job(args, job)
        encode_seconds = time.time() - t_encode
    else:
        encode_seconds = 0.0

    summary = {
        "selected": len(selected),
        "rendered": len(prepared),
        "skipped": skipped,
        "prepare_seconds": prepare_seconds,
        "render_seconds": render_seconds,
        "encode_seconds": encode_seconds,
        "total_seconds": time.time() - t0,
        "output_root": args.output_root,
    }
    os.makedirs(args.work_root, exist_ok=True)
    with open(os.path.join(args.work_root, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))

    if args.cleanup_work:
        shutil.rmtree(args.work_root, ignore_errors=True)


if __name__ == "__main__":
    main()

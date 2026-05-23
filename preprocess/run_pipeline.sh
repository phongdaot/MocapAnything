#!/usr/bin/env bash
# Run the full MocapAnything preprocessing pipeline end-to-end.
#
# Stages (in order):
#   1. move_fbx        : flatten Truebone FBX files into character folders
#   2. extract_char    : extract base mesh + per-motion BVH + rest pose (Blender)
#   3. rotate_bvh      : rotate every motion BVH around Y axis x12
#   4. extract_pose    : per-BVH pose npz (positions + rot6d + traj)
#   5. render_videos   : render BVH+character into mp4 (Blender)
#   6. extract_frames  : ffmpeg mp4 -> 30 fps PNG frames
#   7. remesh          : voxel-remesh animated meshes per frame (Blender)
#                        REQUIRES per-character animated mesh npz in $ZOO_ROOT/anim_meshes/
#   8. rotate_meshes   : rotate remeshed meshes x12
#   9. normalize       : per-species bbox normalization
#  10. preprocess_data : VAE + DINOv2 encode -> npz_train (GPU)
#  11. image_only      : strip latent -> npz_train_image_only
#
# Mesh-free alternative (skips stages 7-11 when you do not have anim_meshes):
#  10b. preprocess_image_only : DINOv2 encode straight from image/ -> npz_train_image_only (GPU)
#
# Usage:
#   bash preprocess/run_pipeline.sh                              # run everything
#   STAGES="rotate_meshes,normalize,preprocess_data" bash preprocess/run_pipeline.sh
#   DATA_ROOT=Truebone_Z-OO ZOO_ROOT=zoo BLENDER=/path/to/blender bash preprocess/run_pipeline.sh
#
#   # Mesh-free path (no anim_meshes available):
#   STAGES="move_fbx,extract_char,rotate_bvh,extract_pose,render_videos,extract_frames,preprocess_image_only" \
#       bash preprocess/run_pipeline.sh

set -euo pipefail

# Run from repo root regardless of where the script was invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-Truebone_Z-OO}"
ZOO_ROOT="${ZOO_ROOT:-zoo}"
BLENDER="${BLENDER:-blender}"
PYTHON="${PYTHON:-python}"
STAGES="${STAGES:-all}"

should_run() {
    [[ "$STAGES" == "all" || ",$STAGES," == *",$1,"* ]]
}

header() {
    echo
    echo "============================================================"
    echo "[$1] $2"
    echo "============================================================"
}

if should_run move_fbx; then
    header "1/11" "move_fbx"
    if [[ ! -d "$DATA_ROOT" ]]; then
        echo "[ERROR] $DATA_ROOT does not exist; place Truebone FBX folders there first."
        exit 1
    fi
    (cd "$DATA_ROOT" && bash "$REPO_ROOT/preprocess/move_fbx.sh")
fi

if should_run extract_char; then
    header "2/11" "extract_character_from_fbx (Blender)"
    "$BLENDER" --background --python preprocess/extract_character_from_fbx.py
fi

if should_run rotate_bvh; then
    header "3/11" "rotate_bvh_parallel"
    "$PYTHON" preprocess/rotate_bvh_parallel.py
fi

if should_run extract_pose; then
    header "4/11" "extract_bvh_pose"
    "$PYTHON" preprocess/extract_bvh_pose.py
fi

if should_run render_videos; then
    header "5/11" "render_bvh_videos (Blender)"
    "$PYTHON" preprocess/render_bvh_videos.py
fi

if should_run extract_frames; then
    header "6/11" "video_to_images (ffmpeg)"
    "$PYTHON" preprocess/video_to_images.py
fi

if should_run remesh; then
    header "7/11" "remesh_meshes (Blender)"
    if [[ ! -d "$ZOO_ROOT/anim_meshes" ]]; then
        echo "[ERROR] $ZOO_ROOT/anim_meshes does not exist."
        echo "        Produce per-character animated mesh npz files (vertices, faces) there first."
        exit 1
    fi
    "$BLENDER" --background --python preprocess/remesh_meshes.py -- \
        --input_root "$ZOO_ROOT/anim_meshes" \
        --output_root "$ZOO_ROOT/remesh_npz"
fi

if should_run rotate_meshes; then
    header "8/11" "rotate_meshes"
    "$PYTHON" preprocess/rotate_meshes.py \
        --input_root "$ZOO_ROOT/remesh_npz" \
        --output_root "$ZOO_ROOT/npz_remesh"
fi

if should_run normalize; then
    header "9/11" "normalize_meshes"
    "$PYTHON" preprocess/normalize_meshes.py \
        --input_root "$ZOO_ROOT/npz_remesh" \
        --pose_root "$ZOO_ROOT/bvh_pose" \
        --output_root "$ZOO_ROOT/npz_mesh_normed"
fi

if should_run preprocess_data; then
    header "10/11" "preprocess_data (VAE + DINOv2, GPU)"
    "$PYTHON" preprocess/preprocess_data.py --dataset_root "$ZOO_ROOT"
fi

if should_run image_only; then
    header "11/11" "extract_image_only"
    "$PYTHON" preprocess/extract_image_only.py \
        --input_root "$ZOO_ROOT/npz_train" \
        --output_root "$ZOO_ROOT/npz_train_image_only"
fi

if should_run preprocess_image_only; then
    header "10b" "preprocess_image_only (DINOv2 only, GPU)"
    "$PYTHON" preprocess/preprocess_image_only.py --dataset_root "$ZOO_ROOT"
fi

echo
echo "Pipeline complete."

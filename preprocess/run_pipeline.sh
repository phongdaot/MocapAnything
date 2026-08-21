#!/usr/bin/env bash
# Run the full MocapAnything preprocessing pipeline end-to-end.
#
# Stages (in order):
#   1. move_fbx        : flatten raw FBX files into character folders
#   2. fix_fbx         : drop no-weight leaf bones (3ds Max Biped `*Nub`) (Blender)
#   3. extract_char    : base mesh + skinning + rest pose + per-motion BVH (Blender)
#   4. align_faces     : rotate characters and motions to face +Z
#   5. rotate_bvh      : rotate every motion BVH around Y axis x12
#   6. extract_pose    : per-BVH pose npz (positions + rot6d + traj)
#   7. species_info    : skeleton topology + joint-name T5 embeddings + static joints
#   8. scale_cache     : per-species normalization scale
#   9. render_videos   : render BVH+character into mp4 (Blender)
#  10. extract_frames  : ffmpeg mp4 -> 30 fps PNG frames
#  11. remesh          : voxel-remesh animated meshes per frame (Blender)
#                        REQUIRES per-character animated mesh npz in $ZOO_ROOT/anim_meshes/
#  12. rotate_meshes   : rotate remeshed meshes x12
#  13. normalize       : per-species bbox normalization
#  14. preprocess_data : VAE + DINOv2 encode -> npz_train (GPU)
#  15. image_only      : strip latent -> npz_train_image_only
#
# Mesh-free alternative (skips stages 11-15 when you do not have anim_meshes):
#  14b. preprocess_image_only : DINOv2 encode straight from image/ -> npz_train_image_only (GPU)
#
# Stages 7 and 8 produce the two files inference refuses to run without:
# species_info_dict.npy and cache/__mesh2pose1002_species_scale_cache.pkl.
#
# Usage:
#   bash preprocess/run_pipeline.sh                              # run everything
#   STAGES="rotate_meshes,normalize,preprocess_data" bash preprocess/run_pipeline.sh
#   DATA_ROOT=Truebone_Z-OO ZOO_ROOT=zoo BLENDER=/path/to/blender bash preprocess/run_pipeline.sh
#
#   # Mesh-free path (no anim_meshes available) — this is the path a custom rig takes:
#   STAGES="move_fbx,fix_fbx,extract_char,align_faces,rotate_bvh,extract_pose,species_info,scale_cache,render_videos,extract_frames,preprocess_image_only" \
#       bash preprocess/run_pipeline.sh
#
# See examples/custom_rig/README.md for bringing your own rigged FBX through this.

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

FBX_ROOT="${FBX_ROOT:-$DATA_ROOT}"      # what extract_char reads; fix_fbx redirects it

if should_run move_fbx; then
    header "1/15" "move_fbx"
    if [[ ! -d "$DATA_ROOT" ]]; then
        echo "[ERROR] $DATA_ROOT does not exist; place your character FBX folders there first."
        exit 1
    fi
    (cd "$DATA_ROOT" && bash "$REPO_ROOT/preprocess/move_fbx.sh")
fi

if should_run fix_fbx; then
    header "2/15" "fix_fbx_batch (Blender)"
    # 3ds Max Biped rigs carry `*Nub` leaf bones with no skin weights. They must go,
    # or every downstream joint count differs from the released dataset.
    "$PYTHON" preprocess/fix_fbx_batch.py \
        --input_root "$DATA_ROOT" \
        --output_root "$ZOO_ROOT/fixed_fbx" \
        --blender "$BLENDER" \
        --skip_existing
    FBX_ROOT="$ZOO_ROOT/fixed_fbx"
fi

if should_run extract_char; then
    header "3/15" "extract_character_from_fbx (Blender)"
    # PYTHON is passed through: the face-forward pass needs torch, which Blender's
    # bundled interpreter does not have, so that step shells out.
    DATA_ROOT="$FBX_ROOT" PYTHON="$PYTHON" \
        "$BLENDER" --background --python preprocess/extract_character_from_fbx.py
fi

if should_run align_faces; then
    header "4/15" "align_character_face_zplus"
    "$PYTHON" preprocess/align_character_face_zplus.py \
        --input_root "$ZOO_ROOT/characters_fix_facezplus" \
        --output_root "$ZOO_ROOT/characters_face_zplus" \
        --motion_root "$ZOO_ROOT/motions" \
        --motion_output_root "$ZOO_ROOT/motions_face_zplus" \
        ${ALIGN_REF_DIR:+--ref_dir "$ALIGN_REF_DIR"}
fi

if should_run rotate_bvh; then
    header "5/15" "rotate_bvh_parallel"
    # Picks up motions_face_zplus/ automatically when align_faces has run.
    "$PYTHON" preprocess/rotate_bvh_parallel.py
fi

if should_run extract_pose; then
    header "6/15" "extract_bvh_pose"
    "$PYTHON" preprocess/extract_bvh_pose.py
fi

if should_run species_info; then
    header "7/15" "build_species_info (T5)"
    "$PYTHON" preprocess/build_species_info.py --dataset_root "$ZOO_ROOT"
fi

if should_run scale_cache; then
    header "8/15" "build_scale_cache"
    "$PYTHON" preprocess/build_scale_cache.py \
        --bvh_pose_root "$ZOO_ROOT/bvh_pose" \
        --output "$ZOO_ROOT/cache/__mesh2pose1002_species_scale_cache.pkl"
fi

if should_run render_videos; then
    header "9/15" "render_bvh_videos_fast (Blender)"
    BLENDER="$BLENDER" "$PYTHON" preprocess/render_bvh_videos_fast.py \
        --zoo-root "$ZOO_ROOT" \
        --character-root "$ZOO_ROOT/characters_face_zplus"
fi

if should_run extract_frames; then
    header "10/15" "video_to_images (ffmpeg)"
    "$PYTHON" preprocess/video_to_images.py
fi

if should_run remesh; then
    header "11/15" "remesh_meshes (Blender)"
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
    header "12/15" "rotate_meshes"
    "$PYTHON" preprocess/rotate_meshes.py \
        --input_root "$ZOO_ROOT/remesh_npz" \
        --output_root "$ZOO_ROOT/npz_remesh"
fi

if should_run normalize; then
    header "13/15" "normalize_meshes"
    "$PYTHON" preprocess/normalize_meshes.py \
        --input_root "$ZOO_ROOT/npz_remesh" \
        --pose_root "$ZOO_ROOT/bvh_pose" \
        --output_root "$ZOO_ROOT/npz_mesh_normed"
fi

if should_run preprocess_data; then
    header "14/15" "preprocess_data (VAE + DINOv2, GPU)"
    "$PYTHON" preprocess/preprocess_data.py --dataset_root "$ZOO_ROOT"
fi

if should_run image_only; then
    header "15/15" "extract_image_only"
    "$PYTHON" preprocess/extract_image_only.py \
        --input_root "$ZOO_ROOT/npz_train" \
        --output_root "$ZOO_ROOT/npz_train_image_only"
fi

if should_run preprocess_image_only; then
    header "14b" "preprocess_image_only (DINOv2 only, GPU)"
    "$PYTHON" preprocess/preprocess_image_only.py --dataset_root "$ZOO_ROOT"
fi

echo
echo "Pipeline complete."

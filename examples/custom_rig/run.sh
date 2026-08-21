#!/usr/bin/env bash
# Preprocess your own rigged FBX into everything inference needs.
#
# This is the mesh-free path through preprocess/run_pipeline.sh: stages 1-10
# plus 14b, skipping the four stages that require per-character animated meshes.
# See examples/custom_rig/README.md for the full walkthrough.
#
# Usage:
#   export BLENDER=/path/to/blender
#   export PYTHON=/path/to/python          # the torch env, NOT Blender's bundled one
#   bash examples/custom_rig/run.sh MyRig
#
# Environment:
#   BLENDER        Blender 3.6+ binary                        (default: blender)
#   PYTHON         python with torch                          (default: python)
#   DATA_ROOT      where your <RIG>/ FBX folder lives         (default: Truebone_Z-OO)
#   ZOO_ROOT       where processed data is written            (default: zoo)
#   VIDEO_ROOT     where <RIG>#clip.mp4 lives                 (default: videos)
#   ALIGN_REF_DIR  optional dir holding <RIG>/front.npy to override the
#                  inferred facing direction

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RIG="${1:-}"
if [[ -z "$RIG" ]]; then
    echo "Usage: bash examples/custom_rig/run.sh <RIG_NAME>"
    echo "  where \$DATA_ROOT/<RIG_NAME>/ holds your FBX files."
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-Truebone_Z-OO}"
ZOO_ROOT="${ZOO_ROOT:-zoo}"
VIDEO_ROOT="${VIDEO_ROOT:-videos}"
BLENDER="${BLENDER:-blender}"
PYTHON="${PYTHON:-python}"

# ---- checks up front: every one of these fails silently or confusingly later ----

if [[ ! -d "$DATA_ROOT/$RIG" ]]; then
    echo "[ERROR] $DATA_ROOT/$RIG does not exist."
    echo "        Put your FBX files there, e.g. $DATA_ROOT/$RIG/$RIG.fbx"
    exit 1
fi

# No -type f here: the FBX may well be a symlink into a read-only dataset mount.
if [[ -z "$(find -L "$DATA_ROOT/$RIG" -iname '*.fbx' -print -quit)" ]]; then
    echo "[ERROR] No .fbx found under $DATA_ROOT/$RIG."
    exit 1
fi

if ! compgen -G "$VIDEO_ROOT/$RIG#*" >/dev/null; then
    echo "[ERROR] No input clip matching $VIDEO_ROOT/$RIG#* ."
    echo "        Inference resolves the reference skeleton from the part before"
    echo "        '#', so the clip must be named e.g. $VIDEO_ROOT/$RIG#clip.mp4"
    exit 1
fi

if ! command -v "$BLENDER" >/dev/null 2>&1 && [[ ! -x "$BLENDER" ]]; then
    echo "[ERROR] Blender not found at '$BLENDER'. Set BLENDER=/path/to/blender."
    exit 1
fi

# The pipeline walks every folder in DATA_ROOT, not just this one. That is fine,
# but say so rather than let it look like a hang.
OTHER_RIGS=$(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name "$RIG" | wc -l)
if [[ "$OTHER_RIGS" -gt 0 ]]; then
    echo "[NOTE] $DATA_ROOT holds $OTHER_RIGS other character folder(s); they will be"
    echo "       processed too. Point DATA_ROOT at a folder containing only $RIG to"
    echo "       process just this rig."
fi

echo "==> rig=$RIG  data=$DATA_ROOT  out=$ZOO_ROOT  video=$VIDEO_ROOT"

# ---- preprocessing, part 1: FBX -> aligned character + motions ----

STAGES="move_fbx,fix_fbx,extract_char,align_faces" \
DATA_ROOT="$DATA_ROOT" ZOO_ROOT="$ZOO_ROOT" BLENDER="$BLENDER" PYTHON="$PYTHON" \
ALIGN_REF_DIR="${ALIGN_REF_DIR:-}" \
    bash preprocess/run_pipeline.sh

# Gate on alignment before spending time on 12 rotations and a render. When the
# facing direction cannot be inferred, align_character_face_zplus.py logs the
# failure and moves on — everything downstream then silently uses the unaligned
# motions and the result looks subtly wrong rather than failing.
if [[ ! -f "$ZOO_ROOT/characters_face_zplus/$RIG/front.npy" ]]; then
    echo
    echo "[ERROR] Stage 4 could not align $RIG to +Z:"
    echo "        $ZOO_ROOT/characters_face_zplus/$RIG/front.npy was not written."
    if [[ -f "$ZOO_ROOT/characters_face_zplus/_align_failed.tsv" ]]; then
        sed 's/^/        /' "$ZOO_ROOT/characters_face_zplus/_align_failed.tsv"
    fi
    cat <<EOF

        Supply the facing direction yourself — a unit vector in the rig's own
        coordinates pointing out of the character's front — and rerun:

          mkdir -p examples/custom_rig/align_ref/$RIG
          $PYTHON -c "import numpy as np; np.save('examples/custom_rig/align_ref/$RIG/front.npy', np.array([1.,0.,0.]))"
          ALIGN_REF_DIR=examples/custom_rig/align_ref bash examples/custom_rig/run.sh $RIG

        [1,0,0] means the character faces +X; [0,0,1] means it already faces +Z.
EOF
    exit 1
fi

# ---- preprocessing, part 2: rotations, poses, species info, render, embeddings ----

STAGES="rotate_bvh,extract_pose,species_info,scale_cache,render_videos,extract_frames,preprocess_image_only" \
DATA_ROOT="$DATA_ROOT" ZOO_ROOT="$ZOO_ROOT" BLENDER="$BLENDER" PYTHON="$PYTHON" \
    bash preprocess/run_pipeline.sh

# ---- did it actually produce the two files inference refuses to start without? ----

SPECIES_INFO="$ZOO_ROOT/species_info_dict.npy"
SCALE_CACHE="$ZOO_ROOT/cache/__mesh2pose1002_species_scale_cache.pkl"
missing=0
for f in "$SPECIES_INFO" "$SCALE_CACHE"; do
    [[ -f "$f" ]] || { echo "[ERROR] missing: $f"; missing=1; }
done
[[ "$missing" -eq 0 ]] || exit 1

"$PYTHON" - "$SPECIES_INFO" "$RIG" <<'PY'
import sys
import numpy as np

info = np.load(sys.argv[1], allow_pickle=True).item()
rig = sys.argv[2]
if rig not in info:
    print(f"[ERROR] '{rig}' not in species_info_dict.npy (have: {sorted(info)[:10]}...)")
    sys.exit(1)
entry = info[rig]
print(f"[OK] {rig}: {len(entry['joints_name'])} joints, "
      f"t5_embedding {np.asarray(entry['t5_embedding']).shape}")
PY

cat <<EOF

Preprocessing done. Now run inference:

  export BLENDER_BIN="$BLENDER"
  $PYTHON -m inference.video2pose2rot --config examples/custom_rig/inference.yaml

Results land in ./outputs/custom_rig/.
EOF

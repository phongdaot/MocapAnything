# Preprocess

End-to-end data preparation pipeline that turns a raw Truebone FBX dump into the latent + image-embed npz files consumed by training.

## Quick start

```bash
# Place the Truebone FBX folders here first (one folder per character).
ls Truebone_Z-OO/
# Alligator/  Anaconda/  Ant/  ...

# Run everything:
bash preprocess/run_pipeline.sh

# Or pick stages:
STAGES="rotate_meshes,normalize,preprocess_data" bash preprocess/run_pipeline.sh

# Override paths / binaries:
DATA_ROOT=Truebone_Z-OO ZOO_ROOT=zoo BLENDER=/opt/blender/blender PYTHON=python3 \
    bash preprocess/run_pipeline.sh
```

## Prerequisites

- **Python** with `numpy`, `torch`, `scipy`, `tqdm`, `Pillow`, `matplotlib`, `trimesh`, `huggingface_hub`, `transformers`.
- **Blender** (3.6+ / 4.x) on `PATH` or pointed to by `$BLENDER`. Used by stages 2, 5, 7.
- **ffmpeg** on `PATH`. Used by stage 6.
- **GPU + model checkpoints** for stage 10:
  - `checkpoints/TripoSG/` — TripoSG VAE + DINOv2 pipeline weights
  - `checkpoints/RMBG-1.4/` — Bria background-removal weights
- **Repository setup**: `TripoSG/` and `models/v1/video2mesh/pipeline_triposg.py` importable from the repo root.

## Stages

| # | Stage | Script | Tool | Input | Output |
|---|---|---|---|---|---|
| 1 | `move_fbx` | `move_fbx.sh` | bash | `Truebone_Z-OO/*/.../*.fbx` | flattened `Truebone_Z-OO/{character}/*.fbx` |
| 2 | `extract_char` | `extract_character_from_fbx.py` | Blender | `Truebone_Z-OO/{character}/*.fbx` | `zoo/characters_fix_facezplus/{character}/{base_mesh.obj, blender_vertices.npy, skinning_weights.npy, rest.bvh, textures}` and `zoo/motions/{character}#{anim}.bvh` |
| 3 | `rotate_bvh` | `rotate_bvh_parallel.py` | python | `zoo/motions/*.bvh` | `zoo/bvh/{motion}/y{deg}.bvh` (12 angles) |
| 4 | `extract_pose` | `extract_bvh_pose.py` | python | `zoo/bvh/{motion}/y{deg}.bvh` | `zoo/bvh_pose/{motion}/y{deg}.npz` (positions, traj, rot6d, scale, frametime) |
| 5 | `render_videos` | `render_bvh_videos.py` | Blender | `zoo/bvh/*/*.bvh` + `zoo/characters_fix_facezplus/{character}` | `zoo/video/{motion}/y{deg}.mp4` |
| 6 | `extract_frames` | `video_to_images.py` | ffmpeg | `zoo/video/{motion}/y{deg}.mp4` | `zoo/image/{motion}/y{deg}/{00000.png, ...}` at 30 fps |
| 7 | `remesh` | `remesh_meshes.py` | Blender | `zoo/anim_meshes/{character}.npz` (vertices, faces) **— see note below** | `zoo/remesh_npz/{character}.npz` (vertices, normals) |
| 8 | `rotate_meshes` | `rotate_meshes.py` | python | `zoo/remesh_npz/{character}.npz` | `zoo/npz_remesh/{motion}/y{deg}.npz` |
| 9 | `normalize` | `normalize_meshes.py` | python | `zoo/npz_remesh/` + `zoo/bvh_pose/` | `zoo/npz_mesh_normed/{motion}/y{deg}.npz` (per-species bbox normalization, key `vertices_normed`) |
| 10 | `preprocess_data` | `preprocess_data.py` | python + GPU | `zoo/npz_mesh_normed/` + `zoo/image/` | `zoo/npz_train/{motion}/y{deg}.npz` (`latent`, `image_embed`) |
| 11 | `image_only` | `extract_image_only.py` | python | `zoo/npz_train/` | `zoo/npz_train_image_only/{motion}/y{deg}.npz` (`image_embed` only) |

### Mesh-free alternative

If you do not have `zoo/anim_meshes/` (stage 7 input), skip stages 7-11 and run this stage instead. It produces the same `npz_train_image_only/` artifact directly from `zoo/image/`, which is enough for the `video2pose` / `pose2rot` / `video2pose2rot` training pipelines (only `video2mesh` and `mesh2pose` truly require the mesh latent).

| # | Stage | Script | Tool | Input | Output |
|---|---|---|---|---|---|
| 10b | `preprocess_image_only` | `preprocess_image_only.py` | python + GPU | `zoo/image/` | `zoo/npz_train_image_only/{motion}/y{deg}.npz` (`image_embed` only) |

```bash
STAGES="move_fbx,extract_char,rotate_bvh,extract_pose,render_videos,extract_frames,preprocess_image_only" \
    bash preprocess/run_pipeline.sh
```

### Fast BVH video renderer

`render_bvh_videos_fast.py` is a drop-in faster renderer for stage 5. It keeps the same dataset layout as the original renderer:

```text
zoo/video/{motion}/y{deg}.mp4
```

Example:

```bash
BLENDER=/opt/blender/blender python preprocess/render_bvh_videos_fast.py \
    --zoo-root zoo \
    --workers 4 \
    --views y0,y30,y60,y90,y120,y150,y180,y210,y240,y270,y300,y330
```

The speedup comes from four changes:

- EEVEE is used by default instead of the original Cycles render settings.
- Multiple BVH views are rendered in batches inside long-lived Blender processes.
- The character mesh and materials are imported once per Blender worker, then only vertex positions are updated per frame.
- Frames are rendered directly as the requested image format before ffmpeg encodes the final mp4.

To compare against the original renderer on a small Truebones subset:

```bash
BLENDER=/opt/blender/blender python preprocess/benchmark_render_bvh_videos.py \
    --zoo-root zoo \
    --benchmark-root compare_runs/render_benchmark_10 \
    --motions 10 \
    --view y0 \
    --max-frames 100 \
    --fast-workers 4
```

## Output tree

```
zoo/
├── characters_fix_facezplus/
│   └── {Character}/
│       ├── base_mesh.obj / .mtl / textures
│       ├── blender_vertices.npy
│       ├── skinning_weights.npy
│       └── rest.bvh
├── motions/{character}#{anim}.bvh
├── bvh/{motion}/y{deg}.bvh                # 12 rotations
├── bvh_pose/{motion}/y{deg}.npz           # positions, traj, rot6d, scale
├── video/{motion}/y{deg}.mp4
├── image/{motion}/y{deg}/00000.png ...
├── anim_meshes/{character}.npz            # YOU PROVIDE: vertices (T,N,3) + faces
├── remesh_npz/{character}.npz             # voxel-remeshed + sampled
├── npz_remesh/{motion}/y{deg}.npz         # 12 Y-axis rotations
├── npz_mesh_normed/{motion}/y{deg}.npz    # vertices_normed, bbox_center, global_scale
├── npz_train/{motion}/y{deg}.npz          # latent + image_embed
└── npz_train_image_only/{motion}/y{deg}.npz
```

## Configuration

The runner reads these env vars (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `DATA_ROOT` | `Truebone_Z-OO` | folder of raw character FBX folders |
| `ZOO_ROOT` | `zoo` | output root for every intermediate + final artifact |
| `BLENDER` | `blender` | Blender executable |
| `PYTHON` | `python` | Python interpreter |
| `STAGES` | `all` | comma-separated stage names to run |

Individual scripts also accept `--input_root` / `--output_root` / `--num_workers` etc. — see each script's `argparse` block.

## Notes & caveats

- **Stage 7 (`remesh`) needs `zoo/anim_meshes/{character}.npz`** containing per-frame deformed vertices (`vertices` shape `(T, N, 3)`) and shared `faces` (`(F, 3)`). The pipeline does **not** generate this — it's the LBS / skinning step that applies the per-frame BVH pose to `base_mesh.obj` using `skinning_weights.npy`. Drop your own script in, or ping me to add one. The runner will fail with a clear error if this folder is missing.
- **Per-species normalization**: stage 9 collects every motion of a species (`Alligator#*`) and computes one shared bbox center + global scale, so meshes stay comparable across motions. It depends on stage 4's `bvh_pose` output for the root trajectory.
- **Sharded GPU run** for stage 10: launch the same command on N GPUs with `--shard 0..N-1 --num_shards N` to split the workload.

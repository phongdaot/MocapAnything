# Preprocess

End-to-end data preparation pipeline that turns a raw FBX dump into everything downstream needs: the `species_info_dict.npy` + scale cache that inference requires, and the latent + image-embed npz files consumed by training.

It is not Truebones-specific — see [Bringing your own rig](#bringing-your-own-rig).

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

- **Python** with `numpy`, `torch`, `scipy`, `tqdm`, `Pillow`, `matplotlib`, `trimesh`, `huggingface_hub`, `transformers`, `sentencepiece`.
- **Blender** (3.6+ / 4.x) on `PATH` or pointed to by `$BLENDER`. Used by stages 2, 3, 9, 11.
- **ffmpeg** on `PATH`. Used by stage 10.
- **Network access on the first run of stage 7**, which downloads `t5-base` to embed joint names. Point `--t5_model` at a local copy to run offline.
- **GPU + model checkpoints** for stage 14:
  - `checkpoints/TripoSG/` — TripoSG VAE + DINOv2 pipeline weights
  - `checkpoints/RMBG-1.4/` — Bria background-removal weights (or let it fall back to `briaai/RMBG-1.4`)
- **Repository setup** for stage 14: `TripoSG/` and `models/v1/video2mesh/pipeline_triposg.py` importable from the repo root.

Stage 14b needs **neither** — it only uses the DINOv2 half of the pipeline, and loads
`facebook/dinov2-large` directly when TripoSG is unavailable. The mesh-free path is therefore
TripoSG-free end to end.

`$PYTHON` must be the torch environment, *not* Blender's bundled interpreter. Stage 3 runs inside Blender but shells out to `$PYTHON` for the face-forward pass, which needs torch.

## Stages

| # | Stage | Script | Tool | Input | Output |
|---|---|---|---|---|---|
| 1 | `move_fbx` | `move_fbx.sh` | bash | `Truebone_Z-OO/*/.../*.fbx` | flattened `Truebone_Z-OO/{character}/*.fbx` |
| 2 | `fix_fbx` | `fix_fbx_batch.py` → `fix_fbx.py` | Blender | `Truebone_Z-OO/{character}/*.fbx` | `zoo/fixed_fbx/{character}/*.fbx` with no-weight leaf bones removed |
| 3 | `extract_char` | `extract_character_from_fbx.py` | Blender | `zoo/fixed_fbx/{character}/*.fbx` | `zoo/characters_fix_facezplus/{character}/{base_mesh.obj, blender_vertices.npy, skinning_weights.npy, rest.bvh, {character}_ffs.bvh, textures}` and `zoo/motions/{character}#{anim}.bvh` |
| 4 | `align_faces` | `align_character_face_zplus.py` | python | `zoo/characters_fix_facezplus/` + `zoo/motions/` | `zoo/characters_face_zplus/{character}/` (rest pose + `{character}_ffs.bvh` + `front.npy`) and `zoo/motions_face_zplus/*.bvh`, all facing +Z |
| 5 | `rotate_bvh` | `rotate_bvh_parallel.py` | python | `zoo/motions_face_zplus/*.bvh` | `zoo/bvh/{motion}/y{deg}.bvh` (12 angles) |
| 6 | `extract_pose` | `extract_bvh_pose.py` | python | `zoo/bvh/{motion}/y{deg}.bvh` | `zoo/bvh_pose/{motion}/y{deg}.npz` (positions, traj, rot6d, scale, frametime) |
| 7 | `species_info` | `build_species_info.py` | python (T5) | `zoo/bvh/` + `zoo/bvh_pose/` | `zoo/species_info_dict.npy` (joint names, topology, joint-name T5 embeddings, static joints) |
| 8 | `scale_cache` | `build_scale_cache.py` | python | `zoo/bvh_pose/` | `zoo/cache/__mesh2pose1002_species_scale_cache.pkl` |
| 9 | `render_videos` | `render_bvh_videos_fast.py` | Blender | `zoo/bvh/*/*.bvh` + `zoo/characters_face_zplus/{character}` | `zoo/video/{motion}/y{deg}.mp4` |
| 10 | `extract_frames` | `video_to_images.py` | ffmpeg | `zoo/video/{motion}/y{deg}.mp4` | `zoo/image/{motion}/y{deg}/{00000.png, ...}` at 30 fps |
| 11 | `remesh` | `remesh_meshes.py` | Blender | `zoo/anim_meshes/{character}.npz` (vertices, faces) **— see note below** | `zoo/remesh_npz/{character}.npz` (vertices, normals) |
| 12 | `rotate_meshes` | `rotate_meshes.py` | python | `zoo/remesh_npz/{character}.npz` | `zoo/npz_remesh/{motion}/y{deg}.npz` |
| 13 | `normalize` | `normalize_meshes.py` | python | `zoo/npz_remesh/` + `zoo/bvh_pose/` | `zoo/npz_mesh_normed/{motion}/y{deg}.npz` (per-species bbox normalization, key `vertices_normed`) |
| 14 | `preprocess_data` | `preprocess_data.py` | python + GPU | `zoo/npz_mesh_normed/` + `zoo/image/` | `zoo/npz_train/{motion}/y{deg}.npz` (`latent`, `image_embed`) |
| 15 | `image_only` | `extract_image_only.py` | python | `zoo/npz_train/` | `zoo/npz_train_image_only/{motion}/y{deg}.npz` (`image_embed` only) |

**Stages 7 and 8 produce the two files inference refuses to start without**: `species_info_dict.npy` and `cache/__mesh2pose1002_species_scale_cache.pkl`. Every inference entry point looks up the species by the prefix of the clip name and raises `KeyError` if it is not in `species_info_dict.npy`.

### Mesh-free alternative

If you do not have `zoo/anim_meshes/` (stage 11 input), skip stages 11-15 and run this stage instead. It produces the same `npz_train_image_only/` artifact directly from `zoo/image/`, which is enough for the `video2pose` / `pose2rot` / `video2pose2rot` pipelines (only `video2mesh` and `mesh2pose` truly require the mesh latent).

| # | Stage | Script | Tool | Input | Output |
|---|---|---|---|---|---|
| 14b | `preprocess_image_only` | `preprocess_image_only.py` | python + GPU | `zoo/image/` | `zoo/npz_train_image_only/{motion}/y{deg}.npz` (`image_embed` only) |

```bash
STAGES="move_fbx,fix_fbx,extract_char,align_faces,rotate_bvh,extract_pose,species_info,scale_cache,render_videos,extract_frames,preprocess_image_only" \
    bash preprocess/run_pipeline.sh
```

### Fast BVH video renderer

`render_bvh_videos_fast.py` is what stage 9 runs; `render_bvh_videos.py` is the original, slower renderer. Both keep the same dataset layout:

```text
zoo/video/{motion}/y{deg}.mp4
```

Example:

```bash
BLENDER=/opt/blender/blender python preprocess/render_bvh_videos_fast.py \
    --zoo-root zoo \
    --character-root zoo/characters_face_zplus \
    --workers 4 \
    --views y0,y30,y60,y90,y120,y150,y180,y210,y240,y270,y300,y330
```

`--character-root` defaults to `zoo/characters_fix_facezplus`. Pass `characters_face_zplus`
(as stage 9 does) once stage 4 has run, or the rendered mesh will not match the BVH orientation.

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
├── fixed_fbx/{Character}/*.fbx            # no-weight leaf bones removed
├── characters_fix_facezplus/
│   └── {Character}/
│       ├── base_mesh.obj / .mtl / textures
│       ├── blender_vertices.npy
│       ├── skinning_weights.npy
│       ├── rest.bvh
│       └── {Character}_ffs.bvh            # face-forward, scale 0.01 — mesh export template
├── characters_face_zplus/{Character}/     # same contents, rotated to face +Z, plus front.npy
├── motions/{character}#{anim}.bvh
├── motions_face_zplus/{character}#{anim}.bvh
├── bvh/{motion}/y{deg}.bvh                # 12 rotations
├── bvh_pose/{motion}/y{deg}.npz           # positions, traj, rot6d, scale
├── species_info_dict.npy                  # REQUIRED BY INFERENCE — topology + joint-name T5 embeddings
├── joint_name_map.json                    # optional input to stage 7, see below
├── cache/
│   └── __mesh2pose1002_species_scale_cache.pkl   # REQUIRED BY INFERENCE
├── video/{motion}/y{deg}.mp4
├── image/{motion}/y{deg}/00000.png ...
├── anim_meshes/{character}.npz            # YOU PROVIDE: vertices (T,N,3) + faces
├── remesh_npz/{character}.npz             # voxel-remeshed + sampled
├── npz_remesh/{motion}/y{deg}.npz         # 12 Y-axis rotations
├── npz_mesh_normed/{motion}/y{deg}.npz    # vertices_normed, bbox_center, global_scale
├── npz_train/{motion}/y{deg}.npz          # latent + image_embed
└── npz_train_image_only/{motion}/y{deg}.npz
```

## Bringing your own rig

Everything above works on any rigged FBX, not just the Truebones dump. The short
version — see [`../examples/custom_rig/README.md`](../examples/custom_rig/README.md)
for the walkthrough, including a worked end-to-end example.

```bash
Truebone_Z-OO/MyRig/MyRig.fbx      # base: mesh + skeleton (shortest filename wins)
Truebone_Z-OO/MyRig/MyRig-Walk.fbx # one or more motions
videos/MyRig#clip.mp4              # your footage; the prefix before '#' must match the rig folder

export BLENDER=/path/to/blender PYTHON=/path/to/torch/python
bash examples/custom_rig/run.sh MyRig
python -m inference.video2pose2rot --config examples/custom_rig/inference.yaml
```

Two stages guess things about your rig and **neither raises an error when the guess is wrong**:

- **Stage 4** infers which way the character faces. If the mesh comes out rotated in the
  output video while the skeleton looks fine, drop a `front.npy` (unit vector pointing out
  of the character's front, in rig coordinates) into `$ALIGN_REF_DIR/{Character}/` and rerun.
- **Stage 2** deletes leaf bones that carry no skin weights, because that is what the
  released datasets were built with (3ds Max Biped `*Nub` bones). If your rig has meaningful
  weightless leaves, drop `fix_fbx` from `STAGES`.

### Joint names and `joint_name_map.json`

Stage 7 embeds each joint's *name* with T5, and the model uses those embeddings to reason
about what each joint is. This is the only place the two released datasets differ:

| Dataset | How `rename_clean` is produced |
|---|---|
| `obj1k` | automatic cleanup (`mixamorig:LeftUpLeg` → `Left Up Leg`) — no map file needed |
| `zoo1030` | curated canonical names (`quadruped spine 01`, `flyer tail 01`) supplied via `joint_name_map.json` |

Without a map, stage 7 falls back to automatic cleanup and prints a warning. That is correct
and self-consistent for a new rig, but the resulting embeddings will not match the curated
`zoo1030` convention. To use curated names, hand-write the mapping — the file is
`{"MyRig": {"<bone name in the FBX>": "<readable name>", ...}}` — and pass it in:

```bash
python preprocess/build_species_info.py --dataset_root zoo --joint_name_map path/to/joint_name_map.json
```

Note that trailing digits are stripped during automatic cleanup, so `Tail_01 … Tail_07`
collapse to a single `Tail` embedding. Spell ordinals out if they matter for your rig.

To dump the mapping out of an existing `species_info_dict.npy` (e.g. to see the convention
the released data uses):

```bash
python preprocess/build_species_info.py --export_map_from datasets/zoo1030/species_info_dict.npy --output joint_name_map.json
```

To check a rebuild against released data without writing anything:

```bash
python preprocess/build_species_info.py --dataset_root datasets/zoo1030 \
    --joint_name_map joint_name_map.json \
    --verify_against datasets/zoo1030/species_info_dict.npy
```

## Configuration

The runner reads these env vars (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `DATA_ROOT` | `Truebone_Z-OO` | folder of raw character FBX folders |
| `ZOO_ROOT` | `zoo` | output root for every intermediate + final artifact |
| `BLENDER` | `blender` | Blender executable |
| `PYTHON` | `python` | Python interpreter (must have torch) |
| `STAGES` | `all` | comma-separated stage names to run |
| `ALIGN_REF_DIR` | *(unset)* | folder of `{Character}/front.npy` overrides for stage 4 |
| `BVH_SRC` | *(auto)* | stage 5 source; defaults to `motions_face_zplus/` when it exists, else `motions/` |

Individual scripts also accept `--input_root` / `--output_root` / `--num_workers` etc. — see each script's `argparse` block.

## Notes & caveats

- **Stage 11 (`remesh`) needs `zoo/anim_meshes/{character}.npz`** containing per-frame deformed vertices (`vertices` shape `(T, N, 3)`) and shared `faces` (`(F, 3)`). The pipeline does **not** generate this — it's the LBS / skinning step that applies the per-frame BVH pose to `base_mesh.obj` using `skinning_weights.npy`. Drop your own script in, or ping me to add one. The runner will fail with a clear error if this folder is missing.
- **Per-species normalization**: stage 13 collects every motion of a species (`Alligator#*`) and computes one shared bbox center + global scale, so meshes stay comparable across motions. It depends on stage 6's `bvh_pose` output for the root trajectory.
- **Sharded GPU run** for stage 14: launch the same command on N GPUs with `--shard 0..N-1 --num_shards N` to split the workload.
- **`{Character}_ffs.bvh` is not optional.** It is the rest-pose template `utils/npy2bvh.py` writes predicted rotations against, at scale 0.01. Stage 3 writes it and stage 4 rewrites it from the aligned rest pose. A stale or hand-copied one produces a collapsed or 90°-rotated mesh in the output video with no error message.
- **Stage 7 keeps only the majority skeleton** when one species has several rig variants across its motions, matching how the released `species_info_dict.npy` was built.

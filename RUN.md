# Install & Run

This document covers environment setup, dataset layout, training, and inference for MoCapAnything V2. For a high-level overview of the project, see the [README](README.md).

A deep-learning pipeline for animal motion capture from video. The pipeline reconstructs 3D mesh, pose, and joint rotations for arbitrary animal species starting from image sequences, producing animation-ready BVH output that can be rendered in Blender.

## Overview

The pipeline is composed of several trainable stages that can be run independently or chained together end-to-end:

| Stage | Purpose | Input | Output |
| --- | --- | --- | --- |
| `video2mesh` | Per-frame 3D mesh reconstruction (TripoSG-based) | Image sequence | Mesh (`.glb` / latent) |
| `mesh2pose` | Predict 3D joint positions from meshes | Mesh sequence + reference pose | Joint positions |
| `video2pose` | Predict 3D joint positions directly from video | Image sequence + reference pose | Joint positions |
| `pose2rot` | Convert joint positions to joint rotations | Joint positions + rest pose/memory | Joint rotations |
| `video2pose2rot` | Joint end-to-end model (video → pose → rotation) | Image sequence + reference | BVH-ready rotations |

A reference frame (a single pose from a matching species) is used to guide the per-species skeleton and scale, enabling generalization across unseen animals.

## 🚀 Quick Start — run the demo

Clone the repo, grab the weights + demo data from HuggingFace, and you can run the bundled examples (or your own videos) end-to-end — no dataset preprocessing required.

**1. Clone + environment** (Python ≥ 3.10 recommended)
```bash
git clone https://github.com/phongdaot/MocapAnything.git
cd MocapAnything
pip install torch torchvision numpy opencv-python pillow matplotlib scipy scikit-image \
    trimesh roma pyyaml tqdm huggingface_hub transformers gradio imageio-ffmpeg
# TripoSG is only needed for the V1 video2mesh baseline; the V2 demo does not require it.
# (Alternatively `pip install -r requirements.txt` reproduces the exact tested environment.)
```
> **PyTorch/CUDA:** pip's default `torch` wheel targets the newest CUDA and may fail on older drivers with `The NVIDIA driver on your system is too old` (which then surfaces as `torch.cat(): expected a non-empty list of Tensors` during feature extraction). If you hit that, install a build matching your driver from [pytorch.org](https://pytorch.org/get-started/locally/), e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`, or use the tested pin `torch==2.9.0`.

**2. Download weights + demo data from HuggingFace**
```bash
# Weights → ./checkpoints/   (the end-to-end model)
hf download kehong/MoCapAnythingV2-weights --local-dir ./checkpoints
# The background remover (briaai/RMBG-1.4) and DINOv2 (facebook/dinov2-large) auto-download from HuggingFace on first run.

# Demo data (~160 MB: 15 example videos + mini zoo/obj datasets with 1-frame reference features)
hf download kehong/MoCapAnythingV2-data-sample --repo-type dataset --local-dir ./demo/data
```
> The demo ships **1-frame** reference features (only frame 0 of each reference is needed at inference — verified bit-identical to the full features). A full mini-dataset unlocking all **73 species** as retarget targets will be released later; once available, download it into `./datasets/zoo1030/` and it overrides the demo subset by same-name files.

**3. Configure rendering (for 3D mesh output)**

Download a **portable Blender build (4.x / 5.x)** from [blender.org/download](https://www.blender.org/download/) and extract it anywhere — no installation needed. Then point `BLENDER_BIN` to the binary:
```bash
export BLENDER_BIN=/path/to/blender-4.x-linux-x64/blender   # required for the 3D mesh render
# ffmpeg is NOT required system-wide — the pipeline uses the binary bundled with the `imageio-ffmpeg` wheel.
```
On **headless servers** Blender may fail with `error while loading shared libraries: libxkbcommon.so.0` (or similar X11/GL libs). Fix either way:
```bash
sudo apt install -y libxkbcommon0 libgl1 libxi6 libxrender1      # Debian/Ubuntu
# …or, without root: point MOCAP_ENV_LIB at any conda env's lib that ships these
export MOCAP_ENV_LIB=$CONDA_PREFIX/lib
```
Without Blender everything still runs — you get the pose `.npy` + BVH + skeleton videos, just no textured mesh render.

**4. Run the bundled examples (command line)**
```bash
export PYTHONPATH=$PWD:$PWD/TripoSG
# 5 zoo videos (self-driven): predict pose+rotation, render 3D
python inference/video2pose2rot.py --config demo/configs/demo_zoo.yaml
# 5 object videos
python inference/video2pose2rot.py --config demo/configs/demo_obj.yaml
# in-the-wild animal videos (no GT — conditions on a same-species reference skeleton).
# 25 videos ship; the 10 whose species have a reference in the mini demo dataset are
# processed, the rest are skipped with a warning (they unlock with the full dataset).
python inference/video2pose2rot.py --config demo/configs/demo_wild.yaml
```
Each sequence produces, under `demo_outputs/`:
- `*_pose_pred.npy`, `*_rot6d_pred.npy` — predicted joint positions and rotations
- a BVH file — animation-ready joint rotations
- `*_final.mp4` — a side-by-side of **input video | pose skeleton (2 views) | 3D mesh render (2 views)**

> **What success looks like:** the `expected_results/` folder in the [data-sample repo](https://huggingface.co/datasets/kehong/MoCapAnythingV2-data-sample/tree/main/expected_results) holds reference `*_final.mp4` outputs under `wild/`, `zoo/` and `obj/` (5 examples each) — your runs should look the same.

**5. Interactive web demo (Gradio)**
```bash
export PYTHONPATH=$PWD:$PWD/TripoSG
export BLENDER_BIN=/path/to/blender
python demo/app.py            # open http://localhost:7860
```
The app has two tabs:

- **🎯 Mocap · Retarget** — pick an example video (or **drag-and-drop your own**), choose a **target species** to retarget onto, and click **Run**. The result shows the pose skeleton and the 3D render together, with the `.npy` predictions available for download. The species dropdown lists every reference under your data directory (up to 73 with the full mini-dataset).
- **💃 Dance Anything** — drop a dance video **with music**: SAM2 auto-segments it into person layers, you click the dancer, pick a target character, and **Run** — the character performs the dance, re-muxed with the original audio.

> The Dance tab needs **SAM2** (optional): `pip install "sam2 @ git+https://github.com/facebookresearch/sam2.git"`. Weights (`facebook/sam2-hiera-large`) auto-download from HuggingFace on first use. If SAM2 is not installed, the Dance tab falls back to the RMBG background remover; the Mocap tab does not need it.

## Repository Layout

```
MocapAnything/
├── configs/              # YAML configs for each training and inference task
├── data/                 # Dataset loaders (loader_v1.py, loader_v2.py)
├── models/
│   ├── v1/               # mesh2pose, video2mesh (TripoSG)
│   └── v2/               # video2pose, video2pose2, pose2rot, video2pose2rot
├── preprocess/           # Raw FBX → training/inference data (see preprocess/README.md)
├── examples/custom_rig/  # Bring your own rigged FBX: walkthrough, runner, inference config
├── train/                # Training entrypoints (one per stage)
├── inference/            # Inference entrypoints (one per stage)
└── utils/                # Common utilities: loss, rotation, mesh, BVH, visualization, etc.
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/phongdaot/MocapAnything.git
   cd MocapAnything
   ```

2. Create a Python environment and install PyTorch (CUDA build matching your hardware) plus the usual ML stack:
   ```bash
   pip install -r requirements.txt   # or the minimal list from Quick Start, plus tensorboard for training
   ```

3. Install the TripoSG dependency used by the `video2mesh` pipeline (see the TripoSG repository for details) and place it on your `PYTHONPATH` so `from TripoSG.triposg... ` imports resolve.

4. Download model weights into `./checkpoints/`:
   - `checkpoints/TripoSG_temporal/` — temporal TripoSG weights
   - `checkpoints/RMBG-1.4/` — background removal network
   - `checkpoints/video2pose/`, `checkpoints/mesh2pose/`, `checkpoints/video2pose2rot/` — stage-specific weights

5. (Optional) Install Blender for final BVH / mesh rendering; point `output.blender_path` in the inference config to the binary.

## Data

The models are trained on a multi-species animal motion dataset organized under `datasets/zoo1030/`:

```
datasets/zoo1030/
├── bvh/                              # Ground-truth BVH sequences
├── bvh_pose/                         # Cached joint positions (.npz)
├── npz_mesh_normed/                  # Normalized mesh latents
├── npz_train_image_only/             # Precomputed image embeddings
├── species_info_dict.npy             # Per-species skeleton / T5 embeddings / adjacency
├── selected_test_split_release.json     # Train/seen/rare/unseen splits (released split)
├── characters/                       # Per-species mesh + skinning (for rendering / retargeting)
└── cache/                            # Per-species scale cache
```

Every one of those files is produced by `preprocess/run_pipeline.sh`, starting from raw FBX — see [`preprocess/README.md`](preprocess/README.md) for the stage-by-stage breakdown. The two that inference refuses to start without:

| File | Built by |
|---|---|
| `species_info_dict.npy` | `preprocess/build_species_info.py` (stage 7) — skeleton topology, joint-name T5 embeddings, static joints |
| `cache/__mesh2pose1002_species_scale_cache.pkl` | `preprocess/build_scale_cache.py` (stage 8) — per-species normalization scale |

The released end-to-end model trains without FPS memory banks (it conditions on the reference frame directly), so `num_memory: -1` in the config; per-species memory banks are only needed for the standalone `pose2rot` variants.

### Using your own rigged character

The pipeline is not tied to this dataset. Point it at your own FBX and it produces the same
artifacts, which is all inference needs:

```bash
export BLENDER=/path/to/blender PYTHON=/path/to/torch/python
bash examples/custom_rig/run.sh MyRig
python -m inference.video2pose2rot --config examples/custom_rig/inference.yaml
```

See [`examples/custom_rig/README.md`](examples/custom_rig/README.md) for what to prepare,
what each stage produces, and what to do when the automatic facing-direction or leaf-bone
guesses come out wrong.

## Training

Each stage has its own entrypoint and YAML config. All options — optimizer, schedule, losses, model sizes, attention window, split groups — live in the config file.

```bash
# Video → Pose
python -m train.video2pose --config configs/train/train_video2pose.yaml

# Mesh → Pose
python -m train.mesh2pose --config configs/train/train_mesh2pose.yaml

# Pose → Rotation
python -m train.pose2rot --config configs/train/train_pose2rot.yaml

# End-to-end Video → Pose → Rotation (joint fine-tune, single dataset)
python -m train.video2pose2rot --config configs/train/train_video2pose2rot.yaml

# Video → Mesh (TripoSG temporal)
python -m train.video2mesh --config configs/train/train_video2mesh.yaml
```

**Recommended: end-to-end multi-dataset training (the released recipe).**
Trains `video2pose2rot` jointly on `zoo1030 + obj1k` with reference-enhancement, scheduled teacher forcing (`pose_source_mode: mix`, 0.1→1.0 warmup over 20 epochs), and the released loss weights. All hyper-parameters live in `configs/train/train_video2pose2rot_multidata.yaml`.

```bash
# 8-GPU (adjust NGPU / CUDA_VISIBLE_DEVICES for your machine)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 bash train_multidata.sh
# equivalently:
torchrun --nproc_per_node=8 --master_port=29530 \
    -m train.video2pose2rot_multidata \
    --config configs/train/train_video2pose2rot_multidata.yaml
```
Training auto-resumes from the latest checkpoint in the experiment directory if interrupted (just re-run the same command).

Checkpoints are written under `output.checkpoint_root` (e.g. `./checkpoints/video2pose2rot/<exp>/`), along with TensorBoard logs and periodic comparison visualizations. The best checkpoint is selected by `eval.best_metric_split` / `eval.best_metric_name` (e.g. `seen` + `mpjpe` for pose, `rot_l1` for rotation).

Distributed multi-GPU training is supported through `utils/dist_utils.py`; launch with `torchrun` to enable.

## Inference

Inference scripts read the same YAML configs (inference variants) and operate on either:
- **Evaluation mode** (`data.wild_flag: false`) — compares predictions against GT sequences and reports metrics.
- **Wild mode** (`data.wild_flag: true`) — runs on in-the-wild image sequences using only a reference pose.

In wild mode, also set `data.wild_mode: true` to have the reference skeleton resolved from the clip filename: `MyRig#clip.mp4` picks any `MyRig#` sequence out of `data.base_dir`. Without it, a clip whose species has no ground-truth entry is skipped with a `[SKIP] ... ref bvh_pose` warning.

```bash
# Video → Mesh
python -m inference.video2mesh --config configs/inference/inference_video2mesh.yaml

# Mesh → Pose
python -m inference.mesh2pose --config configs/inference/inference_mesh2pose.yaml

# Video → Pose
python -m inference.video2pose --config configs/inference/inference_video2pose.yaml

# End-to-end Video → Pose → Rotation (outputs BVH)
python -m inference.video2pose2rot --config configs/inference/inference_video2pose2rot.yaml

# Your own rig, driven by in-the-wild footage
python -m inference.video2pose2rot --config examples/custom_rig/inference.yaml
```

Outputs are written to `output.save_dir` and include predicted pose `.npz` files, rotation sequences, BVH files, and (if Blender is configured) rendered comparison videos.

### Retargeting

For `mesh2pose`, `video2pose`, and `video2pose2rot`, set `data.retarget.toggle: true` and provide a `ref_seq` of the form `Species#Sequence/yRot` to retarget the predicted pose onto a GT skeleton before evaluation.

## Model Details

The v2 models share a common design: a transformer stack with per-joint tokens, reference-guided cross-attention, and sliding-window temporal self-attention. Key configuration knobs (see any `train_*` yaml):

- `q_dim`, `num_layers`, `num_heads`, `ref_layers` — transformer capacity
- `use_graph_ref_inner`, `use_graph_temporal_inner` — skeleton-graph biased attention
- `use_joint_embed` — per-joint T5-derived embeddings for cross-species generalization
- `attention_kwargs.seq_len`, `selfatt_slidwindow`, `crossatt2_slidwindow` — temporal window sizes
- `num_joints: 150` — maximum joints across all species (masked per sample)

The `pose2rot` model (`Pose2RotMemoryRestModel`) adds a memory branch conditioned on per-species FPS-sampled rotation banks, a rest-pose branch, and FiLM modulation into the decoder so rotations respect the species skeleton.

The `video2pose2rot` model wraps `video2pose` and `pose2rot` into a single module with schedulable teacher forcing (`pose_source_mode: mix`) so rotation training can be warmed up from GT poses and annealed toward predicted poses.

## Metrics & Reproduction

Evaluation splits — `seen`, `rare`, `unseen` — are reported independently. Common metrics:

- `mpjpe` — mean per-joint position error (JP, cm)
- `mpjve` — mean per-joint velocity error (JV, cm)
- `rot_l1`, `rot_smooth_l1` — rotation error (An, degrees)
- `speed_l1`, `speed_l2` — temporal smoothness (AV, degrees)

### Reproducing the released checkpoint

`video2pos2rot_epoch60.pt` was trained with the recipe above — `zoo1030 + obj1k`,
reference enhancement on, 8×GPU, batch 2/GPU, lr 1e-4, 60 epochs.

```bash
# Setting A — the released recipe (reference enhancement ON).
# configs/train/train_video2pose2rot_multidata.yaml ships with
#   zoo1030: ref_enhance: cross_seq     (same species, any sequence, any yaw)
#   obj1k:   ref_enhance: cross_angle   (same sequence, different yaw)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 bash train_multidata.sh

# Setting B — no reference enhancement (the ablation).
# Set ref_enhance: null on both dataset entries in the config, then:
CONFIG=configs/train/train_video2pose2rot_multidata_noaug.yaml \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 bash train_multidata.sh

# Evaluate either checkpoint (data.wild_flag: false → per-split metrics)
python -m inference.video2pose2rot --config configs/inference/inference_video2pose2rot.yaml
```

Both settings are otherwise identical, so they are directly comparable — pick whichever
matches what you want to measure (see *Reference enhancement* below).

Splits: `datasets/zoo1030/selected_test_split_release.json` and
`datasets/obj1k/select_test_obj1k.json`, both shipped in this repository.

### Expected numbers

JP / JV in cm, An / AV in degrees. Lower is better.

| | Zoo-Seen | Zoo-Rare | Zoo-Unseen | Obj |
| --- | --- | --- | --- | --- |
| **Released ckpt, released split** | 2.48 / 0.61 / 11.22 / 0.300 | 4.15 / 0.86 / 14.45 / 0.380 | 5.70 / 0.68 / **19.58** / 0.516 | 4.85 / 1.20 / 12.07 / 0.302 |
| Retrained from this repository | 2.60 / 0.65 / 11.29 / 0.301 | 4.05 / 0.91 / 14.79 / 0.386 | 5.82 / 0.72 / 20.51 / 0.541 | 4.66 / 1.34 / 11.71 / 0.287 |
| *Paper, Table 1* | *2.34 / 0.53 / 10.73 / 0.29* | *2.98 / 0.61 / 14.38 / 0.37* | *3.39 / 0.99 / **6.54** / 0.17* | *3.84 / 1.05 / 11.06 / 0.30* |

Retraining from this repository reproduces the released checkpoint closely (within
~0.1 cm and ~0.5°), so the pipeline here is faithful to how the released weights were
produced.

### ⚠️ These do not match the paper

**The paper's Table 1 was computed on a different train/test split than the one released
here.** The difference is concentrated on Zoo-Unseen — about 19.6° on the released split
against 6.54° in the paper. The seen / rare / obj columns agree to within roughly 1 cm
and 1–2°.

Unseen-species rotation error depends heavily on *which* species land in the unseen
bucket, and the two splits partition them differently. The paper's observation that
unseen error falls below seen error holds on its split, not on this one. If you are
comparing against this work, please compare on the split you can actually download —
the one in this repository.

### Reference enhancement and retargeting

`ref_enhance` draws the reference frame from elsewhere in the same species (`cross_seq`,
animals) or from a different yaw of the same sequence (`cross_angle`, objects). It is a
training-time augmentation only and is never used at evaluation.

There is a real trade-off. An otherwise identical run **without** it scores slightly
better across all four benchmark columns above — so if you are chasing benchmark numbers,
Setting B is the stronger choice. The released checkpoint nonetheless uses Setting A,
because it produces visibly better **retargeting and in-the-wild results**, which is what
the demo and most downstream use actually depends on. Train whichever matches your goal,
and say which one you used when reporting numbers.

## License

MIT — see [LICENSE](LICENSE). The released V2 weights are MIT as well.
`preprocess/briarmbg.py` (RMBG-1.4, © BRIA AI) and the Truebones Zoo motion data are not
covered by that grant and follow their own terms; see the third-party notice in
[LICENSE](LICENSE).

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

## Repository Layout

```
MocapAnything/
├── configs/              # YAML configs for each training and inference task
├── data/                 # Dataset loaders (loader_v1.py, loader_v2.py)
├── models/
│   ├── v1/               # mesh2pose, video2mesh (TripoSG)
│   └── v2/               # video2pose, video2pose2, pose2rot, video2pose2rot
├── preprocess/           # Image preprocessing, background removal, data preparation
├── train/                # Training entrypoints (one per stage)
├── inference/            # Inference entrypoints (one per stage)
└── utils/                # Common utilities: loss, rotation, mesh, BVH, visualization, etc.
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/animotionlab26/MocapAnything.git
   cd MocapAnything
   ```

2. Create a Python environment and install PyTorch (CUDA build matching your hardware) plus the usual ML stack:
   ```bash
   pip install torch torchvision numpy trimesh pyyaml tqdm tensorboard huggingface_hub pillow
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
├── selected_test_split1010.json      # Train/seen/rare/unseen splits
└── cache/
    └── species_fps_memory_yAll/      # FPS-sampled per-species memory banks
```

Use `preprocess/preprocess_data.py` to build mesh and image caches from raw sequences, and `preprocess/species_fps_memory.py` to generate per-species memory banks used by `pose2rot`.

## Training

Each stage has its own entrypoint and YAML config. All options — optimizer, schedule, losses, model sizes, attention window, split groups — live in the config file.

```bash
# Video → Pose
python -m train.video2pose --config configs/train_video2pose.yaml

# Mesh → Pose
python -m train.mesh2pose --config configs/train_mesh2pose.yaml

# Pose → Rotation
python -m train.pose2rot --config configs/train_pose2rot.yaml

# End-to-end Video → Pose → Rotation (joint fine-tune)
python -m train.video2pose2rot --config configs/train_video2pose2rot.yaml

# Video → Mesh (TripoSG temporal)
python -m train.video2mesh --config configs/train_video2mesh.yaml
```

Checkpoints are written under `output.checkpoint_root` (e.g. `./checkpoints/video2pose/<exp>/`), along with TensorBoard logs and periodic comparison visualizations. The best checkpoint is selected by `eval.best_metric_split` / `eval.best_metric_name` (e.g. `seen` + `mpjpe` for pose, `rot_l1` for rotation).

Distributed multi-GPU training is supported through `utils/dist_utils.py`; launch with `torchrun` to enable.

## Inference

Inference scripts read the same YAML configs (inference variants) and operate on either:
- **Evaluation mode** (`data.wild_flag: false`) — compares predictions against GT sequences and reports metrics.
- **Wild mode** (`data.wild_flag: true`) — runs on in-the-wild image sequences using only a reference pose.

```bash
# Video → Mesh
python -m inference.video2mesh --config configs/inference_video2mesh.yaml

# Mesh → Pose
python -m inference.mesh2pose --config configs/inference_mesh2pose.yaml

# Video → Pose
python -m inference.video2pose --config configs/inference_video2pose.yaml

# End-to-end Video → Pose → Rotation (outputs BVH)
python -m inference.video2pose2rot --config configs/inference_video2pose2rot.yaml
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

## Metrics

Evaluation splits — `seen`, `rare`, `unseen` — are reported independently. Common metrics:

- `mpjpe` — mean per-joint position error
- `mpjve` — mean per-joint velocity error
- `rot_l1`, `rot_smooth_l1` — rotation error
- `speed_l1`, `speed_l2` — temporal smoothness

## License

See the repository for license information.

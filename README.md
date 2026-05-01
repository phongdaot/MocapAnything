# MoCapAnything V2

**End-to-End Motion Capture for Arbitrary Skeletons from Monocular Videos**

[Project Page](https://animotionlab.github.io/MoCapAnythingV2/) · [Paper (arXiv)](https://arxiv.org/abs/2604.28130) · [Install & Run](RUN.md)

> ⚠️ **Unofficial code release.** This repository is a reimplementation based on the paper — use as a reference, not a reproduction.

<p align="center">
  <a href="https://animotionlab.github.io/MoCapAnythingV2/" title="Click to watch the 90-second teaser on the project page">
    <img src="assets/teaser_play.png" width="92%" alt="MoCapAnything V2 teaser — click to watch the video on the project page" />
  </a>
</p>

<p align="center"><sub>▶ Click the image to watch the 90-second teaser on the project page.</sub></p>

## Highlights

- 🔗 **Fully end-to-end.** Video → Pose → Rotation jointly optimized — no analytical IK in the loop.
- ⚓ **Reference-anchored rotation.** A single reference pose–rotation pair from the target asset defines the rotation coordinate system, turning pose-to-rotation into a well-constrained problem.
- ⚡ **Mesh-free and fast.** Joints predicted directly from video, ~20× faster than mesh-based pipelines.

## Pipeline

The V2 main model is **`video2pose2rot`** — a single end-to-end network that maps a video directly to BVH-ready joint rotations. Internally it composes two subtasks, `video2pose` and `pose2rot`, that share weights and are jointly fine-tuned; they can be run standalone (e.g. for ablations or debugging), but normal usage is the joint model. The V1 mesh-based pipeline (`video2mesh` + `mesh2pose`) is included as a baseline for comparison.

| Stage | Role | Input | Output |
| --- | --- | --- | --- |
| **`video2pose2rot`** | **V2 — main end-to-end model** | Image sequence + reference | Joint rotations (BVH) |
| &nbsp;&nbsp;↳ `video2pose` | V2 subtask (standalone-runnable) | Image sequence + reference pose | Joint positions |
| &nbsp;&nbsp;↳ `pose2rot` | V2 subtask (standalone-runnable) | Joint positions + rest pose / reference pose-rot pair | Joint rotations |
| `video2mesh` | V1 baseline — mesh sequence (TripoSG) | Image sequence | Mesh (`.glb` / latent) |
| `mesh2pose` | V1 baseline — joints from per-frame meshes | Mesh sequence + reference pose | Joint positions |

A reference frame from a matching species guides the per-species skeleton and scale, enabling generalization to unseen animals.

## Install & Run

Environment setup, dataset layout, training commands, and inference (including in-the-wild mode) live in **[RUN.md](RUN.md)**.

## Citation

If you use this code, please consider cite:

```bibtex
@article{gong2026mocapanythingv2,
  title   = {MoCapAnything V2: End-to-End Motion Capture for Arbitrary Skeletons},
  author  = {Gong, Kehong and Wen, Zhengyu and Phong, Dao Thien and
             Xu, Mingxi and He, Weixia and Wang, Qi and Zhang, Ning and
             Li, Zhengyu and Hou, Guanli and Lian, Dongze and He, Xiaoyu and
             Zhang, Mingyuan and Zhang, Hanwang},
  journal = {arXiv preprint arXiv:2604.28130},
  year    = {2026}
}
```

If you build on the V1 baselines, please also cite the underlying papers — `mesh2pose` is from **MoCapAnything (V1)**, and `video2mesh` is from **SWiT-4D**:

```bibtex
@article{gong2025mocapanything,
  title   = {MoCapAnything: Unified 3D Motion Capture for Arbitrary Skeletons from Monocular Videos},
  author  = {Gong, Kehong and Wen, Zhengyu and He, Weixia and Xu, Mingxi and
             Wang, Qi and Zhang, Ning and Li, Zhengyu and
             Lian, Dongze and Zhao, Wei and He, Xiaoyu and Zhang, Mingyuan},
  journal = {arXiv preprint arXiv:2512.10881},
  year    = {2025}
}

@article{gong2025swit4d,
  title   = {SWiT-4D: Sliding-Window Transformer for Lossless and Parameter-Free Temporal 4D Generation},
  author  = {Gong, Kehong and Wen, Zhengyu and Xu, Mingxi and He, Weixia and Wang, Qi and
             Zhang, Ning and Li, Zhengyu and Li, Chenbin and Lian, Dongze and
             Zhao, Wei and He, Xiaoyu and Zhang, Mingyuan},
  journal = {arXiv preprint arXiv:2512.10860},
  year    = {2025}
}
```

## License

See the repository for license information.

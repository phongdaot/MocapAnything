# 🎮 Interactive Demo

<p align="center">
  <img src="../assets/demo_app.png" width="96%" alt="MocapAnything V2 interactive demo — pick a video, pick a target species/object, run: pose skeleton + 3D mesh render, with retargeting and a Dance Anything tab" />
</p>

A Gradio web app with two tabs:

- **🎯 Mocap · Retarget** — pick an example video (or drag-and-drop your own), pick a **target species/object** (can differ from the input → retargeting), hit **Run**: you get the pose skeleton + the 3D mesh render side by side, with the `.npy` predictions downloadable.
- **💃 Dance Anything** — drop a dance video with music: SAM2 segments the person layers, you click the dancer, pick a target character, and the character performs the dance re-muxed with the original audio.

## Launch

```bash
# from the repo root — weights + demo data from HuggingFace first (see ../RUN.md):
hf download kehong/MoCapAnythingV2-weights --local-dir ./checkpoints
hf download kehong/MoCapAnythingV2-data-sample --repo-type dataset --local-dir ./demo/data

export PYTHONPATH=$PWD:$PWD/TripoSG
export BLENDER_BIN=/path/to/blender      # portable Blender 4.x/5.x, for the 3D mesh render
python demo/app.py                        # → http://localhost:7860
```

- `APP_PORT` / `APP_DEVICE` env vars override the port and GPU.
- The Dance tab needs **SAM2** (optional — falls back to RMBG if absent): `pip install "sam2 @ git+https://github.com/facebookresearch/sam2.git"`.
- Reference outputs for the bundled examples live in the data repo's [`expected_results/`](https://huggingface.co/datasets/kehong/MoCapAnythingV2-data-sample/tree/main/expected_results).

## Layout

| Path | What it is |
| --- | --- |
| `app.py` | The Gradio app (loads the model once, streams stage progress to the run terminal) |
| `configs/demo_{zoo,obj,wild}.yaml` | Command-line inference configs for the bundled examples |
| `assets/` | Species gallery images + example-video thumbnails |
| `data/` | Demo dataset (downloaded from HuggingFace, not in git) |
| `dance_utils.py` / `sam_utils.py` | Audio mux helpers · SAM2 person segmentation (lazy-loaded) |

### eval_noise_sweep.py ###
"""
Reference-noise robustness sweep for the pose->rotation model.

Loads a trained video2pose2rot checkpoint and evaluates it repeatedly while
injecting noise into the *reference* that anchors the rotation coordinate
system (the memory pose/rotation pair, and optionally the static-joint ref).
Produces:

  1. noise_vs_<metric>.png   - accuracy vs. noise level, one curve per split
  2. noise_sweep_results.json - raw numbers (mean/std over seeds)
  3. ref_noise_vis.png        - the reference skeleton clean vs. increasingly
                                noised, so you can *see* the noise

The noise magnitude for rotations is expressed in DEGREES (a random rotation of
that magnitude is composed onto each joint's reference rotation). No retraining
is needed for the sweep: `ref_noise_std` is a forward kwarg, so one checkpoint
is evaluated across all noise levels.

Example
-------
    python -m train.eval_noise_sweep \
        --config configs/train/train_video2pose2rot.yaml \
        --ckpt   checkpoints/video2pose2rot/<exp>/video2pose2rot_ckpt_best.pt \
        --levels 0,5,10,20,30,45,60,90 \
        --target mem_rot --metric angle_l1 --seeds 0,1,2 \
        --out-dir ./noise_sweep_out
"""
import os
import json
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection)

from utils.config_utils import load_yaml_config, instantiate_from_config
from utils.common import set_seed
from utils.rotation import add_rot6d_noise, rot6d_to_fk_positions
from train.video2pose import build_test_dataloaders
from train.video2pose2rot import run_evaluation


# =========================================================
# helpers
# =========================================================
def parse_levels(s):
    if s is None:
        return None
    return [float(x) for x in str(s).split(",") if x != ""]


def load_model(cfg, ckpt_path, device):
    model = instantiate_from_config(cfg["model"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        sd = None
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in ckpt:
                sd = ckpt[key]
                break
        if sd is None:
            sd = ckpt
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[load]   e.g. missing: {list(missing)[:5]}")
    if unexpected:
        print(f"[load]   e.g. unexpected: {list(unexpected)[:5]}")
    model.eval()
    return model


def _draw_skeleton(ax, pos, parents, mask, color, label):
    J = pos.shape[0]
    for j in range(J):
        if not mask[j]:
            continue
        p = int(parents[j])
        if 0 <= p < J and p != j and mask[p]:
            ax.plot(
                [pos[j, 0], pos[p, 0]],
                [pos[j, 1], pos[p, 1]],
                [pos[j, 2], pos[p, 2]],
                c=color, lw=1.0,
            )
    ax.scatter(pos[mask, 0], pos[mask, 1], pos[mask, 2], c=color, s=8, label=label)


def visualize_ref_noise(batch, levels, target, out_dir, sample_idx=0, seed=0):
    """Save a row of 3D plots: reference skeleton FK'd from the (noised) memory
    rotation, at each noise level, with the clean skeleton overlaid in grey."""
    set_seed(seed)
    offset = batch["offset_a"]
    parents = batch["parent_a"]
    gscale = batch["global_scale"]
    mem_rot = batch["memory_rot6d"][:, :1].contiguous()   # [B,1,J,6] first memory slot

    mask = batch["joint_mask"][sample_idx].bool().cpu().numpy()
    par = parents[sample_idx].cpu().numpy()
    clean_pos = rot6d_to_fk_positions(mem_rot, offset, parents, gscale)[sample_idx, 0].cpu().numpy()

    n = len(levels)
    fig = plt.figure(figsize=(4 * n, 4))
    for i, std in enumerate(levels):
        noisy = add_rot6d_noise(mem_rot, std) if std > 0 else mem_rot
        pos = rot6d_to_fk_positions(noisy, offset, parents, gscale)[sample_idx, 0].cpu().numpy()
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        if std > 0:
            _draw_skeleton(ax, clean_pos, par, mask, "0.7", "clean")
        _draw_skeleton(ax, pos, par, mask, "C3" if std > 0 else "C0", f"{std:.0f}deg")
        ax.set_title(f"ref noise = {std:.0f} deg")
        ax.set_axis_off()
    sp = batch["species"][sample_idx] if "species" in batch else ""
    fig.suptitle(f"reference skeleton vs rotation noise (target={target})  {sp}")
    fig.tight_layout()
    path = os.path.join(out_dir, "ref_noise_vis.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Reference-noise robustness sweep")
    ap.add_argument("--config", required=True, help="training yaml (model/data/eval/train sections)")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt to evaluate")
    ap.add_argument("--out-dir", default="./noise_sweep_out")
    ap.add_argument("--levels", default="0,5,10,20,30,45,60,90",
                    help="rotation noise levels in degrees (comma-separated)")
    ap.add_argument("--pos-levels", default=None,
                    help="position noise stds (comma-separated); used when target includes mem_pose")
    ap.add_argument("--target", default="mem_rot",
                    choices=["mem_rot", "mem_pose", "static", "all"])
    ap.add_argument("--metric", default="angle_l1",
                    help="metric key from evaluate_joint_metrics (e.g. angle_l1, rot_l1, fk_l1)")
    ap.add_argument("--seeds", default="0,1,2", help="seeds to average the (stochastic) noise over")
    ap.add_argument("--splits", default=None, help="comma-separated subset of split_groups")
    ap.add_argument("--vis-levels", default="0,15,30,60,90", help="noise levels for the skeleton viz")
    ap.add_argument("--no-vis", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_yaml_config(args.config)
    # disable eval-time BVH visualization inside run_evaluation
    cfg.setdefault("train", {})["vis_every"] = 10 ** 9
    # run_evaluation reads cfg["runtime"]["debug"]; guard if a lean config is passed
    cfg.setdefault("runtime", {}).setdefault("debug", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention_design = cfg["model"]["attention_kwargs"]

    model = load_model(cfg, args.ckpt, device)

    _, test_loaders = build_test_dataloaders(
        data_cfg=cfg["data"],
        eval_cfg=cfg["eval"],
        attention_design=attention_design,
        distributed=False,
        rank=0,
        world_size=1,
    )

    splits = args.splits.split(",") if args.splits else list(test_loaders.keys())
    levels = parse_levels(args.levels)
    pos_levels = parse_levels(args.pos_levels)
    if pos_levels is None:
        pos_levels = [0.0] * len(levels)
    assert len(pos_levels) == len(levels), "--pos-levels must match --levels length"
    seeds = [int(s) for s in args.seeds.split(",")]
    char_dir = cfg["data"].get("character_dir", "")

    results = {}
    for split in splits:
        loader = test_loaders[split]
        means, stds = [], []
        for li, std in enumerate(levels):
            pstd = pos_levels[li] if args.target in ("mem_pose", "all") else 0.0
            vals = []
            for sd in seeds:
                set_seed(sd)  # different noise draw per seed, reproducible
                m = run_evaluation(
                    loader=loader,
                    model=model,
                    device=device,
                    attention_design=attention_design,
                    cfg=cfg,
                    pose_pred_prob=1.0,          # end-to-end (pred pose) path
                    base_dir=args.out_dir,
                    character_dir=char_dir,
                    writer=None,
                    epoch=0,                     # int (vis disabled via huge vis_every)
                    tag_prefix=f"{split}_std{std:g}",
                    ref_noise_std=std,
                    ref_pos_noise_std=pstd,
                    ref_noise_target=args.target,
                )
                vals.append(float(m[args.metric]))
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            print(f"[sweep][{split}] std={std:g} pos_std={pstd:g} "
                  f"{args.metric}={means[-1]:.4f} +/- {stds[-1]:.4f}")
        results[split] = {"levels": levels, "mean": means, "std": stds}

    # ---- plot: noise vs metric ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, r in results.items():
        ax.errorbar(r["levels"], r["mean"], yerr=r["std"], marker="o", capsize=3, label=split)
    xlabel = "reference position noise (std)" if args.target == "mem_pose" \
        else "reference rotation noise (deg)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{args.metric}  (lower = better)")
    ax.set_title(f"Reference noise vs. accuracy  (target={args.target}, seeds={seeds})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, f"noise_vs_{args.metric}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    with open(os.path.join(args.out_dir, "noise_sweep_results.json"), "w") as f:
        json.dump(
            {"metric": args.metric, "target": args.target, "seeds": seeds,
             "levels": levels, "results": results},
            f, indent=2,
        )

    # ---- visualize the noise on the reference skeleton ----
    vis_path = None
    if not args.no_vis:
        vis_levels = parse_levels(args.vis_levels)
        batch = next(iter(test_loaders[splits[0]]))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        vis_path = visualize_ref_noise(batch, vis_levels, args.target, args.out_dir)

    print("=" * 60)
    print(f"[done] plot   : {plot_path}")
    print(f"[done] json   : {os.path.join(args.out_dir, 'noise_sweep_results.json')}")
    if vis_path:
        print(f"[done] noisevis: {vis_path}")


if __name__ == "__main__":
    main()

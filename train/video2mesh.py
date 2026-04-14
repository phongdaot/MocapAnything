
### video2mesh.py ###
# Training script for the video2mesh temporal DiT (TripoSGDiTModel4D).
#
# Assumes the dataset has already been preprocessed with
# preprocess/preprocess_data.py, which stores per-sample .npz files of
# (VAE-encoded mesh latent, frozen DinoV2 image embedding) sequences.
# Training optimises only the transformer using flow matching on the
# frozen latents — the VAE and image encoder are not needed here.

import os
import time
import math
import argparse
from typing import Dict, Any, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils.dist_utils import (
    setup_distributed,
    is_main_process,
    cleanup_distributed,
    worker_init_fn,
    reduce_sum_scalar,
)
from utils.train_utils import (
    save_checkpoint_with_epoch,
    cleanup_old_checkpoints,
    find_latest_ckpt,
    load_checkpoint,
    load_partial_pretrain,
)
from utils.common import set_seed
from utils.logger import logger
from utils.config_utils import load_yaml_config, dump_yaml_config, instantiate_from_config

from data.loader_v1_mesh import VideoMeshLatentDataset, collate_video_mesh

torch.multiprocessing.set_start_method("spawn", force=True)

PROCESS_NAME = "video2mesh"


# =========================================================
# flow-matching helpers
# =========================================================
def _sample_flow_matching_t(
    batch_size: int,
    device: torch.device,
    schedule: str = "uniform",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
) -> torch.Tensor:
    """
    Sample flow-matching time parameter u in (0, 1) per sample.

    schedule:
        uniform       : u ~ U(0, 1)
        logit_normal  : u = sigmoid(N(mean, std))  (as used in SD3/Flux)
    """
    if schedule == "uniform":
        u = torch.rand(batch_size, device=device)
    elif schedule == "logit_normal":
        z = torch.randn(batch_size, device=device) * logit_normal_std + logit_normal_mean
        u = torch.sigmoid(z)
    else:
        raise ValueError(f"Unknown flow matching schedule: {schedule}")

    # Clamp away from endpoints for numerical stability.
    return u.clamp(1e-5, 1.0 - 1e-5)


def compute_flow_matching_loss(
    transformer: nn.Module,
    latent: torch.Tensor,           # [B, T, N, D]
    image_embed: torch.Tensor,      # [B, T, C, D_cond]
    attention_kwargs: Optional[Dict[str, Any]],
    t_schedule: str,
    t_mean: float,
    t_std: float,
    timestep_scale: float,
    loss_type: str = "l2",
):
    """
    Standard rectified-flow / flow-matching loss:
        x_0 = data (mesh latent)
        x_1 = noise ~ N(0, I)
        x_t = (1 - u) x_0 + u x_1
        target = x_1 - x_0
        pred   = transformer(x_t, t_model, cond)
        loss   = || pred - target ||
    """
    B = latent.shape[0]
    device = latent.device

    u = _sample_flow_matching_t(
        batch_size=B,
        device=device,
        schedule=t_schedule,
        logit_normal_mean=t_mean,
        logit_normal_std=t_std,
    )
    # Broadcast u over [T, N, D] dims.
    u_b = u.view(B, 1, 1, 1).to(latent.dtype)

    noise = torch.randn_like(latent)
    x_t = (1.0 - u_b) * latent + u_b * noise
    target = noise - latent

    # TripoSG expects the timestep in [0, 1000] range (positional embedding).
    timestep = (u * timestep_scale).to(latent.dtype)

    pred_out = transformer(
        hidden_states=x_t,
        timestep=timestep,
        encoder_hidden_states=image_embed,
        attention_kwargs=attention_kwargs,
        return_dict=False,
    )
    pred = pred_out[0]

    if loss_type == "l2":
        per_sample = (pred.float() - target.float()).pow(2).mean(dim=[1, 2, 3])
    elif loss_type == "l1":
        per_sample = (pred.float() - target.float()).abs().mean(dim=[1, 2, 3])
    elif loss_type == "smooth_l1":
        per_sample = nn.functional.smooth_l1_loss(
            pred.float(), target.float(), reduction="none"
        ).mean(dim=[1, 2, 3])
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss = per_sample.mean()
    return loss, {
        "loss": loss.detach(),
        "u_mean": u.mean().detach(),
        "u_std": u.std(unbiased=False).detach() if B > 1 else torch.zeros((), device=device),
    }


# =========================================================
# dataloaders
# =========================================================
def build_dataloaders(data_cfg, train_cfg, eval_cfg, attention_design, distributed, rank, world_size):
    window = attention_design["seq_len"]

    train_ds = VideoMeshLatentDataset(
        npz_dir=data_cfg["npz_dir"],
        window=window,
        split_json=data_cfg.get("split_json"),
        split_mode="train",
        min_frames=data_cfg.get("min_frames"),
        preload_all=data_cfg.get("preload_all", False),
        debug_limit=data_cfg.get("debug_limit"),
    )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=train_cfg.get("num_workers_train", 4),
        collate_fn=collate_video_mesh,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    eval_ds = VideoMeshLatentDataset(
        npz_dir=data_cfg["npz_dir"],
        window=window,
        split_json=data_cfg.get("split_json"),
        split_mode="test",
        min_frames=data_cfg.get("min_frames"),
        preload_all=data_cfg.get("preload_all", False),
        debug_limit=data_cfg.get("debug_limit"),
    )

    if len(eval_ds) == 0:
        if is_main_process():
            logger.info("[video2mesh] No test split found — evaluation will be skipped.")
        eval_loader = None
    else:
        eval_sampler = (
            DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False)
            if distributed else None
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=eval_cfg.get("batch_size", train_cfg["batch_size"]),
            sampler=eval_sampler,
            shuffle=False,
            num_workers=eval_cfg.get("num_workers", 2),
            collate_fn=collate_video_mesh,
            worker_init_fn=worker_init_fn,
            drop_last=False,
        )

    if is_main_process():
        logger.info(f"[video2mesh] train samples: {len(train_ds)}")
        logger.info(f"[video2mesh] eval  samples: {len(eval_ds)}")

    return train_loader, eval_loader


# =========================================================
# eval
# =========================================================
@torch.no_grad()
def run_evaluation(
    loader,
    transformer,
    device,
    attention_design,
    cfg,
    writer=None,
    epoch=None,
    tag_prefix="test",
):
    if loader is None:
        return {}

    transformer.eval()

    loss_cfg = cfg["train"]["loss"]
    loss_sum = 0.0
    sample_count = 0

    tqdm_loader = (
        tqdm(loader, total=len(loader), desc=f"Eval: {tag_prefix}", ncols=120)
        if is_main_process() else loader
    )

    cnt = 0
    for batch in tqdm_loader:
        cnt += 1
        if cfg["runtime"]["debug"] and cnt >= 10:
            break

        latent = batch["latent"].to(device, non_blocking=True)
        image_embed = batch["image_embed"].to(device, non_blocking=True)

        loss, _ = compute_flow_matching_loss(
            transformer=transformer,
            latent=latent,
            image_embed=image_embed,
            attention_kwargs=attention_design,
            t_schedule=loss_cfg.get("t_schedule", "uniform"),
            t_mean=loss_cfg.get("t_mean", 0.0),
            t_std=loss_cfg.get("t_std", 1.0),
            timestep_scale=loss_cfg.get("timestep_scale", 1000.0),
            loss_type=loss_cfg.get("loss_type", "l2"),
        )

        bs = latent.size(0)
        loss_sum += float(loss.item()) * bs
        sample_count += bs

    sample_count = reduce_sum_scalar(sample_count, device)
    loss_sum = reduce_sum_scalar(loss_sum, device) / max(sample_count, 1)

    if is_main_process():
        logger.info(f"[{tag_prefix}] fm_loss={loss_sum:.6f}")

    if writer is not None and epoch is not None:
        writer.add_scalar(f"{tag_prefix}/fm_loss", loss_sum, epoch)

    return {"fm_loss": loss_sum}


# =========================================================
# main
# =========================================================
def train_video2mesh(cfg):
    distributed, rank, world_size, local_rank = setup_distributed()

    runtime_cfg = cfg["runtime"]
    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    output_cfg = cfg["output"]

    set_seed(runtime_cfg.get("seed", 42))

    base_dir = os.path.join(output_cfg["checkpoint_root"], exp_cfg["exp"])
    os.makedirs(base_dir, exist_ok=True)

    if is_main_process():
        dump_yaml_config(cfg, os.path.join(base_dir, "config.yaml"))

    logdir = os.path.join(base_dir, f"logs_{PROCESS_NAME}")
    if is_main_process():
        logger.info(f"Log directory: {logdir}")

    device_str = runtime_cfg.get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    attention_design = model_cfg["attention_kwargs"]

    # ---- data ----
    train_loader, eval_loader = build_dataloaders(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        attention_design=attention_design,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    # ---- model (transformer only) ----
    transformer: nn.Module = instantiate_from_config(model_cfg)

    if train_cfg.get("use_grad_checkpoint", False):
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        else:
            transformer.gradient_checkpointing = True

    pretrain_ckpt = train_cfg.get("pretrain_ckpt")
    if pretrain_ckpt is not None:
        load_partial_pretrain(transformer, pretrain_ckpt)

    transformer = transformer.to(device)

    if distributed:
        transformer = torch.nn.parallel.DistributedDataParallel(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=train_cfg.get("find_unused_parameters", False),
        )

    # ---- optimizer ----
    lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 0.0)
    betas = tuple(train_cfg.get("betas", (0.9, 0.999)))

    if weight_decay == 0.0:
        optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=betas)
    else:
        optimizer = optim.AdamW(
            transformer.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )

    writer = SummaryWriter(logdir) if is_main_process() else None

    if is_main_process():
        total_params = sum(p.numel() for p in transformer.parameters())
        trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        logger.info("=" * 60)
        logger.info(f"[{PROCESS_NAME}] total     params: {total_params:,}")
        logger.info(f"[{PROCESS_NAME}] trainable params: {trainable:,}")
        logger.info(f"[{PROCESS_NAME}] seq_len: {attention_design['seq_len']}")
        logger.info(f"[{PROCESS_NAME}] loss cfg: {train_cfg['loss']}")
        logger.info("=" * 60)

    best_test_loss = float("inf")
    best_ckpt_path = os.path.join(base_dir, f"{PROCESS_NAME}_ckpt_best.pt")
    start_epoch = 0

    ckpt_path_latest = find_latest_ckpt(base_dir, PROCESS_NAME)
    if ckpt_path_latest and os.path.exists(ckpt_path_latest):
        if is_main_process():
            logger.info(f"Loading checkpoint: {ckpt_path_latest}")
        start_epoch = load_checkpoint(
            transformer.module if distributed else transformer,
            optimizer,
            ckpt_path_latest,
            device,
        )
        if is_main_process():
            logger.info(f"Resumed from epoch {start_epoch}")

    amp_dtype_str = runtime_cfg.get("amp_dtype", "float16")
    amp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(amp_dtype_str, torch.float16)
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    global_step = 0
    epochs = train_cfg["epochs"]
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    for epoch in range(start_epoch, epochs):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        transformer.train()
        running_loss = 0.0
        seen_samples = 0
        epoch_start_time = time.time()

        loader_tqdm = (
            tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs} [train][rank{rank}]",
                ncols=120,
            )
            if is_main_process()
            else enumerate(train_loader)
        )

        batch_times = []
        optimizer.zero_grad(set_to_none=True)

        cnt = 0
        for i, batch in loader_tqdm:
            cnt += 1
            if cfg["runtime"]["debug"] and cnt >= 10:
                break
            batch_start_time = time.time()

            latent = batch["latent"].to(device, non_blocking=True)
            image_embed = batch["image_embed"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                raw_loss, loss_info = compute_flow_matching_loss(
                    transformer=transformer,
                    latent=latent,
                    image_embed=image_embed,
                    attention_kwargs=attention_design,
                    t_schedule=train_cfg["loss"].get("t_schedule", "uniform"),
                    t_mean=train_cfg["loss"].get("t_mean", 0.0),
                    t_std=train_cfg["loss"].get("t_std", 1.0),
                    timestep_scale=train_cfg["loss"].get("timestep_scale", 1000.0),
                    loss_type=train_cfg["loss"].get("loss_type", "l2"),
                )

            loss = raw_loss / grad_accum_steps
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_grad_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            bs = latent.size(0)
            running_loss += raw_loss.item() * bs
            seen_samples += bs

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("train/fm_loss", raw_loss.item(), global_step)
                writer.add_scalar("train/u_mean", float(loss_info["u_mean"].item()), global_step)
                writer.add_scalar("train/u_std", float(loss_info["u_std"].item()), global_step)

            global_step += 1

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            if is_main_process() and i % 10 == 0 and i > 0:
                avg_batch = sum(batch_times) / len(batch_times)
                remaining = avg_batch * (len(train_loader) - i - 1)
                loader_tqdm.set_postfix_str(
                    f"Loss={raw_loss.item():.4f} | ETA={remaining / 60:.1f}min"
                )

        train_loss = running_loss / max(seen_samples, 1)
        epoch_time = time.time() - epoch_start_time

        if is_main_process():
            logger.info(
                f"Epoch {epoch + 1}: train fm_loss={train_loss:.6f} | time {epoch_time:.1f}s"
            )
        if writer is not None:
            writer.add_scalar("epoch/train_fm_loss", train_loss, epoch + 1)

        # -------- eval + checkpoint --------
        if (epoch + 1) % train_cfg.get("test_every", 1) == 0 or (epoch + 1) == epochs:
            eval_metrics = run_evaluation(
                loader=eval_loader,
                transformer=transformer,
                device=device,
                attention_design=attention_design,
                cfg=cfg,
                writer=writer,
                epoch=epoch + 1,
                tag_prefix="test",
            )

            if is_main_process():
                save_checkpoint_with_epoch(
                    transformer.module if distributed else transformer,
                    PROCESS_NAME,
                    optimizer,
                    epoch + 1,
                    base_dir,
                )
                cleanup_old_checkpoints(base_dir, PROCESS_NAME, train_cfg.get("max_ckpt", 5))

                best_metric_name = eval_cfg.get("best_metric_name", "fm_loss")
                score = eval_metrics.get(best_metric_name, train_loss)
                if score < best_test_loss:
                    best_test_loss = score
                    torch.save(
                        {
                            "model_state": (transformer.module if distributed else transformer).state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch + 1,
                            "best_test_loss": best_test_loss,
                            "eval_metrics": eval_metrics,
                        },
                        best_ckpt_path,
                    )
                    logger.info(
                        f"New best checkpoint saved: {best_ckpt_path} "
                        f"({best_metric_name}={best_test_loss:.6f})"
                    )

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Video2Mesh (TripoSG temporal DiT)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_video2mesh.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train_video2mesh(cfg)

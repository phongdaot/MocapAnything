### video2mesh.py ###
"""Training entrypoint for the temporal TripoSG DiT (video2mesh).

This script trains ``TripoSGDiTModel4D`` — the 4D (temporal) TripoSG diffusion
transformer — with rectified-flow matching loss on precomputed
``(image_embed, latent)`` sequences.  Optional LoRA adapters can be attached
and, on save, merged back into a full checkpoint on CPU.

Layout follows the rest of the repo:

- Single ``--config`` YAML argument; all hyperparameters live there.
- DDP via ``utils/dist_utils.setup_distributed`` (launch with ``torchrun``).
- Model/optimiser/step checkpoints saved under
  ``<output.checkpoint_root>/<experiment.exp>/checkpoint_step_<N>/`` — diffusers
  ``save_pretrained`` format so they can be reloaded with ``from_pretrained``.
- ``data.FullSequenceLatentDataset`` provides sliding windows over per-sequence
  ``.npz`` files.
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.loader_video2mesh import FullSequenceLatentDataset
from models.v1.video2mesh.triposg_transformer import TripoSGDiTModel4D
from utils.config_utils import dump_yaml_config, load_yaml_config
from utils.dist_utils import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
)
from utils.logger import logger

# Rectified-flow scheduler ships with the TripoSG third-party dependency. Users
# must export PYTHONPATH=...:./TripoSG (see train.sh) before launching.
from TripoSG.triposg.schedulers.scheduling_rectified_flow import (
    RectifiedFlowScheduler,
    compute_density_for_timestep_sampling,
    compute_loss_weighting,
)

PROCESS_NAME = "video2mesh"


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------
def find_latest_step_ckpt(root: str) -> Optional[str]:
    """Return the ``checkpoint_step_<N>`` subdir with the largest N, if any."""
    if not os.path.isdir(root):
        return None
    best_step, best_path = -1, None
    for name in os.listdir(root):
        m = re.match(r"checkpoint_step_(\d+)$", name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step, best_path = step, os.path.join(root, name)
    return best_path


def auto_resume_and_load(model, optimizer, base_dir: str, device, is_lora: bool) -> int:
    """Restore model + optimizer + global_step from the latest step checkpoint."""
    ckpt_dir = find_latest_step_ckpt(base_dir)
    if ckpt_dir is None:
        logger.warning(f"[resume] No checkpoint_step_<N> found under {base_dir}; start fresh.")
        return 0

    logger.info(f"[resume] Loading checkpoint: {ckpt_dir}")

    adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    full_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors")

    if is_lora and os.path.exists(adapter_path):
        model.load_adapter(ckpt_dir, "default", is_trainable=True)
        logger.info(f"[resume] Loaded LoRA adapter: {ckpt_dir}")
    elif os.path.exists(full_path):
        from safetensors.torch import load_file

        state_dict = load_file(full_path, device=str(device))
        model.load_state_dict(state_dict)
        logger.info(f"[resume] Loaded full weights: {full_path}")
    else:
        raise FileNotFoundError(f"No model weights in {ckpt_dir}")

    opt_path = os.path.join(ckpt_dir, "optimizer.pt")
    if os.path.exists(opt_path):
        opt_ckpt = torch.load(opt_path, map_location=device)
        optimizer.load_state_dict(opt_ckpt["optimizer"])
        step = int(opt_ckpt.get("step", 0))
        logger.info(f"[resume] Optimizer restored, global_step={step}")
        return step

    logger.warning("[resume] optimizer.pt missing; global_step reset to 0")
    return 0


def save_checkpoint(model, optimizer, base_dir: str, global_step: int) -> str:
    """Save model (diffusers format) + optimizer state + step into a new subdir."""
    base_model = model.module if hasattr(model, "module") else model
    step_dir = os.path.join(base_dir, f"checkpoint_step_{global_step}")
    os.makedirs(step_dir, exist_ok=True)

    base_model.save_pretrained(step_dir)
    torch.save({"optimizer": optimizer.state_dict(), "step": global_step},
               os.path.join(step_dir, "optimizer.pt"))
    logger.info(f"[ckpt] Saved step {global_step} -> {step_dir}")
    return step_dir


@torch.inference_mode()
def merge_lora_to_full_weights_safely(
    step_dir: str,
    base_ckpt_dir: str,
    save_subdir: str = "transformer",
):
    """Merge LoRA adapters in ``step_dir`` into a fresh base on CPU and save.

    We load the pretrained base separately (not the potentially-wrapped model
    from training) so the PEFT merge is numerically clean.
    """
    from peft import PeftModel  # deferred: peft is only needed when LoRA is used

    base = TripoSGDiTModel4D.from_pretrained(
        base_ckpt_dir, torch_dtype=torch.float32, device_map="cpu"
    )
    peft_model = PeftModel.from_pretrained(base, step_dir, is_trainable=False)
    merged = peft_model.merge_and_unload()
    out_dir = os.path.join(step_dir, save_subdir)
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir)
    logger.info(f"[merge] Wrote merged LoRA -> {out_dir}")


# ---------------------------------------------------------------------------
# LoRA target collection
# ---------------------------------------------------------------------------
def collect_lora_targets(model, target_mode: str):
    """Return the list of module names to attach LoRA to, block by block."""
    # ``LoRACompatibleLinear`` lives in older diffusers versions; tolerate absence.
    try:
        from diffusers.models.lora import LoRACompatibleLinear
        linear_types = (torch.nn.Linear, LoRACompatibleLinear)
    except Exception:
        linear_types = (torch.nn.Linear,)

    all_linear_names = [
        n for n, m in model.named_modules()
        if isinstance(m, linear_types) and "temporal_block" not in n
    ]
    all_set = set(all_linear_names)
    targets = []

    for i in range(len(model.blocks)):
        for attn_idx in (1, 2):
            for proj in ("to_q", "to_k", "to_v"):
                name = f"blocks.{i}.attn{attn_idx}.{proj}"
                if name in all_set:
                    targets.append(name)
            if target_mode == "all":
                out_name = f"blocks.{i}.attn{attn_idx}.to_out.0"
                if out_name in all_set:
                    targets.append(out_name)
        if target_mode == "all":
            for name in (
                f"blocks.{i}.ff.net.0.proj",
                f"blocks.{i}.ff.net.2",
                f"blocks.{i}.skip_linear",
            ):
                if name in all_set:
                    targets.append(name)

    if target_mode == "all":
        for extra in ("time_proj.linear_1", "time_proj.linear_2", "proj_in", "proj_out"):
            if extra in all_set:
                targets.append(extra)
    return targets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_video2mesh(cfg):
    distributed, rank, world_size, local_rank = setup_distributed()

    runtime_cfg = cfg["runtime"]
    exp_cfg = cfg["experiment"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    output_cfg = cfg["output"]
    lora_cfg = cfg.get("lora", {}) or {}

    torch.manual_seed(runtime_cfg.get("seed", 42))
    np.random.seed(runtime_cfg.get("seed", 42))

    device_str = runtime_cfg.get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    logger.info(f"[runtime] device={device}")

    base_dir = os.path.join(output_cfg["checkpoint_root"], exp_cfg["exp"])
    os.makedirs(base_dir, exist_ok=True)
    dump_yaml_config(cfg, os.path.join(base_dir, "config.yaml"))

    writer = None
    if is_main_process():
        log_dir = os.path.join(base_dir, "logs",
                               f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"[runtime] tensorboard -> {log_dir}")

    # ---- dataset ----
    attention_design = dict(model_cfg["attention_kwargs"])
    dataset = FullSequenceLatentDataset(
        npz_dirs=data_cfg["npz_dirs"],
        seq_len=attention_design["seq_len"],
        image_key=data_cfg.get("image_key", "image_embed"),
        latent_key=data_cfg.get("latent_key", "latent"),
        frame_step=data_cfg.get("frame_step", 1),
        hop=data_cfg.get("hop", attention_design["seq_len"]),
        drop_last=data_cfg.get("drop_last", True),
        preload=data_cfg.get("preload", False),
        mmap_mode=data_cfg.get("mmap_mode", "r"),
        num_threads=data_cfg.get("num_threads", 32),
        skip_json=data_cfg.get("skip_json"),
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(sampler is None),
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
        sampler=sampler,
    )

    # ---- model ----
    pretrained_model_path = model_cfg["pretrained_model_path"]
    logger.info(f"[model] Loading TripoSGDiTModel4D from {pretrained_model_path}")
    model = TripoSGDiTModel4D.from_pretrained(pretrained_model_path)

    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing = True
        logger.info("[model] gradient checkpointing enabled")

    # ---- LoRA (optional) ----
    use_lora = bool(lora_cfg.get("use_lora", False))
    if use_lora:
        for p in model.parameters():
            p.requires_grad = False
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = collect_lora_targets(model, lora_cfg.get("target_mode", "all"))
        if is_main_process():
            logger.info(f"[lora] {len(target_modules)} target modules "
                        f"(mode={lora_cfg.get('target_mode', 'all')})")

        lora_config = LoraConfig(
            r=lora_cfg.get("rank", 4),
            lora_alpha=lora_cfg.get("alpha", lora_cfg.get("rank", 4)),
            target_modules=target_modules,
            lora_dropout=lora_cfg.get("dropout", 0.0),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, lora_config)
        if is_main_process():
            model.print_trainable_parameters()

    model.to(device)

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    # ---- resume ----
    global_step = 0
    if train_cfg.get("resume", False):
        global_step = auto_resume_and_load(model, optimizer, base_dir, device, is_lora=use_lora)

    # ---- AMP ----
    use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    amp_dtype = torch.bfloat16 if train_cfg.get("amp_dtype", "bf16") in ("bf16", "bfloat16") else torch.float16

    # ---- scheduler ----
    with open(model_cfg["scheduler_config_path"], "r") as f:
        sched_config = json.load(f)
    scheduler = RectifiedFlowScheduler(**sched_config)

    # ---- DDP ----
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if is_main_process():
        logger.info("***** Training config *****")
        logger.info(f"  windows            = {len(dataset)}")
        logger.info(f"  epochs             = {train_cfg['num_epochs']}")
        logger.info(f"  batch_size (per)   = {train_cfg['batch_size']}")
        logger.info(f"  grad_accum_steps   = {train_cfg.get('gradient_accumulation_steps', 1)}")
        logger.info(f"  world_size         = {world_size}")
        logger.info(f"  attention_design   = {attention_design}")

    model.train()

    uncond_dropout_prob = train_cfg.get("uncond_dropout_prob", 0.1)
    gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    save_every = train_cfg.get("save_every", 0)
    log_every = train_cfg.get("log_every_n_steps", 50)
    timestep_sampling_weighting = train_cfg.get("timestep_sampling_weighting", "logit_normal")
    loss_weighting_scheme = train_cfg.get("loss_weighting_scheme", "sigma_sqrt")
    logit_mean = train_cfg.get("logit_mean", 0.0)
    logit_std = train_cfg.get("logit_std", 1.0)
    merge_lora_on_save = use_lora and lora_cfg.get("merge_on_save", True)

    # CFG unconditional token cache (shape matches TripoSG image conditioner).
    uncond_S = train_cfg.get("uncond_seq_len", 257)
    uncond_D = train_cfg.get("uncond_dim", 1024)
    uncond_token = torch.zeros(1, 1, uncond_S, uncond_D, device=device)

    debug = runtime_cfg.get("debug", False)

    try:
        for epoch in range(train_cfg["num_epochs"]):
            if distributed:
                sampler.set_epoch(epoch)

            progress = tqdm(
                dataloader,
                desc=f"Epoch {epoch} (rank {rank})",
                disable=not is_main_process(),
                ncols=110,
            )

            for step_idx, (image_feat, latent_gt) in enumerate(progress):
                if debug and step_idx >= 5:
                    break

                image_feat = image_feat.to(device, non_blocking=True)
                latent_gt = latent_gt.to(device, non_blocking=True)
                B, frames = image_feat.shape[:2]

                # CFG dropout: replace entire sample's conditioning with zeros.
                uncond = uncond_token.expand(B, frames, uncond_S, uncond_D)
                mask = torch.rand(B, device=device) < uncond_dropout_prob
                if mask.any():
                    image_feat[mask] = uncond[mask]

                # Timestep / noise / target (rectified flow).
                t_sigmas = compute_density_for_timestep_sampling(
                    weighting_scheme=timestep_sampling_weighting,
                    batch_size=B,
                    logit_mean=logit_mean,
                    logit_std=logit_std,
                ).to(device)
                timesteps = scheduler._sigma_to_t(t_sigmas).long()

                noise = torch.randn_like(latent_gt)
                noisy_latent = scheduler.scale_noise(
                    original_samples=latent_gt,
                    noise=noise,
                    timesteps=timesteps,
                )
                target_velocity = latent_gt - noise

                def _forward_and_loss():
                    out = model(
                        hidden_states=noisy_latent,
                        timestep=timesteps,
                        encoder_hidden_states=image_feat,
                        attention_kwargs=attention_design,
                        return_dict=True,
                    ).sample
                    per = F.mse_loss(out, target_velocity, reduction="none")
                    w = compute_loss_weighting(loss_weighting_scheme, sigmas=t_sigmas)
                    return (per * w.view(B, 1, 1, 1)).mean() / gradient_accumulation_steps

                if use_amp:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        loss = _forward_and_loss()
                    scaler.scale(loss).backward()
                else:
                    loss = _forward_and_loss()
                    loss.backward()

                if (global_step + 1) % gradient_accumulation_steps == 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if is_main_process() and (global_step % log_every == 0):
                    raw_loss = float(loss.item() * gradient_accumulation_steps)
                    lr = optimizer.param_groups[0]["lr"]
                    if writer is not None:
                        writer.add_scalar("train/loss", raw_loss, global_step)
                        writer.add_scalar("train/lr", lr, global_step)
                    logger.info(f"step {global_step} epoch {epoch} loss={raw_loss:.4f} lr={lr:.2e}")
                    progress.set_postfix(loss=raw_loss, lr=lr)

                # --- periodic checkpoint (+ optional LoRA merge) ---
                should_save = save_every > 0 and global_step % save_every == 0
                if should_save and is_main_process():
                    step_dir = save_checkpoint(model, optimizer, base_dir, global_step)
                    if merge_lora_on_save:
                        try:
                            merge_lora_to_full_weights_safely(
                                step_dir=step_dir,
                                base_ckpt_dir=pretrained_model_path,
                                save_subdir="transformer",
                            )
                        except Exception as e:
                            logger.warning(f"[merge] failed: {e}")
                if distributed and should_save:
                    dist.barrier()

        if is_main_process():
            save_checkpoint(model, optimizer, base_dir, global_step)
            if writer is not None:
                writer.close()
            logger.info("Training complete.")

    finally:
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

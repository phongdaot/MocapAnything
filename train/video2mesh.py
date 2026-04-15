
### video2mesh.py ###
# Training script for the video2mesh temporal DiT (TripoSGDiTModel4D).
#
# Port of train_s2_0821_ddp.py (LoRA + RectifiedFlow + CFG dropout) into the
# mocapv2 project style (YAML-driven, shared dist/log utilities).
#
# The dataset must already be preprocessed with preprocess/preprocess_data.py,
# producing per-sample .npz files containing:
#     image_embed: [T, 257, 1024]   frozen DinoV2 embeddings
#     latent:      [T, 2048, 64]    VAE-encoded mesh latents
#
# Only the transformer is trained here (VAE + image encoder are frozen and
# already applied during preprocessing).

import argparse
import copy
import json
import os
import re
from datetime import datetime
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.loader_v1_mesh import FullSequenceLatentDataset
from models.v1.video2mesh.triposg_transformer import TripoSGDiTModel4D
from utils.common import set_seed
from utils.config_utils import load_yaml_config, dump_yaml_config
from utils.dist_utils import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
)
from utils.logger import logger

# RectifiedFlow scheduler comes from the TripoSG package (same as preprocess/).
from TripoSG.triposg.schedulers.scheduling_rectified_flow import (
    RectifiedFlowScheduler,
    compute_density_for_timestep_sampling,
    compute_loss_weighting,
)

try:
    from diffusers.models.lora import LoRACompatibleLinear
except Exception:  # diffusers moved this around across versions
    LoRACompatibleLinear = None

torch.multiprocessing.set_start_method("spawn", force=True)

PROCESS_NAME = "video2mesh"


# =========================================================
# checkpoint io
# =========================================================
def _find_latest_step_ckpt(root: str) -> Optional[str]:
    if not os.path.isdir(root):
        return None
    max_step = -1
    latest = None
    for name in os.listdir(root):
        m = re.match(r"checkpoint_step_(\d+)", name)
        if m:
            step = int(m.group(1))
            if step > max_step:
                max_step = step
                latest = os.path.join(root, name)
    return latest


def auto_resume_and_load(model, optimizer, output_dir, device, is_lora=False):
    """Restore the latest checkpoint_step_{N} under ``output_dir``."""
    ckpt_dir = _find_latest_step_ckpt(output_dir)
    if ckpt_dir is None:
        if is_main_process():
            logger.info(f"[resume] no checkpoint under {output_dir}, starting from step 0")
        return 0

    if is_main_process():
        logger.info(f"[resume] loading {ckpt_dir}")

    adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
    full_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors")

    if is_lora and os.path.exists(adapter_path):
        model.load_adapter(ckpt_dir, "default", is_trainable=True)
        if is_main_process():
            logger.info(f"[resume] loaded LoRA adapter from {ckpt_dir}")
    elif os.path.exists(full_path):
        from safetensors.torch import load_file
        device_str = str(device)
        state_dict = load_file(full_path, device=device_str)
        model.load_state_dict(state_dict)
        if is_main_process():
            logger.info(f"[resume] loaded full transformer weights from {ckpt_dir}")
    else:
        raise FileNotFoundError(f"no transformer weights under {ckpt_dir}")

    optimizer_path = os.path.join(ckpt_dir, "optimizer.pt")
    if os.path.exists(optimizer_path):
        opt_ckpt = torch.load(optimizer_path, map_location=device)
        optimizer.load_state_dict(opt_ckpt["optimizer"])
        step = int(opt_ckpt.get("step", 0))
        if is_main_process():
            logger.info(f"[resume] restored optimizer, global_step={step}")
        return step

    if is_main_process():
        logger.warning(f"[resume] no optimizer.pt in {ckpt_dir}, global_step=0")
    return 0


def save_checkpoint(model, optimizer, output_dir, global_step):
    base_model = model.module if hasattr(model, "module") else model
    step_dir = os.path.join(output_dir, f"checkpoint_step_{global_step}")
    os.makedirs(step_dir, exist_ok=True)
    base_model.save_pretrained(step_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": global_step},
        os.path.join(step_dir, "optimizer.pt"),
    )
    logger.info(f"[ckpt] saved step {global_step} -> {step_dir}")
    return step_dir


@torch.inference_mode()
def merge_lora_to_full_weights_safely(
    step_dir: str,
    base_ckpt_dir: str,
    save_subdir: str = "transformer",
):
    """CPU-merge LoRA adapter back into a clean base, save full weights."""
    from peft import PeftModel

    cpu_base = TripoSGDiTModel4D.from_pretrained(
        base_ckpt_dir, torch_dtype=torch.float32, device_map="cpu"
    )
    cpu_peft = PeftModel.from_pretrained(cpu_base, step_dir, is_trainable=False)
    merged = cpu_peft.merge_and_unload()
    out_dir = os.path.join(step_dir, save_subdir)
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir)
    logger.info(f"[merge] merged LoRA on CPU -> {out_dir}")


def cleanup_old_step_ckpts(output_dir: str, max_ckpt: int):
    if max_ckpt is None or max_ckpt <= 0:
        return
    if not os.path.isdir(output_dir):
        return
    entries = []
    for name in os.listdir(output_dir):
        m = re.match(r"checkpoint_step_(\d+)$", name)
        if m:
            entries.append((int(m.group(1)), os.path.join(output_dir, name)))
    entries.sort()
    to_delete = entries[: max(0, len(entries) - max_ckpt)]
    import shutil
    for step, path in to_delete:
        try:
            shutil.rmtree(path)
            logger.info(f"[ckpt] removed old {path}")
        except Exception as e:
            logger.warning(f"[ckpt] failed to remove {path}: {e}")


# =========================================================
# LoRA helpers
# =========================================================
def build_lora_target_modules(model, lora_target_mode: str) -> List[str]:
    """Pick LoRA target module names on the TripoSG DiT blocks."""
    all_linear_names: List[str] = []
    for name, module in model.named_modules():
        is_linear = isinstance(module, torch.nn.Linear)
        if LoRACompatibleLinear is not None and isinstance(module, LoRACompatibleLinear):
            is_linear = True
        if is_linear and "temporal_block" not in name:
            all_linear_names.append(name)

    final: List[str] = []
    num_blocks = len(model.blocks)
    for i in range(num_blocks):
        for attn_idx in (1, 2):
            q = f"blocks.{i}.attn{attn_idx}.to_q"
            k = f"blocks.{i}.attn{attn_idx}.to_k"
            v = f"blocks.{i}.attn{attn_idx}.to_v"
            o = f"blocks.{i}.attn{attn_idx}.to_out.0"
            if q in all_linear_names: final.append(q)
            if k in all_linear_names: final.append(k)
            if v in all_linear_names: final.append(v)
            if lora_target_mode == "all" and o in all_linear_names:
                final.append(o)
        if lora_target_mode == "all":
            ff1 = f"blocks.{i}.ff.net.0.proj"
            ff2 = f"blocks.{i}.ff.net.2"
            skip_linear = f"blocks.{i}.skip_linear"
            if ff1 in all_linear_names: final.append(ff1)
            if ff2 in all_linear_names: final.append(ff2)
            if skip_linear in all_linear_names: final.append(skip_linear)

    if lora_target_mode == "all":
        for extra in ("time_proj.linear_1", "time_proj.linear_2", "proj_in", "proj_out"):
            if extra in all_linear_names:
                final.append(extra)
    return final


def wrap_with_lora(model, lora_cfg):
    from peft import LoraConfig, TaskType, get_peft_model

    for p in model.parameters():
        p.requires_grad = False

    targets = build_lora_target_modules(model, lora_cfg.get("target_mode", "all"))
    if is_main_process():
        logger.info(f"[lora] target modules ({len(targets)}): {targets}")

    cfg = LoraConfig(
        r=lora_cfg.get("rank", 4),
        lora_alpha=lora_cfg.get("alpha", lora_cfg.get("rank", 4)),
        target_modules=targets,
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, cfg)
    if is_main_process():
        model.print_trainable_parameters()
    return model


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
    lora_cfg = cfg.get("lora", {"enable": False})
    output_cfg = cfg["output"]

    set_seed(runtime_cfg.get("seed", 42))

    base_dir = os.path.join(output_cfg["checkpoint_root"], exp_cfg["exp"])
    os.makedirs(base_dir, exist_ok=True)

    if is_main_process():
        dump_yaml_config(cfg, os.path.join(base_dir, "config.yaml"))

    if is_main_process():
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(base_dir, "logs", f"run_{run_id}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard log dir: {log_dir}")
    else:
        writer = None

    device_str = runtime_cfg.get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if is_main_process():
        logger.info(f"device: {device}")

    # ---- data ----
    seq_len = int(data_cfg["seq_len"])
    dataset = FullSequenceLatentDataset(
        npz_dirs=data_cfg["train_dirs"],
        seq_len=seq_len,
        frame_step=data_cfg.get("frame_step", 1),
        hop=data_cfg.get("hop", 1),
        drop_last=data_cfg.get("drop_last", True),
        preload=data_cfg.get("preload", False),
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
        drop_last=True,
    )

    # ---- model ----
    if is_main_process():
        logger.info(f"loading transformer from {model_cfg['pretrained_path']}")
    model = TripoSGDiTModel4D.from_pretrained(model_cfg["pretrained_path"])

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing = True
        if is_main_process():
            logger.info("gradient checkpointing enabled")

    # ---- LoRA wrap (optional) ----
    if lora_cfg.get("enable", False):
        model = wrap_with_lora(model, lora_cfg)

    model.to(device)

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    global_step = 0
    if train_cfg.get("resume", False):
        global_step = auto_resume_and_load(
            model, optimizer, base_dir, device, is_lora=lora_cfg.get("enable", False)
        )

    if is_main_process():
        trainable_count = sum(1 for _, p in model.named_parameters() if p.requires_grad)
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"trainable param tensors: {trainable_count}")
        logger.info(f"trainable param count:   {trainable_num:,}")

    # ---- AMP ----
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    amp_dtype_str = runtime_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }.get(amp_dtype_str, torch.bfloat16)
    # GradScaler is only needed for fp16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # ---- scheduler ----
    with open(train_cfg["scheduler_config_path"], "r") as f:
        sched_config = json.load(f)
    scheduler = RectifiedFlowScheduler(**sched_config)

    # ---- DDP wrap ----
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=train_cfg.get("find_unused_parameters", False),
        )

    # ---- attention design ----
    attention_design = dict(model_cfg["attention_kwargs"])
    attention_design["seq_len"] = seq_len

    if is_main_process():
        logger.info("***** training *****")
        logger.info(f"  windows: {len(dataset)}")
        logger.info(f"  epochs : {train_cfg['num_epochs']}")
        logger.info(f"  bs/gpu : {train_cfg['batch_size']}")
        logger.info(f"  accum  : {train_cfg.get('gradient_accumulation_steps', 1)}")
        logger.info(f"  world  : {world_size}")
        logger.info(f"  attn   : {attention_design}")

    uncond_dropout_prob = train_cfg.get("uncond_dropout_prob", 0.1)
    t_weighting = train_cfg.get("timestep_sampling_weighting", "logit_normal")
    loss_weighting_scheme = train_cfg.get("loss_weighting_scheme", "sigma_sqrt")
    logit_mean = train_cfg.get("logit_mean", 0.0)
    logit_std = train_cfg.get("logit_std", 1.0)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    save_every = train_cfg.get("save_every", 0)
    log_every = train_cfg.get("log_every", 10)
    max_ckpt = train_cfg.get("max_ckpt", 0)
    debug = runtime_cfg.get("debug", False)

    model.train()

    try:
        for epoch in range(train_cfg["num_epochs"]):
            if distributed and sampler is not None:
                sampler.set_epoch(epoch)

            progress = tqdm(
                dataloader,
                desc=f"Epoch {epoch} (rank {rank})",
                disable=not is_main_process(),
                ncols=120,
            )

            step_in_epoch = 0
            for batch in progress:
                step_in_epoch += 1
                if debug and step_in_epoch >= 10:
                    break

                image_feat, latent_gt = batch
                image_feat = image_feat.to(device, non_blocking=True)
                latent_gt = latent_gt.to(device, non_blocking=True)
                B, frames = image_feat.shape[:2]

                # CFG unconditional dropout (replace cond with zeros)
                if uncond_dropout_prob > 0:
                    uncond_mask = torch.rand(B, device=device) < uncond_dropout_prob
                    if uncond_mask.any():
                        image_feat[uncond_mask] = 0.0

                # sample timesteps via density from the RF scheduler
                t_sigmas = compute_density_for_timestep_sampling(
                    weighting_scheme=t_weighting,
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

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    model_output = model(
                        hidden_states=noisy_latent,
                        timestep=timesteps,
                        encoder_hidden_states=image_feat,
                        attention_kwargs=attention_design,
                        return_dict=True,
                    ).sample

                    loss = F.mse_loss(
                        model_output.float(), target_velocity.float(), reduction="none"
                    )
                    loss_weights = compute_loss_weighting(
                        loss_weighting_scheme, sigmas=t_sigmas
                    )
                    loss = (loss * loss_weights.view(B, 1, 1, 1)).mean()
                    loss = loss / grad_accum

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (global_step + 1) % grad_accum == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if is_main_process() and (global_step % log_every == 0):
                    raw_loss_val = float(loss.item()) * grad_accum
                    if writer is not None:
                        writer.add_scalar("Loss/train", raw_loss_val, global_step)
                        writer.add_scalar(
                            "LR", optimizer.param_groups[0]["lr"], global_step
                        )
                    logger.info(
                        f"step {global_step} epoch {epoch} loss={raw_loss_val:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

                should_save = (save_every > 0) and (global_step % save_every == 0)
                if should_save and is_main_process():
                    step_dir = save_checkpoint(model, optimizer, base_dir, global_step)
                    if lora_cfg.get("enable", False) and lora_cfg.get("merge_on_save", True):
                        merge_lora_to_full_weights_safely(
                            step_dir=step_dir,
                            base_ckpt_dir=model_cfg["pretrained_path"],
                            save_subdir="transformer",
                        )
                    cleanup_old_step_ckpts(base_dir, max_ckpt)
                if distributed and should_save:
                    dist.barrier()

                if is_main_process():
                    progress.set_postfix(
                        loss=float(loss.item()) * grad_accum,
                        lr=optimizer.param_groups[0]["lr"],
                    )

        if is_main_process():
            save_checkpoint(model, optimizer, base_dir, global_step)
            if writer is not None:
                writer.close()
            logger.info("training done.")

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

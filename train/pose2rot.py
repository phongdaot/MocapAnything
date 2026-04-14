
### pose2rot.py ###
import os
import sys
import time
import random
import subprocess
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded
from utils.config_utils import instantiate_from_config
from utils.common import set_seed
from utils.rotation import bvh_forward, rot6d_to_fk_positions, rot6d_to_rotmat_tensor
from utils.dist_utils import *
from utils.loss import compute_rot_loss, get_loss_fn, masked_loss, rot6d_vel_loss, rot6d_acc_loss, angle_L1, angle_velocity_L1
from utils.train_utils import *
from utils.config_utils import load_yaml_config, dump_yaml_config
import argparse
torch.multiprocessing.set_start_method("spawn", force=True)


PROCESS_NAME="pose2rot"
MAX_JOINTS = 150

def evaluate_batch_metrics(model_out, batch):
    joint_mask = batch["joint_mask"].bool()
    static_rot_mask = batch["static_rot_joint_mask"].bool()
    static_pos_mask = batch["static_pos_joint_mask"].bool()

    rot_joint_mask_for_loss = joint_mask & (~static_rot_mask)
    pos_joint_mask_for_loss = joint_mask & (~static_pos_mask)

    gt_rot6d = batch["rot6d_a"]
    target_pose = batch["position"]

    parents = batch["parent_a"]
    offsets = batch["offset_a"]
    global_scales = batch["global_scale"]

    pred_rot6d = model_out["pred_rot6d"]

    metrics = {}
    metrics["rot_l1"] = masked_loss(pred_rot6d, gt_rot6d, rot_joint_mask_for_loss, nn.L1Loss(reduction='none'))
    metrics["rot_l2"] = masked_loss(pred_rot6d, gt_rot6d, rot_joint_mask_for_loss, nn.MSELoss(reduction='none'))

    rot_mask_np = rot_joint_mask_for_loss.detach().cpu().numpy()
    metrics["angle_l1"] = angle_L1(pred_rot6d, gt_rot6d, mask=rot_mask_np)
    metrics["anglevel_l1"] = angle_velocity_L1(pred_rot6d, gt_rot6d, mask=rot_mask_np)

    pred_pos = rot6d_to_fk_positions(pred_rot6d, offsets, parents, global_scales)
    metrics["fk_l1"] = masked_loss(
        pred_pos,
        target_pose,
        pos_joint_mask_for_loss,
        nn.L1Loss(reduction='none')
    ) * 3

    root_mask = torch.zeros_like(joint_mask)
    root_mask[:, 0] = True
    root_mask_np = root_mask.detach().cpu().numpy()
    metrics["root_rot_l1"] = masked_loss(pred_rot6d, gt_rot6d, root_mask, nn.L1Loss(reduction='none'))
    metrics["root_angle_l1"] = angle_L1(pred_rot6d, gt_rot6d, mask=root_mask_np)

    return metrics


def run_evaluation(
    model,
    test_loader,
    device,
    epoch,
    cfg,
    base_dir,
    writer=None,
):
    model.eval()
    metric_sums = {
        "rot_l1": 0.0,
        "rot_l2": 0.0,
        "angle_l1": 0.0,
        "anglevel_l1": 0.0,
        "fk_l1": 0.0,
        "root_rot_l1": 0.0,
        "root_angle_l1": 0.0,
    }

    sample_count = 0
    
    with torch.no_grad():
        test_loader_tqdm = tqdm(
            test_loader,
            total=len(test_loader),
            desc="Eval",
            ncols=120,
        ) if is_main_process() else test_loader
        
        vis_save_dir = os.path.join(base_dir, f"vis_compare_epoch{epoch + 1}")
        if is_main_process() and (epoch + 1) % cfg["train"]["vis_every"] == 0:
            os.makedirs(vis_save_dir, exist_ok=True)
        vis_done_species = set()
        max_vis_species = 60
        
        cnt = 0
        for bi, batch in enumerate(test_loader_tqdm):
            cnt += 1
            if cfg["runtime"]["debug"] and cnt >= 10:
                break
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            model_out = model(batch)
            metrics = evaluate_batch_metrics(model_out, batch)

            bs = batch["rot6d_a"].size(0)
            sample_count += bs

            for k in metric_sums:
                metric_sums[k] += float(metrics[k]) * bs

                if is_main_process() and (epoch + 1) % cfg["train"]["vis_every"] == 0:
                    species_list = batch["species"]
                    for si, species_name in enumerate(species_list):
                        if len(vis_done_species) >= max_vis_species:
                            break
                        if species_name in vis_done_species:
                            continue

                        vis_done_species.add(species_name)

                        pred_np = model_out["pred_rot6d"].detach().cpu().numpy()[si]
                        gt_np = batch["rot6d_a"].detach().cpu().numpy()[si]

                        npy_pred_path = os.path.join(vis_save_dir, f"{species_name}_pred.npy")
                        npy_gt_path = os.path.join(vis_save_dir, f"{species_name}_gt.npy")
                        np.save(npy_pred_path, pred_np)
                        np.save(npy_gt_path, gt_np)

                        pred_command = [
                            sys.executable, "utils/npy2bvh.py",
                            "--npy_path", npy_pred_path,
                            "--species_name", species_name
                        ]
                        gt_command = [
                            sys.executable, "utils/npy2bvh.py",
                            "--npy_path", npy_gt_path,
                            "--species_name", species_name
                        ]

                        try:
                            subprocess.run(pred_command, capture_output=True, text=True)
                            subprocess.run(gt_command, capture_output=True, text=True)
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Error running npy2bvh for {species_name}: {e}")
                        logger.info(f"[VIS] Saved BVH for {species_name} at {npy_gt_path}")

    sample_count = reduce_sum_scalar(sample_count, device)
    
    for k in metric_sums:
        metric_sums[k] = reduce_sum_scalar(metric_sums[k], device) / max(sample_count, 1)
            
    if is_main_process():
        print(
            f"[Eval] "
            f"rot_l1={metric_sums['rot_l1']:.6f}, "
            f"rot_l2={metric_sums['rot_l2']:.6f}, "
            f"angle_l1={metric_sums['angle_l1']:.2f}, "
            f"anglevel_l1={metric_sums['anglevel_l1']:.2f}, "
            f"fk_l1={metric_sums['fk_l1']:.6f}, "
            f"root_rot_l1={metric_sums['root_rot_l1']:.6f}, "
            f"root_angle_l1={metric_sums['root_angle_l1']:.2f}"
        )

        if writer is not None:
            for k, v in metric_sums.items():
                writer.add_scalar(f"test/{k}", v, epoch + 1)

    return metric_sums

def train_pose2rot(cfg):
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
    
    # Copy config to checkpoint dir for record
    if is_main_process():
        config_save_path = os.path.join(base_dir, "config.yaml")
        dump_yaml_config(cfg, config_save_path)
    
    logdir = os.path.join(base_dir, "logs_pose2rot")

    if is_main_process():
        logger.info(f"Log directory: {logdir}")

    device_str = runtime_cfg.get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    dataset_train = AnySpeciesPoseDataset(
        bvh_dir=data_cfg["bvh_dir"],
        window=data_cfg["seq_len"],
        mmap=data_cfg.get("mmap", True),
        cache_scale=data_cfg.get("cache_scale", True),
        limit_species_debug=data_cfg.get("limit_species_debug", []),
        split_json=data_cfg["split_json"],
        split_mode="train",
        memory_pkl_path=data_cfg.get("train_memory_pkl_path", None),
        preload_all=False
    )

    dataset_test = AnySpeciesPoseDataset(
        bvh_dir=data_cfg["bvh_dir"],
        window=data_cfg["seq_len"],
        mmap=data_cfg.get("mmap", True),
        cache_scale=data_cfg.get("cache_scale", True),
        limit_species_debug=data_cfg.get("limit_species_debug", []),
        split_json=data_cfg["split_json"],
        split_mode="test",
        memory_pkl_path=data_cfg.get("test_memory_pkl_path", data_cfg.get("train_memory_pkl_path", None)),
        preload_all=False
    )

    sampler_train = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    sampler_test = DistributedSampler(dataset_test, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        dataset_train,
        batch_size=train_cfg["batch_size"],
        sampler=sampler_train,
        shuffle=(sampler_train is None),
        num_workers=train_cfg.get("num_workers_train", 4),
        collate_fn=collate_anyspecies_padded,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=1,
        sampler=sampler_test,
        shuffle=False,
        num_workers=eval_cfg.get("num_workers", 2),
        collate_fn=collate_anyspecies_padded,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )


    model: torch.nn.Module = instantiate_from_config(model_cfg)
    model = model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 0.0)

    if weight_decay == 0.0:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    rot_fn = get_loss_fn(train_cfg["loss"].get("rot_loss_type", "smooth_l1"))
    vel_fn = get_loss_fn(train_cfg["loss"].get("vel_loss_type", "smooth_l1"))
    acc_fn = get_loss_fn(train_cfg["loss"].get("acc_loss_type", "smooth_l1"))

    writer = SummaryWriter(logdir) if is_main_process() else None

    if is_main_process():
        logger.info("=" * 60)
        logger.info(model)
        logger.info("=" * 60)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info("=" * 60)

    pretrain_ckpt = train_cfg.get("pretrain_ckpt")
    if pretrain_ckpt is not None and is_main_process():
        load_partial_pretrain(model.module if distributed else model, pretrain_ckpt)

    best_test_loss = float("inf")
    best_ckpt_path = os.path.join(base_dir, f"{PROCESS_NAME}_ckpt_best.pt")
    start_epoch = 0

    ckpt_path_latest = find_latest_ckpt(base_dir, PROCESS_NAME)
    if ckpt_path_latest and os.path.exists(ckpt_path_latest):
        if is_main_process():
            logger.info(f"Loading checkpoint: {ckpt_path_latest}")
        start_epoch = load_checkpoint(
            model.module if distributed else model,
            optimizer,
            ckpt_path_latest,
            device
        )
        if is_main_process():
            logger.info(f"Resumed from epoch {start_epoch}")

    scaler = torch.amp.GradScaler()

    global_step = 0
    epochs = train_cfg["epochs"]
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)

    for epoch in range(start_epoch, epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        loader_tqdm = (
            tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{epochs} [train][rank{rank}]",
                ncols=100
            ) if is_main_process() else enumerate(train_loader)
        )

        batch_times = []
        optimizer.zero_grad(set_to_none=True)
        
        cnt = 0
        for i, batch in loader_tqdm:
            cnt += 1
            if cfg["runtime"]["debug"] and cnt >= 10: break
            batch_start_time = time.time()
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                model_out = model(batch)
                total_loss, loss_dict = compute_rot_loss(
                    model_out=model_out,
                    batch=batch,
                    weight_cfg=train_cfg["weight"],
                    rot_criterion=rot_fn,
                    vel_criterion=vel_fn,
                    acc_criterion=acc_fn,
                )

            raw_loss = total_loss
            loss = raw_loss / grad_accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += raw_loss.item() * batch["rot6d_a"].size(0)

            if writer is not None and global_step % 10 == 0:
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        if k in ["loss_rot", "loss_fk", "loss_root"]:
                            writer.add_scalar(f"train_main/{k}", v.item(), global_step)
                        elif k in ["loss_vel", "loss_acc"]:
                            writer.add_scalar(f"train_temporal/{k}", v.item(), global_step)
                        else:
                            writer.add_scalar(f"train_others/{k}", v.item(), global_step)

            global_step += 1

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            if is_main_process() and i % 10 == 0 and i > 0:
                avg_batch = sum(batch_times) / len(batch_times)
                remaining = avg_batch * (len(train_loader) - i - 1)
                loader_tqdm.set_postfix_str(f"Loss={raw_loss.item():.4f} | ETA={remaining / 60:.1f}min")

        train_loss = running_loss / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start_time

        if is_main_process():
            logger.info(f"Epoch {epoch + 1}: train loss={train_loss:.6f} | time {epoch_time:.1f}s")

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)

        test_every = train_cfg.get("test_every", 1)
        if (epoch + 1) % test_every == 0 or (epoch + 1) == epochs:
            eval_metrics = run_evaluation(
                model=model,
                test_loader=test_loader,
                device=device,
                epoch=epoch,
                cfg=cfg,
                base_dir=base_dir,
                writer=writer,
            )

        if is_main_process():
            save_checkpoint_with_epoch(
                model.module if distributed else model,
                PROCESS_NAME,
                optimizer,
                epoch + 1,
                base_dir
            )
            cleanup_old_checkpoints(base_dir, PROCESS_NAME, train_cfg.get("max_ckpt", 100))

            if eval_metrics["rot_l1"] < best_test_loss:
                best_test_loss = eval_metrics["rot_l1"]
                torch.save({
                    "model_state": (model.module if distributed else model).state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_test_loss": best_test_loss,
                }, best_ckpt_path)
                logger.info(f"New best checkpoint saved: {best_ckpt_path} (test_regular_loss={best_test_loss:.6f})")

    if writer is not None:
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Pose2Rot")
    parser.add_argument("--config", type=str, default="configs/train/train_pose2rot.yaml", help="Path to the YAML config file")
    args = parser.parse_args()
    
    cfg = load_yaml_config(args.config)
    
    train_pose2rot(cfg)
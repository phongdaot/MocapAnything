### video2pose2rot_v2pv3.py ###
from train.pose2rot import PROCESS_NAME
from utils.dist_utils import setup_distributed, is_main_process, cleanup_distributed
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
from utils.loss import *
from utils.npy2bvh import convert_npy_to_bvh
from utils.train_utils import *
from utils.config_utils import load_yaml_config, dump_yaml_config
import argparse
torch.multiprocessing.set_start_method("spawn", force=True)
from .video2pose_v3 import build_test_dataloaders
import math

PROCESS_NAME = "video2pose2rot_v2pv3"
MAX_JOINTS = 150

def get_pose_mix_prob(
    epoch: int,
    total_epochs: int,
    mode: str,
    start_prob: float,
    end_prob: float,
    warmup_epochs: int,
    schedule: str = "linear",
) -> float:
    mode = mode.lower()
    if mode == "gt":
        return 0.0
    if mode == "pred":
        return 1.0

    assert mode == "mix", f"Unknown pose_source_mode: {mode}"
    start_prob = float(np.clip(start_prob, 0.0, 1.0))
    end_prob = float(np.clip(end_prob, 0.0, 1.0))

    if warmup_epochs <= 0:
        return end_prob

    progress = min(max(epoch / warmup_epochs, 0.0), 1.0)

    if schedule == "linear":
        prob = start_prob + (end_prob - start_prob) * progress
    elif schedule == "cosine":
        alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        prob = start_prob + (end_prob - start_prob) * alpha
    else:
        raise ValueError(f"Unknown pose_mix_schedule: {schedule}")

    return float(np.clip(prob, 0.0, 1.0))

def evaluate_pose_metrics(model_out: Dict[str, Any], batch: Dict[str, Any]):
    pred_position = model_out["pred_position"]
    gt_position = batch["position"]

    joint_mask = batch["joint_mask"].bool()
    static_pos_mask = batch["static_pos_joint_mask"].bool()
    pos_joint_mask = joint_mask & (~static_pos_mask)

    metrics = {}
    metrics["pose_l1"] = masked_loss(pred_position, gt_position, pos_joint_mask, nn.L1Loss(reduction="none"))
    metrics["pose_l2"] = masked_loss(pred_position, gt_position, pos_joint_mask, nn.MSELoss(reduction="none"))
    metrics["pose_mpjpe"] = masked_mpjpe(pred_position, gt_position, pos_joint_mask)

    if pred_position.size(1) > 1:
        metrics["pose_mpjve"] = masked_mpjve(pred_position, gt_position, pos_joint_mask)
        pred_diff = pred_position[:, 1:] - pred_position[:, :-1]
        gt_diff = gt_position[:, 1:] - gt_position[:, :-1]
        pos_joint_mask_t = pos_joint_mask.unsqueeze(1).expand(-1, pred_diff.size(1), -1)
        metrics["pose_speed_l1"] = masked_loss(pred_diff, gt_diff, pos_joint_mask_t, nn.L1Loss(reduction="none"))
        metrics["pose_speed_l2"] = masked_loss(pred_diff, gt_diff, pos_joint_mask_t, nn.MSELoss(reduction="none"))
    else:
        zero = torch.tensor(0.0, device=pred_position.device)
        metrics["pose_mpjve"] = zero
        metrics["pose_speed_l1"] = zero
        metrics["pose_speed_l2"] = zero

    return metrics


def evaluate_rot_metrics(model_out: Dict[str, Any], batch: Dict[str, Any]):
    pred_rot6d = model_out["pred_rot6d"]
    gt_rot6d = batch["rot6d_a"]

    joint_mask = batch["joint_mask"].bool()
    static_rot_mask = batch["static_rot_joint_mask"].bool()
    static_pos_mask = batch["static_pos_joint_mask"].bool()

    rot_joint_mask = joint_mask & (~static_rot_mask)
    pos_joint_mask = joint_mask & (~static_pos_mask)

    target_pose = batch["position"]
    parents = batch["parent_a"]
    offsets = batch["offset_a"]
    global_scales = batch["global_scale"]

    metrics = {}
    metrics["rot_l1"] = masked_loss(pred_rot6d, gt_rot6d, rot_joint_mask, nn.L1Loss(reduction="none"))
    metrics["rot_l2"] = masked_loss(pred_rot6d, gt_rot6d, rot_joint_mask, nn.MSELoss(reduction="none"))

    rot_mask_np = rot_joint_mask.detach().cpu().numpy()
    metrics["angle_l1"] = angle_L1(pred_rot6d, gt_rot6d, mask=rot_mask_np)
    metrics["anglevel_l1"] = angle_velocity_L1(pred_rot6d, gt_rot6d, mask=rot_mask_np)

    pred_fk_pos = rot6d_to_fk_positions(pred_rot6d, offsets, parents, global_scales)
    metrics["fk_l1"] = masked_loss(
        pred_fk_pos,
        target_pose,
        pos_joint_mask,
        nn.L1Loss(reduction="none")
    ) * 3.0

    root_mask = torch.zeros_like(joint_mask)
    root_mask[:, 0] = True
    root_mask_np = root_mask.detach().cpu().numpy()

    metrics["root_rot_l1"] = masked_loss(pred_rot6d, gt_rot6d, root_mask, nn.L1Loss(reduction="none"))
    metrics["root_angle_l1"] = angle_L1(pred_rot6d, gt_rot6d, mask=root_mask_np)

    return metrics


def evaluate_joint_metrics(model_out: Dict[str, Any], batch: Dict[str, Any]):
    metrics = {}
    metrics.update(evaluate_pose_metrics(model_out, batch))
    metrics.update(evaluate_rot_metrics(model_out, batch))
    return metrics


def save_pose_npy(save_dir, name, pred, gt):
    os.makedirs(save_dir, exist_ok=True)

    pred_path = os.path.join(save_dir, f"{name}_pos_pred.npy")
    gt_path = os.path.join(save_dir, f"{name}_pos_gt.npy")

    np.save(pred_path, pred)
    np.save(gt_path, gt)

    return pred_path, gt_path


def save_rot_npy(save_dir, name, pred, gt):
    os.makedirs(save_dir, exist_ok=True)

    pred_path = os.path.join(save_dir, f"{name}_rot_pred.npy")
    gt_path = os.path.join(save_dir, f"{name}_rot_gt.npy")

    np.save(pred_path, pred)
    np.save(gt_path, gt)

    return pred_path, gt_path

def visualize_joint_sample(
    save_dir,
    character_dir,
    species_name,
    pred_pos,
    gt_pos,
    pred_rot,
    gt_rot,
):
    """
    一个 sample 输出四类结果：
        pos_pred / pos_gt
        rot_pred / rot_gt (-> bvh)
    """

    pos_pred_path, pos_gt_path = save_pose_npy(
        save_dir, species_name, pred_pos, gt_pos
    )

    rot_pred_path, rot_gt_path = save_rot_npy(
        save_dir, species_name, pred_rot, gt_rot
    )

    # 转 BVH
    convert_npy_to_bvh(rot_pred_path, species_name)
    convert_npy_to_bvh(rot_gt_path, species_name)

# =========================================================
# eval
# =========================================================
def run_evaluation(
    loader,
    model,
    device,
    attention_design,
    cfg,
    pose_pred_prob,
    base_dir,
    character_dir,
    writer=None,
    epoch=None,
    tag_prefix="test",
):
    model.eval()
    metric_sums = {
        # pose
        "pose_l1": 0.0,
        "pose_l2": 0.0,
        "pose_mpjpe": 0.0,
        "pose_mpjve": 0.0,
        "pose_speed_l1": 0.0,
        "pose_speed_l2": 0.0,
        # rot
        "rot_l1": 0.0,
        "rot_l2": 0.0,
        "angle_l1": 0.0,
        "anglevel_l1": 0.0,
        "fk_l1": 0.0,
        "root_rot_l1": 0.0,
        "root_angle_l1": 0.0,
    }
    sample_count = 0

    if is_main_process():
        tqdm_loader = tqdm(loader, total=len(loader), desc=f"Eval: {tag_prefix}", ncols=120)
    else:
        tqdm_loader = loader

    vis_done_species = set()
    max_vis_species = 50

    vis_save_dir = os.path.join(base_dir, f"vis_video2pose2rot_v2pv3_epoch{epoch + 1}")
    if is_main_process() and (epoch + 1) % cfg["train"]["vis_every"] == 0:
        os.makedirs(vis_save_dir, exist_ok=True)

    with torch.no_grad():

        cnt = 0
        for batch in tqdm_loader:
            cnt += 1
            if cfg["runtime"]["debug"] and cnt >= 10: break
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            model_out = model(
                batch=batch,
                attention_kwargs=attention_design,
                pose_source_mode="pred",
                pose_mix_prob=pose_pred_prob,
                detach_pred_pose_for_rot=cfg["train"]["pose_input"]["detach_pred_pose_for_rot"],
            )

            metrics = evaluate_joint_metrics(model_out, batch)
            bs = batch["position"].size(0)
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

                    pred_pos = model_out["pred_position"][si].detach().cpu().numpy()
                    gt_pos = batch["position"][si].detach().cpu().numpy()

                    pred_rot = model_out["pred_rot6d"][si].detach().cpu().numpy()
                    gt_rot = batch["rot6d_a"][si].detach().cpu().numpy()

                    try:
                        visualize_joint_sample(
                            save_dir=vis_save_dir,
                            species_name=species_name,
                            character_dir=character_dir,
                            pred_pos=pred_pos,
                            gt_pos=gt_pos,
                            pred_rot=pred_rot,
                            gt_rot=gt_rot,
                        )
                        print(f"[VIS] saved {species_name}")
                    except Exception as e:
                        print(f"[VIS] failed on {species_name}: {e}")

    sample_count = reduce_sum_scalar(sample_count, device)

    for k in metric_sums:
        metric_sums[k] = reduce_sum_scalar(metric_sums[k], device) / max(sample_count, 1)


    if is_main_process():
        print(
            f"[Eval] "
            f"pose_l1={metric_sums['pose_l1']:.6f}, "
            f"pose_l2={metric_sums['pose_l2']:.6f}, "
            f"pose_mpjpe={metric_sums['pose_mpjpe']:.6f}, "
            f"pose_mpjve={metric_sums['pose_mpjve']:.6f}, "
            f"pose_speed_l1={metric_sums['pose_speed_l1']:.6f}, "
            f"pose_speed_l2={metric_sums['pose_speed_l2']:.6f}, "
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
            writer.add_scalar(f"{tag_prefix}/{k}", v, epoch)

    return metric_sums

def train_video2pose2rot_v2pv3(cfg):
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

    logdir = os.path.join(base_dir, "logs_video2pose2rot_v2pv3")

    if is_main_process():
        logger.info(f"Log directory: {logdir}")

    device_str = runtime_cfg.get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    attention_design = model_cfg["attention_kwargs"]
    seq_len = attention_design["seq_len"]

    dataset_train = AnySpeciesPoseDataset(
        bvh_dir=data_cfg["bvh_dir"],
        window=seq_len,
        mmap=data_cfg.get("mmap", True),
        cache_scale=data_cfg.get("cache_scale", True),
        limit_species_debug=data_cfg.get("limit_species_debug", []),
        split_json=data_cfg["split_json"],
        split_mode="train",
        memory_pkl_path=data_cfg["train_memory_pkl_path"],
        preload_all=data_cfg.get("preload_all", False)
    )

    sampler_train = (
        DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=train_cfg["batch_size"],
        sampler=sampler_train,
        shuffle=(sampler_train is None),
        num_workers=train_cfg.get("num_workers_train", 4),
        collate_fn=collate_anyspecies_padded,
        worker_init_fn=worker_init_fn,
    )

    _, test_loaders = build_test_dataloaders(
        data_cfg=data_cfg,
        eval_cfg=eval_cfg,
        seq_len=seq_len,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    model: torch.nn.Module = instantiate_from_config(model_cfg)
    model = model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 0.0)

    if weight_decay == 0.0:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    pose_fn = get_loss_fn(train_cfg["loss"].get("pose_loss_type", "smooth_l1"))
    pose_vel_fn = get_loss_fn(train_cfg["loss"].get("pose_vel_loss_type", "smooth_l1"))
    rot_fn = get_loss_fn(train_cfg["loss"].get("rot_loss_type", "smooth_l1"))
    vel_fn = get_loss_fn(train_cfg["loss"].get("vel_loss_type", "smooth_l1"))
    acc_fn = get_loss_fn(train_cfg["loss"].get("acc_loss_type", "smooth_l1"))

    writer = SummaryWriter(logdir) if is_main_process() else None

    if is_main_process():
        print("=" * 100)
        print(model)
        print("=" * 100)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"pose_source_mode: {train_cfg['pose_input']['pose_source_mode']}")
        print(f"pose_mix_start_prob: {train_cfg['pose_input']['pose_mix_start_prob']}")
        print(f"pose_mix_end_prob: {train_cfg['pose_input']['pose_mix_end_prob']}")
        print(f"pose_mix_warmup_epochs: {train_cfg['pose_input']['pose_mix_warmup_epochs']}")
        print(f"detach_pred_pose_for_rot: {train_cfg['pose_input']['detach_pred_pose_for_rot']}")
        print("=" * 100)

    pretrain_ckpt = train_cfg.get("pretrain_ckpt")
    if pretrain_ckpt is not None and is_main_process():
        load_partial_pretrain(model.module if distributed else model, pretrain_ckpt)

    best_test_loss = float("inf")
    best_ckpt_path = os.path.join(base_dir, f"{PROCESS_NAME}_ckpt_best.pt")
    start_epoch = 0

    ckpt_path_latest = find_latest_ckpt(base_dir, PROCESS_NAME)
    if ckpt_path_latest and os.path.exists(ckpt_path_latest):
        if is_main_process():
            print(f"Loading checkpoint: {ckpt_path_latest}")
        start_epoch = load_checkpoint(
            model.module if distributed else model,
            optimizer,
            ckpt_path_latest,
            device,
        )
        if is_main_process():
            print(f"Resumed from epoch {start_epoch}")

    scaler = torch.amp.GradScaler("cuda")

    global_step = 0
    epochs = train_cfg["epochs"]
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)

    split_groups = eval_cfg.get("split_groups", ["seen", "rare", "unseen"])

    for epoch in range(start_epoch, epochs):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        pose_pred_prob = get_pose_mix_prob(
            epoch=epoch,
            total_epochs=epochs,
            mode=train_cfg["pose_input"]["pose_source_mode"],
            start_prob=train_cfg["pose_input"]["pose_mix_start_prob"],
            end_prob=train_cfg["pose_input"]["pose_mix_end_prob"],
            warmup_epochs=train_cfg["pose_input"]["pose_mix_warmup_epochs"],
            schedule=train_cfg["pose_input"]["pose_mix_schedule"],
        )

        if is_main_process():
            print(f"[Epoch {epoch + 1}] pose_pred_prob = {pose_pred_prob:.4f}")

        loader_tqdm = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs} [train][rank{rank}]",
            ncols=120,
        ) if is_main_process() else enumerate(train_loader)

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
                model_out = model(
                    batch=batch,
                    attention_kwargs=attention_design,
                    # pose_source_mode=args.pose_source_mode,
                    pose_source_mode="pred", # pred
                    pose_mix_prob=pose_pred_prob,
                    detach_pred_pose_for_rot=train_cfg["pose_input"]["detach_pred_pose_for_rot"],
                )

                total_loss, loss_dict = compute_joint_total_loss(
                    model_out=model_out,
                    batch=batch,
                    weight_cfg=train_cfg["weight"],
                    pose_criterion=pose_fn,
                    pose_vel_criterion=pose_vel_fn,
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

            running_loss += raw_loss.item() * batch["position"].size(0)

            if writer is not None and global_step % 10 == 0:
                for k, v in loss_dict.items():
                    if not torch.is_tensor(v):
                        continue

                    if k in ["loss_pose", "loss_pose_vel"]:
                        writer.add_scalar(f"train_pose/{k}", v.item(), global_step)
                    elif k in ["loss_rot", "loss_vel", "loss_acc", "loss_fk", "loss_root"]:
                        writer.add_scalar(f"train_rot/{k}", v.item(), global_step)
                    elif k in ["pose_total_loss", "rot_total_loss", "total_loss"]:
                        writer.add_scalar(f"train_total/{k}", v.item(), global_step)
                    else:
                        writer.add_scalar(f"train_misc/{k}", v.item(), global_step)
                psi = model_out["pose_source_info"]
                writer.add_scalar("train_schedule/pred_prob_target", psi["pred_prob"], global_step)
                writer.add_scalar("train_schedule/used_pred_ratio", psi["used_pred_ratio"], global_step)
                writer.add_scalar("train_schedule/used_gt_ratio", psi["used_gt_ratio"], global_step)

            global_step += 1

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            if is_main_process() and i % 10 == 0 and i > 0:
                avg_batch = sum(batch_times) / len(batch_times)
                remaining = avg_batch * (len(train_loader) - i - 1)
                loader_tqdm.set_postfix_str(
                    f"Loss={raw_loss.item():.4f} | posePred={pose_pred_prob:.2f} | ETA={remaining / 60:.1f}min"
                )

        train_loss = running_loss / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start_time

        if is_main_process():
            print(f"Epoch {epoch + 1}: train total loss={train_loss:.6f} | 用时 {epoch_time:.1f}s")
        if writer is not None:
            writer.add_scalar("epoch/train_total_loss", train_loss, epoch + 1)
            writer.add_scalar("epoch/pose_pred_prob", pose_pred_prob, epoch + 1)

        if (epoch + 1) % train_cfg["test_every"] == 0 or (epoch + 1) == train_cfg["epochs"]:
            split_metrics = {}
            for split in split_groups:
                loader = test_loaders[split]
                tag_prefix = f"test_{split}"
                metrics = run_evaluation(
                    loader=loader,
                    model=model,
                    device=device,
                    attention_design=attention_design,
                    cfg=cfg,
                    pose_pred_prob=pose_pred_prob,
                    writer=writer,
                    epoch=epoch + 1,
                    tag_prefix=tag_prefix,
                    base_dir=base_dir,
                    character_dir=data_cfg['character_dir'],
                )
                split_metrics[split] = metrics

            if is_main_process():
                save_checkpoint_with_epoch(
                    model.module if distributed else model,
                    PROCESS_NAME,
                    optimizer,
                    epoch + 1,
                    base_dir
                )
                cleanup_old_checkpoints(base_dir, PROCESS_NAME, train_cfg.get("max_ckpt", 100))

                best_metric_split = eval_cfg.get("best_metric_split", "seen")
                best_metric_name = eval_cfg.get("best_metric_name", "rot_l1")
                score = split_metrics[best_metric_split][best_metric_name]
                if score < best_test_loss:
                    best_test_loss = score
                    torch.save({
                        "model_state": (model.module if distributed else model).state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "best_test_loss": best_test_loss,
                        "split_metrics": split_metrics,
                    }, best_ckpt_path)
                    print(f"New best checkpoint saved: {best_ckpt_path} ({best_metric_split}/{best_metric_name}={best_test_loss:.6f})")

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Video2Pose2Rot (v2p v3)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/train_video2pose2rot_v2pv3.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train_video2pose2rot_v2pv3(cfg)
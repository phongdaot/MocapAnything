### mesh2pose.py ###
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils.dist_utils import *
from utils.train_utils import *
from utils.loss import *
from utils.common import *
from utils.visualization import plot_pose_compare_from_npy
from utils.logger import logger
from utils.config_utils import load_yaml_config, dump_yaml_config, instantiate_from_config

from data.loader_v1 import AnySpeciesPoseDataset, collate_anyspecies_padded
from models.v1.mesh2pose.model import RefGuidedMesh2PoseModel

torch.multiprocessing.set_start_method("spawn", force=True)

PROCESS_NAME = "mesh2pose"
MAX_JOINTS = 150


def run_evaluation(
    test_loader,
    model,
    device,
    attention_kwargs,
    writer=None,
    epoch=None,
    vis_save_dir=None,
    bvh_roots=None,
    tag_prefix="test",
):
    model.eval()

    test_loss_l1 = 0.0
    test_loss_l2 = 0.0
    test_mpjpe = 0.0
    test_mpjve = 0.0
    test_speed_l1 = 0.0
    test_speed_l2 = 0.0

    sample_count = 0
    speed_cnt = 0
    
    tqdm_loader = (
        tqdm(test_loader, total=len(test_loader), desc=f"Eval: {tag_prefix}", ncols=100)
        if is_main_process()
        else test_loader
    )

    vis_done_species = set()

    with torch.no_grad():
        cnt = 0
        for bi, batch in enumerate(tqdm_loader):            
            cnt += 1
            if cfg["runtime"]["debug"] and cnt >= 10: break

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            target = batch["position"]
            joint_mask = batch["joint_mask"].bool()
            static_mask = batch["static_mask"].bool()
            joint_mask_for_loss = joint_mask & (~static_mask)

            pred = model(batch, attention_kwargs=attention_kwargs)

            loss_l1 = masked_loss(
                pred, target, joint_mask_for_loss, nn.L1Loss(reduction="none")
            )
            loss_l2 = masked_loss(
                pred, target, joint_mask_for_loss, nn.MSELoss(reduction="none")
            )
            mpjpe = masked_mpjpe(pred, target, joint_mask_for_loss)

            bs = target.size(0)

            test_loss_l1 += loss_l1.item() * bs
            test_loss_l2 += loss_l2.item() * bs
            test_mpjpe += mpjpe.item() * bs
            sample_count += bs

            if pred.size(1) > 1:
                pred_diff = pred[:, 1:] - pred[:, :-1]
                target_diff = target[:, 1:] - target[:, :-1]
                joint_mask_for_loss_t = joint_mask_for_loss.unsqueeze(1).expand(
                    -1, pred_diff.size(1), -1
                )

                speed_l1 = masked_loss(
                    pred_diff,
                    target_diff,
                    joint_mask_for_loss_t,
                    nn.L1Loss(reduction="none"),
                )
                speed_l2 = masked_loss(
                    pred_diff,
                    target_diff,
                    joint_mask_for_loss_t,
                    nn.MSELoss(reduction="none"),
                )
                mpjve = masked_mpjve(pred, target, joint_mask_for_loss)

                n_speed = pred_diff.size(0) * pred_diff.size(1)

                test_speed_l1 += speed_l1.item() * n_speed
                test_speed_l2 += speed_l2.item() * n_speed
                test_mpjve += mpjve.item() * n_speed
                speed_cnt += n_speed

            if is_main_process() and vis_save_dir is not None:
                os.makedirs(vis_save_dir, exist_ok=True)
                species_list = batch["species"]
                for si, species_name in enumerate(species_list):
                    if species_name not in vis_done_species:
                        vis_done_species.add(species_name)

                        pred_np = pred.detach().cpu().numpy()[si]
                        gt_np = target.detach().cpu().numpy()[si]

                        npy_pred_path = os.path.join(
                            vis_save_dir, f"{tag_prefix}_{species_name}_pred.npy"
                        )
                        npy_gt_path = os.path.join(
                            vis_save_dir, f"{tag_prefix}_{species_name}_gt.npy"
                        )
                        np.save(npy_pred_path, pred_np)
                        np.save(npy_gt_path, gt_np)

                        try:
                            plot_pose_compare_from_npy(
                                npy_pred_path,
                                npy_gt_path,
                                species_name,
                                vis_save_dir,
                                fps=15,
                                bvh_roots=bvh_roots,
                            )
                        except Exception as e:
                            logger.info(f"[VIS-{tag_prefix}] Fail: {species_name}: {e}")

    test_loss_l1 = reduce_sum_scalar(test_loss_l1, device)
    test_loss_l2 = reduce_sum_scalar(test_loss_l2, device)
    test_mpjpe = reduce_sum_scalar(test_mpjpe, device)
    test_speed_l1 = reduce_sum_scalar(test_speed_l1, device)
    test_speed_l2 = reduce_sum_scalar(test_speed_l2, device)
    test_mpjve = reduce_sum_scalar(test_mpjve, device)

    sample_count = reduce_sum_scalar(sample_count, device)
    speed_cnt = reduce_sum_scalar(speed_cnt, device)
    
    test_loss_l1 /= sample_count
    test_loss_l2 /= sample_count
    test_mpjpe /= sample_count
    
    if speed_cnt > 0:
        test_speed_l1 /= speed_cnt
        test_speed_l2 /= speed_cnt
        test_mpjve /= speed_cnt
        
    if is_main_process():
        logger.info(
            f"[{tag_prefix}] L1={test_loss_l1:.4f} "
            f"L2={test_loss_l2:.4f} "
            f"MPJPE={test_mpjpe:.4f}"
        )
        logger.info(
            f"[{tag_prefix}] speed_L1={test_speed_l1:.4f} "
            f"speed_L2={test_speed_l2:.4f} "
            f"MPJVE={test_mpjve:.4f}"
        )

    if writer is not None and epoch is not None:
        writer.add_scalar(f"{tag_prefix}/loss_L1", test_loss_l1, epoch)
        writer.add_scalar(f"{tag_prefix}/loss_L2", test_loss_l2, epoch)
        writer.add_scalar(f"{tag_prefix}/mpjpe", test_mpjpe, epoch)
        writer.add_scalar(f"{tag_prefix}/speed_L1", test_speed_l1, epoch)
        writer.add_scalar(f"{tag_prefix}/speed_L2", test_speed_l2, epoch)
        writer.add_scalar(f"{tag_prefix}/mpjve", test_mpjve, epoch)

    return {
        "loss_l1": test_loss_l1,
        "loss_l2": test_loss_l2,
        "mpjpe": test_mpjpe,
        "speed_l1": test_speed_l1,
        "speed_l2": test_speed_l2,
        "mpjve": test_mpjve,
    }


def build_loss(loss_type: str):
    if loss_type == "l1":
        return nn.L1Loss(reduction="none")
    elif loss_type == "l2":
        return nn.MSELoss(reduction="none")
    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss(reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def build_test_dataloaders(data_cfg, eval_cfg, attention_design, distributed, rank, world_size):
    
    test_types = eval_cfg.get("test_types", {"gtmesh": None})

    test_datasets = {}
    test_loaders = {}

    for test_type, mesh_dir in test_types.items():
        kwargs = dict(
            bvh_dir=data_cfg["bvh_dir"],
            window=attention_design["seq_len"],
            use_rot6d=False,
            mmap=data_cfg.get("mmap", True),
            cache_scale=data_cfg.get("cache_scale", True),
            limit_species_debug=data_cfg.get("limit_species_debug", []),
            split_json=data_cfg["split_json"],
            split_mode="test",
        )
        if mesh_dir is not None:
            kwargs["mesh_dir"] = mesh_dir

        ds = AnySpeciesPoseDataset(**kwargs)

        sampler = (
            DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
            if distributed
            else None
        )

        loader = DataLoader(
            ds,
            batch_size=eval_cfg.get("batch_size", 1),
            sampler=sampler,
            shuffle=False,
            num_workers=eval_cfg.get("num_workers", 2),
            collate_fn=collate_anyspecies_padded,
            worker_init_fn=worker_init_fn,
        )

        test_datasets[test_type] = ds
        test_loaders[test_type] = loader

    return test_datasets, test_loaders


def train_mesh2pose(cfg):
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

    config_save_path = os.path.join(base_dir, "config.yaml")
    dump_yaml_config(cfg, config_save_path)
    logger.info(f"Config saved to {config_save_path}")

    logdir = os.path.join(base_dir, f"logs_{PROCESS_NAME}")
    if is_main_process():
        logger.info(f"Log directory: {logdir}")

    device_str = runtime_cfg.get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    attention_design = model_cfg["attention_kwargs"]

    dataset_train = AnySpeciesPoseDataset(
        bvh_dir=data_cfg["bvh_dir"],
        window=attention_design["seq_len"],
        use_rot6d=False,
        mmap=data_cfg.get("mmap", True),
        cache_scale=data_cfg.get("cache_scale", True),
        limit_species_debug=data_cfg.get("limit_species_debug", []),
        split_json=data_cfg["split_json"],
        split_mode="train",
    )

    sampler_train = (
        DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
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
        attention_design=attention_design,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    model: torch.nn.Module = instantiate_from_config(model_cfg)
    model = model.to(device)

    pretrain_ckpt = train_cfg.get("pretrain_ckpt")
    if pretrain_ckpt is not None:
        load_partial_pretrain(model, pretrain_ckpt)

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

    criterion = build_loss(train_cfg["loss_type"])
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
            device,
        )
        if is_main_process():
            logger.info(f"Resumed from epoch {start_epoch}")

    scaler = torch.amp.GradScaler()
    global_step = 0
    epochs = train_cfg["epochs"]
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)

    split_groups = eval_cfg.get("split_groups", ["seen", "rare", "unseen"])
    test_types = eval_cfg.get("test_types", {"gtmesh": None})

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
                ncols=100,
            )
            if is_main_process()
            else enumerate(train_loader)
        )

        batch_times = []

        cnt = 0
        for i, batch in loader_tqdm:
            cnt += 1
            if cfg["runtime"]["debug"] and cnt >= 10: break

            batch_start_time = time.time()

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            target = batch["position"]
            joint_mask = batch["joint_mask"].bool()
            static_mask = batch["static_mask"].bool()
            joint_mask_for_loss = joint_mask & (~static_mask)

            with torch.cuda.amp.autocast():
                pred = model(batch, attention_kwargs=attention_design)
                raw_loss = masked_loss(pred, target, joint_mask_for_loss, criterion)

            loss = raw_loss / grad_accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += raw_loss.item() * target.size(0)

            if writer is not None and global_step % 10 == 0:
                writer.add_scalar("train/loss", raw_loss.item(), global_step)

            global_step += 1

            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            if i % 10 == 0 and i > 0 and is_main_process():
                avg_batch = sum(batch_times) / len(batch_times)
                remaining = avg_batch * (len(train_loader) - i - 1)
                loader_tqdm.set_postfix_str(
                    f"Loss={raw_loss.item():.4f} | ETA={remaining / 60:.1f}min"
                )

        train_loss = running_loss / len(train_loader.dataset)
        epoch_time = time.time() - epoch_start_time

        if is_main_process():
            logger.info(f"Epoch {epoch + 1}: train loss={train_loss:.6f} | time {epoch_time:.1f}s")

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)

        test_every = train_cfg.get("test_every", 1)
        vis_every = train_cfg.get("vis_every", 5)

        if (epoch + 1) % test_every == 0 or (epoch + 1) == epochs:
            vis_save_dir = (
                os.path.join(base_dir, f"vis_compare_epoch{epoch + 1}")
                if is_main_process() and (epoch + 1) % vis_every == 0
                else None
            )

            eval_metrics = {}

            for test_type in test_types:
                loader = test_loaders[test_type]
                tag_prefix = f"test_{test_type}"

                metrics = run_evaluation(
                    test_loader=loader,
                    model=model,
                    device=device,
                    attention_kwargs=attention_design,
                    writer=writer,
                    epoch=epoch + 1,
                    vis_save_dir=vis_save_dir,
                    bvh_roots=data_cfg.get("bvh_roots"),
                    tag_prefix=tag_prefix,
                )
                eval_metrics[test_type] = metrics

            if is_main_process():
                save_checkpoint_with_epoch(
                    model.module if distributed else model,
                    PROCESS_NAME,
                    optimizer,
                    epoch + 1,
                    base_dir,
                )
                cleanup_old_checkpoints(base_dir, PROCESS_NAME, train_cfg.get("max_ckpt", 5))

                best_metric_test_type = eval_cfg.get("best_metric_test_type", "gtmesh")
                best_metric_name = eval_cfg.get("best_metric_name", "mpjpe")

                cur_loss = eval_metrics[best_metric_test_type][best_metric_name]

                if cur_loss < best_test_loss:
                    best_test_loss = cur_loss
                    torch.save(
                        {
                            "model_state": (model.module if distributed else model).state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch + 1,
                            "best_test_loss": best_test_loss,
                        },
                        best_ckpt_path,
                    )
                    logger.info(
                        f"New best checkpoint saved: {best_ckpt_path} "
                        f"({best_metric_test_type}-{best_metric_name}={best_test_loss:.6f})"
                    )

    if writer is not None:
        writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    cfg = load_yaml_config("configs/train/train_mesh2pose.yaml")
    train_mesh2pose(cfg)
    
    
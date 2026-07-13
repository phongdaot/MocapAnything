### video2pose2rot_multidata.py ###
# 多数据集(zoo1030 + obj1k [+ mobjaverse])联合训练。
# 与 train/video2pose2rot.py 的单数据集版等价,差别只在:
#   · 训练集 = 多个 AnySpeciesPoseDataset(每集各自 bvh_dir/split_json) 的 ConcatDataset
#   · 测试集 = 每 (test_set, split) 一个实例,分别报告 test_{name}_{split} 指标
#   · best ckpt 判据可指定某个 (test_set, split, metric)
# 模型 / 前向 / loss / 评测指标 / scheduled-sampling 全部复用 video2pose2rot.py,保证与其一致。
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from data.loader_v2 import AnySpeciesPoseDataset, collate_anyspecies_padded
from utils.config_utils import instantiate_from_config, load_yaml_config, dump_yaml_config
from utils.common import set_seed
from utils.dist_utils import setup_distributed, is_main_process, cleanup_distributed
from utils.dist_utils import *
from utils.loss import *
from utils.train_utils import *

# 复用单数据集版的调度 / 评测,保证 clean 两条路径完全一致
from train.video2pose2rot import (
    PROCESS_NAME,
    get_pose_mix_prob,
    run_evaluation,
)

torch.multiprocessing.set_start_method("spawn", force=True)

import time
import numpy as np
from tqdm import tqdm


# =========================================================
# 多数据集 loader 构建
# =========================================================
def build_multidata_train_loader(data_cfg, attention_design, train_cfg,
                                  distributed, rank, world_size):
    """每个 train_sets 条目建一个 split_mode='train' 的 AnySpeciesPoseDataset,ConcatDataset 合并。"""
    parts = []
    for d in data_cfg["train_sets"]:
        ds = AnySpeciesPoseDataset(
            bvh_dir=d["bvh_dir"],
            window=attention_design["seq_len"],
            mmap=data_cfg.get("mmap", True),
            cache_scale=data_cfg.get("cache_scale", True),
            limit_species_debug=data_cfg.get("limit_species_debug", []),
            split_json=d["split_json"],
            split_mode="train",
            memory_pkl_path=d.get("memory_pkl_path"),
            preload_all=data_cfg.get("preload_all", False),
            blocklist=d.get("blocklist"),
            pose_jumps_path=d.get("pose_jumps_path"),
            raw_position=d.get("raw_position", False),
            epoch_sample_ratio=d.get("epoch_sample_ratio"),   # mobjaverse=0.25 均衡
            ref_enhance=d.get("ref_enhance"),                 # 对齐发布配方:cross_seq(zoo)/cross_angle(obj)
        )
        if is_main_process():
            logger.info(f"[train] {d.get('name', d['bvh_dir'])}: {len(ds)} windows")
        parts.append(ds)

    concat = ConcatDataset(parts)
    if is_main_process():
        logger.info(f"[train] ConcatDataset total = {len(concat)} windows over {len(parts)} datasets")

    sampler = (
        DistributedSampler(concat, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    _nw = train_cfg.get("num_workers_train", 4)
    loader = DataLoader(
        concat,
        batch_size=train_cfg["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=_nw,
        collate_fn=collate_anyspecies_padded,
        worker_init_fn=worker_init_fn,
        persistent_workers=(_nw > 0),                  # 发布配方:避免每 epoch 重建 worker
        prefetch_factor=(2 if _nw > 0 else None),      # 对齐发布配方
    )
    return concat, loader


def build_multidata_test_loaders(data_cfg, eval_cfg, attention_design,
                                 distributed, rank, world_size):
    """返回嵌套字典 {test_set_name: {split: loader}} 和同结构的 character_dir 表。"""
    test_loaders = {}
    char_dirs = {}
    for t in data_cfg["test_sets"]:
        name = t["name"]
        test_loaders[name] = {}
        char_dirs[name] = t.get("character_dir")
        for split in t["splits"]:
            ds = AnySpeciesPoseDataset(
                bvh_dir=t["bvh_dir"],
                window=attention_design["seq_len"],
                mmap=data_cfg.get("mmap", True),
                cache_scale=data_cfg.get("cache_scale", True),
                limit_species_debug=data_cfg.get("limit_species_debug", []),
                split_json=t["split_json"],
                split_mode="test",
                split_group=split,
                memory_pkl_path=t.get("memory_pkl_path"),
                preload_all=eval_cfg.get("preload_all", False),
                blocklist=t.get("blocklist"),
                pose_jumps_path=t.get("pose_jumps_path"),
                raw_position=t.get("raw_position", False),
            )
            sampler = (
                DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
                if distributed else None
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
            test_loaders[name][split] = loader
            if is_main_process():
                logger.info(f"[test] {name}/{split}: {len(ds)} windows")
    return test_loaders, char_dirs


# =========================================================
# train
# =========================================================
def train_video2pose2rot_multidata(cfg):
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
    logdir = os.path.join(base_dir, "logs_video2pose2rot")

    device_str = runtime_cfg.get("device", "cuda")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    attention_design = model_cfg["attention_kwargs"]
    no_joint_embed = train_cfg.get("no_joint_embed", False)

    # ---- data ----
    _, train_loader = build_multidata_train_loader(
        data_cfg, attention_design, train_cfg, distributed, rank, world_size
    )
    test_loaders, char_dirs = build_multidata_test_loaders(
        data_cfg, eval_cfg, attention_design, distributed, rank, world_size
    )

    # ---- model (与单数据集版同一构建方式) ----
    model: torch.nn.Module = instantiate_from_config(model_cfg).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )

    lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 0.0)
    optimizer = (optim.Adam(model.parameters(), lr=lr) if weight_decay == 0.0
                 else optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay))

    pose_fn = get_loss_fn(train_cfg["loss"].get("pose_loss_type", "smooth_l1"))
    pose_vel_fn = get_loss_fn(train_cfg["loss"].get("pose_vel_loss_type", "smooth_l1"))
    rot_fn = get_loss_fn(train_cfg["loss"].get("rot_loss_type", "smooth_l1"))
    vel_fn = get_loss_fn(train_cfg["loss"].get("vel_loss_type", "smooth_l1"))
    acc_fn = get_loss_fn(train_cfg["loss"].get("acc_loss_type", "smooth_l1"))

    writer = SummaryWriter(logdir) if is_main_process() else None

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        print("=" * 100)
        print(f"[multidata] train_sets={[d.get('name') for d in data_cfg['train_sets']]} "
              f"test_sets={[t['name'] for t in data_cfg['test_sets']]}")
        print(f"总参数量: {total_params:,}  seq_len={attention_design['seq_len']}  epochs={train_cfg['epochs']}")
        print("=" * 100)

    pretrain_ckpt = train_cfg.get("pretrain_ckpt")
    if pretrain_ckpt is not None and is_main_process():
        load_partial_pretrain(model.module if distributed else model, pretrain_ckpt)

    best_metric = eval_cfg.get("best_metric", {"test_set": data_cfg["test_sets"][0]["name"],
                                               "split": "seen", "name": "rot_l1"})
    best_test_loss = float("inf")
    best_ckpt_path = os.path.join(base_dir, f"{PROCESS_NAME}_ckpt_best.pt")

    start_epoch = 0
    ckpt_path_latest = find_latest_ckpt(base_dir, PROCESS_NAME)
    if ckpt_path_latest and os.path.exists(ckpt_path_latest):
        if is_main_process():
            print(f"Loading checkpoint: {ckpt_path_latest}")
        start_epoch = load_checkpoint(model.module if distributed else model,
                                      optimizer, ckpt_path_latest, device)

    scaler = torch.amp.GradScaler("cuda")
    global_step = 0
    epochs = train_cfg["epochs"]
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)

    for epoch in range(start_epoch, epochs):
        # 子采样数据集(mobjaverse 1/4)每 epoch 重洗生效子集;所有 rank 同种子 → 一致。
        # 顺序:先重洗子集(改变 __len__ 对应的映射),再让 sampler set_epoch。
        _concat = getattr(train_loader, "dataset", None)
        for _sub in getattr(_concat, "datasets", []):
            if hasattr(_sub, "set_epoch"):
                _sub.set_epoch(epoch)
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        pose_pred_prob = get_pose_mix_prob(
            epoch=epoch, total_epochs=epochs,
            mode=train_cfg["pose_input"]["pose_source_mode"],
            start_prob=train_cfg["pose_input"]["pose_mix_start_prob"],
            end_prob=train_cfg["pose_input"]["pose_mix_end_prob"],
            warmup_epochs=train_cfg["pose_input"]["pose_mix_warmup_epochs"],
            schedule=train_cfg["pose_input"]["pose_mix_schedule"],
        )
        if is_main_process():
            print(f"[Epoch {epoch + 1}] pose_pred_prob = {pose_pred_prob:.4f}")

        loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader),
                           desc=f"Epoch {epoch + 1}/{epochs} [train][rank{rank}]", ncols=120) \
            if is_main_process() else enumerate(train_loader)

        optimizer.zero_grad(set_to_none=True)
        cnt = 0
        for i, batch in loader_tqdm:
            cnt += 1
            if cfg["runtime"]["debug"] and cnt >= 10:
                break
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # no-joint-embed:batch 层面把 joint_t5embed 置零(与 lab --ablate_no_t5 同做法)
            if no_joint_embed:
                batch["joint_t5embed"] = torch.zeros_like(batch["joint_t5embed"])

            with torch.amp.autocast("cuda", dtype=torch.float16):
                model_out = model(
                    batch=batch,
                    attention_kwargs=attention_design,
                    # 发布配方:训练前向用 config 的 mode(mix),不再写死 pred,
                    # 否则 scheduled-sampling warmup(pose_pred_prob)整条失效。
                    pose_source_mode=train_cfg["pose_input"]["pose_source_mode"],
                    pose_mix_prob=pose_pred_prob,
                    detach_pred_pose_for_rot=train_cfg["pose_input"]["detach_pred_pose_for_rot"],
                )
                total_loss, loss_dict = compute_joint_total_loss(
                    model_out=model_out, batch=batch,
                    weight_cfg=train_cfg["weight"],
                    pose_criterion=pose_fn, pose_vel_criterion=pose_vel_fn,
                    rot_criterion=rot_fn, vel_criterion=vel_fn, acc_criterion=acc_fn,
                )

            # NaN 安全网:个别极端样本可能让 loss=NaN/Inf;跳过该步不 backward,避免污染参数。
            # 必须分布式一致:backward 是集合通信,若只有部分 rank 跳过(local continue),
            # 其余 rank 会卡在 all-reduce 死锁。用 all_reduce(MAX) 让全体 rank 一起跳过同一步。
            _bad = torch.tensor([0.0 if torch.isfinite(total_loss) else 1.0], device=total_loss.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(_bad, op=torch.distributed.ReduceOp.MAX)
            if _bad.item() > 0:
                optimizer.zero_grad(set_to_none=True)
                if is_main_process() and i % 50 == 0:
                    print(f"[warn] non-finite loss at step {i}, 全体 rank 跳过")
                global_step += 1
                continue

            loss = total_loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += total_loss.item() * batch["position"].size(0)

            if writer is not None and global_step % 10 == 0:
                for k, v in loss_dict.items():
                    if torch.is_tensor(v):
                        writer.add_scalar(f"train/{k}", v.item(), global_step)
            global_step += 1
            if is_main_process() and i % 10 == 0 and i > 0:
                loader_tqdm.set_postfix_str(f"Loss={total_loss.item():.4f} | posePred={pose_pred_prob:.2f}")

        train_loss = running_loss / len(train_loader.dataset)
        if is_main_process():
            print(f"Epoch {epoch + 1}: train loss={train_loss:.6f} | {time.time() - epoch_start_time:.1f}s")
        if writer is not None:
            writer.add_scalar("epoch/train_total_loss", train_loss, epoch + 1)

        # ---- eval: 每个 (test_set, split) 分别报告 ----
        if (epoch + 1) % train_cfg["test_every"] == 0 or (epoch + 1) == epochs:
            all_metrics = {}
            for name, split_loaders in test_loaders.items():
                all_metrics[name] = {}
                for split, loader in split_loaders.items():
                    metrics = run_evaluation(
                        loader=loader, model=model, device=device,
                        attention_design=attention_design, cfg=cfg,
                        pose_pred_prob=pose_pred_prob,
                        base_dir=base_dir, character_dir=char_dirs.get(name),
                        writer=writer, epoch=epoch + 1,
                        tag_prefix=f"test_{name}_{split}",
                    )
                    all_metrics[name][split] = metrics

            if is_main_process():
                save_checkpoint_with_epoch(model.module if distributed else model,
                                           PROCESS_NAME, optimizer, epoch + 1, base_dir)
                cleanup_old_checkpoints(base_dir, PROCESS_NAME, train_cfg.get("max_ckpt", 100))

                score = all_metrics[best_metric["test_set"]][best_metric["split"]][best_metric["name"]]
                if score < best_test_loss:
                    best_test_loss = score
                    torch.save({
                        "model_state": (model.module if distributed else model).state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "best_test_loss": best_test_loss,
                        "split_metrics": all_metrics,
                    }, best_ckpt_path)
                    print(f"New best: {best_ckpt_path} "
                          f"({best_metric['test_set']}/{best_metric['split']}/{best_metric['name']}={best_test_loss:.6f})")

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multidata training for Video2Pose2Rot")
    parser.add_argument("--config", type=str,
                        default="configs/train/train_video2pose2rot_multidata.yaml")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    train_video2pose2rot_multidata(cfg)

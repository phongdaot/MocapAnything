import torch
import numpy as np
from .logger import logger
import glob
import os
import re
import torch.distributed as dist

# ==== checkpoint相关工具 ====
def save_checkpoint_with_epoch(model: torch.nn, name, optimizer, epoch, base_dir):
    ckpt_path = os.path.join(base_dir, f"{name}_ckpt_epoch{epoch}.pt")
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
    }, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")
    return ckpt_path


def cleanup_old_checkpoints(base_dir, name, max_ckpt=5):
    ckpt_list = sorted(glob.glob(os.path.join(base_dir, f"{name}_ckpt_epoch*.pt")), key=os.path.getmtime)
    if len(ckpt_list) > max_ckpt:
        for ckpt in ckpt_list[:-max_ckpt]:
            os.remove(ckpt)
            logger.info(f"Deleted old checkpoint: {ckpt}")


def find_latest_ckpt(base_dir, name):
    ckpt_files = glob.glob(os.path.join(base_dir, f"{name}_ckpt_epoch*.pt"))
    if not ckpt_files:
        return None

    def epoch_num(path):
        m = re.search(r'epoch(\d+)', path)
        return int(m.group(1)) if m else 0

    ckpt_files.sort(key=epoch_num)
    return ckpt_files[-1]


def load_checkpoint(model, optimizer, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch']
    return start_epoch


def load_partial_pretrain(model, pretrain_path):
    logger.info(f"==> Loading pretrain weights from {pretrain_path}")
    pretrain = torch.load(pretrain_path, map_location='cpu')
    pretrain_dict = pretrain['model_state'] if 'model_state' in pretrain else pretrain
    model_dict = model.state_dict()
    # 只保留能对上的参数
    load_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    # 打印加载和未加载的key
    logger.info(f"Loaded params: {list(load_dict.keys())}")
    logger.info(f"Missed params: {list(set(model_dict.keys()) - set(load_dict.keys()))}")
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
### dist_utils.py ###
import torch.distributed as dist
import torch
import numpy as np
import os

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2 ** 32))

# ========= 分布式训练基础函数 =========
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
        if is_main_process():
            print(f"[INFO] 初始化分布式训练: rank {rank}/{world_size} (GPU {local_rank})")
        return True, rank, world_size, local_rank
    else:
        if is_main_process():
            print("[INFO] 未启用分布式训练，采用单卡模式")
        return False, 0, 1, 0

def reduce_sum_scalar(x, device):
    t = torch.tensor(x, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()
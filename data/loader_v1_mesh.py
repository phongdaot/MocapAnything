
### loader_v1_mesh.py ###
# Dataset for training the video2mesh temporal DiT with RectifiedFlow.
#
# Consumes .npz files produced by preprocess/preprocess_data.py, each holding:
#     image_embed: [T, 257, 1024]   (frozen DinoV2 embedding, per-frame)
#     latent:      [T, 2048, 64]    (VAE-encoded mesh latent, per-frame)
#
# Sliding windows of length ``seq_len`` are produced from every sequence.
# A JSON skip-list (typically the test split) can be applied to drop
# held-out sequences. Meta information (per-file frame count) is cached on
# disk so subsequent runs don't need to re-scan every .npz.

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm


def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _print_rank0(*args, **kwargs):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def load_or_cache_meta(
    files: List[str],
    num_threads: int = 16,
    preload: bool = False,
    mmap_mode: Optional[str] = "r",
    cache_path: Optional[str] = None,
):
    """
    Load per-file meta (``T``) from .npz files, caching the result as JSON.
    DDP safe: rank 0 does the scan + write, other ranks wait on a barrier
    and then read the cache.
    """
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0

    def load_meta_single(p):
        if preload:
            arr = np.load(p, mmap_mode=None)
            img = np.asarray(arr["image_embed"])
            lat = np.asarray(arr["latent"])
            if img.shape[0] != lat.shape[0]:
                raise ValueError(f"Frame count mismatch in {p}")
            return {"img": img, "lat": lat, "T": int(img.shape[0]), "path": p}
        else:
            arr = np.load(p, mmap_mode=mmap_mode)
            try:
                T = int(arr["image_embed"].shape[0])
            finally:
                arr.close()
            return {"path": p, "T": T}

    if cache_path is None:
        first_dir = os.path.dirname(os.path.commonpath(files))
        cache_path = os.path.join(first_dir, "dataset_meta_cache.json")

    if is_main:
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and all("path" in m and "T" in m for m in cached):
                    _print_rank0(
                        f"[INFO][Rank0] Loaded meta cache: {len(cached)} files "
                        f"({cache_path})"
                    )
                    if distributed:
                        dist.barrier()
                    return cached
                else:
                    _print_rank0(f"[WARN][Rank0] cache structure invalid, rescanning.")
            except Exception as e:
                _print_rank0(f"[WARN][Rank0] failed to read cache: {e}, rescanning.")

        _print_rank0(
            f"[INFO][Rank0] Scanning meta for {len(files)} files "
            f"(threads={num_threads})"
        )
        loaded = []
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = {ex.submit(load_meta_single, p): p for p in files}
            for f in tqdm(as_completed(futures), total=len(futures), ncols=100, desc="[Load meta]"):
                try:
                    loaded.append(f.result())
                except Exception as e:
                    _print_rank0(f"[WARN] failed {futures[f]}: {e}")
        _print_rank0(f"[INFO][Rank0] meta loaded: {len(loaded)} files")

        try:
            with open(cache_path, "w") as f:
                json.dump(loaded, f, indent=2)
            _print_rank0(f"[INFO][Rank0] meta cached to {cache_path}")
        except Exception as e:
            _print_rank0(f"[WARN][Rank0] failed to save cache: {e}")

        if distributed:
            dist.barrier()
        return loaded

    # non-main ranks: wait and read
    if distributed:
        dist.barrier()
    with open(cache_path, "r") as f:
        return json.load(f)


class FullSequenceLatentDataset(Dataset):
    """
    Sliding-window dataset of (image_embed, latent) pairs from preprocessed
    .npz files.
    """

    def __init__(
        self,
        npz_dirs,
        seq_len: int = 16,
        image_key: str = "image_embed",
        latent_key: str = "latent",
        frame_step: int = 1,
        hop: Optional[int] = None,
        drop_last: bool = True,
        preload: bool = False,
        mmap_mode: Optional[str] = "r",
        num_threads: int = 32,
        skip_json: Optional[str] = None,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.image_key = image_key
        self.latent_key = latent_key
        self.frame_step = int(frame_step)
        self.hop = int(hop) if hop is not None else self.seq_len
        self.drop_last = drop_last
        self.preload = preload
        self.mmap_mode = None if preload else mmap_mode
        self.num_threads = num_threads

        # Gather .npz files
        if isinstance(npz_dirs, str):
            npz_dirs = [d.strip() for d in npz_dirs.split(",") if d.strip()]

        files: List[str] = []
        for npz_dir in npz_dirs:
            if not os.path.isdir(npz_dir):
                raise ValueError(f"{npz_dir} is not a directory")
            for root, _, fnames in os.walk(npz_dir):
                for f in fnames:
                    if f.endswith(".npz"):
                        files.append(os.path.join(root, f))
        files.sort(key=_natural_sort_key)

        # Apply skip list (held-out sequences)
        if skip_json is not None:
            skip_path = skip_json
            if not os.path.isabs(skip_path):
                # resolve relative to the first dataset dir
                skip_path = os.path.join(os.path.dirname(npz_dirs[0]), skip_json)
                if not skip_path.endswith(".json"):
                    skip_path = skip_path + ".json"

            if os.path.exists(skip_path):
                with open(skip_path, "r") as f:
                    skip_dict = json.load(f)
                skip_list = set()
                if isinstance(skip_dict, dict):
                    for v in skip_dict.values():
                        if isinstance(v, list):
                            skip_list.update(v)
                elif isinstance(skip_dict, list):
                    skip_list.update(skip_dict)

                _print_rank0(f"[INFO] skip list: {len(skip_list)} sequences ({skip_path})")
                kept = [
                    f for f in files
                    if os.path.basename(os.path.dirname(f)) not in skip_list
                ]
                _print_rank0(
                    f"[DEBUG] original files={len(files)} kept={len(kept)}"
                )
                files = kept

        if not files:
            raise FileNotFoundError(f"No npz files found under {npz_dirs}")
        self.files = files

        cache_path = os.path.join(
            os.path.dirname(npz_dirs[0]), "dataset_meta_cache.json"
        )
        self._loaded = load_or_cache_meta(
            self.files,
            num_threads=self.num_threads,
            preload=self.preload,
            mmap_mode=self.mmap_mode,
            cache_path=cache_path,
        )

        # Build sliding window index
        self._index: List[Tuple[int, int]] = []
        for fi, meta in enumerate(self._loaded):
            T = meta["T"]
            max_start = T - (self.seq_len - 1) * self.frame_step
            end = max(0, max_start) if self.drop_last else T
            if self.drop_last and max_start <= 0:
                continue
            start = 0
            while start < end:
                self._index.append((fi, start))
                start += self.hop

        if not self._index:
            if self.drop_last:
                raise ValueError(
                    "No valid sequence windows. Try smaller seq_len/frame_step or drop_last=False."
                )
            self._index.append((0, 0))

    def __len__(self):
        return len(self._index)

    def _load_arrays(self, fi: int):
        meta = self._loaded[fi]
        if self.preload:
            return meta["img"], meta["lat"]
        arr = np.load(meta["path"], mmap_mode=self.mmap_mode)
        img = arr[self.image_key]
        lat = arr[self.latent_key]
        return img, lat

    def __getitem__(self, idx: int):
        fi, start = self._index[idx]
        img_np, lat_np = self._load_arrays(fi)
        T = img_np.shape[0]
        idxs = start + np.arange(self.seq_len) * self.frame_step

        if self.drop_last:
            img_win = img_np[idxs]
            lat_win = lat_np[idxs]
        else:
            valid = idxs < T
            idxs = idxs[valid]
            img_win = img_np[idxs]
            lat_win = lat_np[idxs]
            if img_win.shape[0] < self.seq_len:
                pad = self.seq_len - img_win.shape[0]
                img_win = np.concatenate([img_win, np.repeat(img_win[-1:], pad, axis=0)], axis=0)
                lat_win = np.concatenate([lat_win, np.repeat(lat_win[-1:], pad, axis=0)], axis=0)

        return (
            torch.from_numpy(np.asarray(img_win)).float(),
            torch.from_numpy(np.asarray(lat_win)).float(),
        )

    def get_full_sequence(self, file_idx: int = 0):
        img_np, lat_np = self._load_arrays(file_idx)
        return (
            torch.from_numpy(np.asarray(img_np)).float(),
            torch.from_numpy(np.asarray(lat_np)).float(),
        )

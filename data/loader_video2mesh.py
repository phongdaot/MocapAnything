### loader_video2mesh.py ###
"""Dataset utilities for video2mesh (TripoSG temporal DiT) training.

Each `.npz` file is expected to contain two arrays of equal length T along the
first axis:

- ``image_embed`` — per-frame image encoder output (shape ``[T, S2, D2]``)
- ``latent``      — per-frame mesh latent produced by the TripoSG VAE
                    (shape ``[T, S1, D1]``)

Sequences are sampled as sliding windows of length ``seq_len`` with a stride of
``hop`` (and optional ``frame_step`` between frames inside a window).  The
loader supports on-demand mmap reads so datasets with many long clips do not
need to fit in RAM, and caches the per-file frame counts to JSON so re-scanning
every run is cheap.
"""

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

from utils.logger import logger


def _natural_sort_key(s: str):
    """Natural sort key: ``file2`` < ``file10``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def load_or_cache_meta(
    files: List[str],
    num_threads: int = 16,
    preload: bool = False,
    mmap_mode: Optional[str] = "r",
    cache_path: Optional[str] = None,
):
    """Return per-file meta (``path``, ``T`` and optionally preloaded arrays).

    Rank-0 scans the files, writes a JSON cache, and all other ranks read the
    cache once rank-0 finishes (via ``dist.barrier()``).
    """

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    is_main = rank == 0

    def load_meta_single(p: str):
        if preload:
            arr = np.load(p, mmap_mode=None)
            img = np.asarray(arr["image_embed"])
            lat = np.asarray(arr["latent"])
            if img.shape[0] != lat.shape[0]:
                raise ValueError(f"Frame count mismatch in {p}")
            return {"img": img, "lat": lat, "T": int(img.shape[0]), "path": p}

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
                    logger.info(f"[meta] Loaded cache with {len(cached)} files from {cache_path}")
                    if distributed:
                        dist.barrier()
                    return cached
                logger.warning("[meta] Cache format invalid; rescanning.")
            except Exception as e:
                logger.warning(f"[meta] Cache read failed ({e}); rescanning.")

        logger.info(f"[meta] Scanning {len(files)} npz files (threads={num_threads})")
        loaded = []
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futures = {ex.submit(load_meta_single, p): p for p in files}
            for f in tqdm(as_completed(futures), total=len(futures), ncols=100, desc="[meta]"):
                try:
                    loaded.append(f.result())
                except Exception as e:
                    logger.warning(f"[meta] Failed {futures[f]}: {e}")
        logger.info(f"[meta] Scan done, {len(loaded)} files")

        try:
            with open(cache_path, "w") as f:
                json.dump(loaded, f, indent=2)
            logger.info(f"[meta] Cached meta to {cache_path}")
        except Exception as e:
            logger.warning(f"[meta] Cache write failed: {e}")

        if distributed:
            dist.barrier()
        return loaded

    # Non-main ranks: wait for rank-0 to finish and then read the cache.
    if distributed:
        dist.barrier()
    with open(cache_path, "r") as f:
        cached = json.load(f)
    return cached


class FullSequenceLatentDataset(Dataset):
    """Sliding-window sampler over precomputed ``(image_embed, latent)`` npz files."""

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
        self.seq_len = seq_len
        self.image_key = image_key
        self.latent_key = latent_key
        self.frame_step = frame_step
        self.hop = hop or seq_len
        self.drop_last = drop_last
        self.preload = preload
        self.mmap_mode = None if preload else mmap_mode
        self.num_threads = num_threads

        if isinstance(npz_dirs, str):
            npz_dirs = [d.strip() for d in npz_dirs.split(",") if d.strip()]
        elif not isinstance(npz_dirs, (list, tuple)):
            raise TypeError(f"npz_dirs must be str or list, got {type(npz_dirs)}")

        files: List[str] = []
        for d in npz_dirs:
            if not os.path.isdir(d):
                raise ValueError(f"Not a directory: {d}")
            for root, _, fnames in os.walk(d):
                for f in fnames:
                    if f.endswith(".npz"):
                        files.append(os.path.join(root, f))
        files.sort(key=_natural_sort_key)

        if skip_json:
            skip_path = (
                skip_json
                if os.path.isabs(skip_json) or os.path.exists(skip_json)
                else os.path.join(os.path.dirname(npz_dirs[0]), f"{skip_json}.json")
            )
            if os.path.exists(skip_path):
                with open(skip_path, "r") as f:
                    skip_dict = json.load(f)
                skip_set = set()
                for v in skip_dict.values():
                    skip_set.update(v)
                logger.info(f"[dataset] skip list: {len(skip_set)} sequences from {skip_path}")
                kept = [f for f in files if os.path.basename(os.path.dirname(f)) not in skip_set]
                logger.info(f"[dataset] files {len(files)} -> {len(kept)} after skip filtering")
                files = kept
            else:
                logger.warning(f"[dataset] skip_json not found: {skip_path}")

        if not files:
            raise FileNotFoundError(f"No npz files found in {npz_dirs}")
        self.files = files

        cache_path = os.path.join(os.path.dirname(npz_dirs[0]), "dataset_meta_cache.json")
        self._loaded = load_or_cache_meta(
            self.files,
            num_threads=self.num_threads,
            preload=self.preload,
            mmap_mode=self.mmap_mode,
            cache_path=cache_path,
        )

        # Build sliding-window index: (file_idx, start_frame).
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
                    "No valid sequence windows. Reduce seq_len/frame_step or set drop_last=False."
                )
            self._index.append((0, 0))

    def __len__(self):
        return len(self._index)

    def _load_arrays(self, fi: int):
        meta = self._loaded[fi]
        if self.preload and "img" in meta:
            return meta["img"], meta["lat"]
        arr = np.load(meta["path"], mmap_mode=self.mmap_mode)
        return arr[self.image_key], arr[self.latent_key]

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

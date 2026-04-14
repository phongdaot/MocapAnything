
### loader_v1_mesh.py ###
# Dataset for training video2mesh (TripoSG-style flow-matching DiT).
#
# Consumes the .npz files produced by preprocess/preprocess_data.py, which
# contain per-frame pairs of (VAE-encoded mesh latent, DinoV2 image embedding).
# Each .npz holds:
#     latent:      [T, num_tokens, latent_dim]       e.g. [T, 2048, 64]
#     image_embed: [T, cond_tokens, cond_dim]        e.g. [T, 257, 1024]
#
# The dataset walks a root directory recursively for .npz files, applies an
# optional JSON split, and yields fixed-length windows suitable for the
# temporal DiT in model/v1/video2mesh.

import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _find_all_npz(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".npz"):
                out.append(os.path.join(dirpath, f))
    out.sort()
    return out


def _load_split_ids(split_json: Optional[str], split_mode: str) -> Optional[set]:
    """
    Load a set of allowed sample ids from a split JSON.

    Expected JSON layout (flexible):
        {"train": [...], "test": [...]} -> list of relative paths or ids
    Returns None when no split filtering should be applied.
    """
    if split_json is None or not os.path.exists(split_json):
        return None

    with open(split_json, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        ids = data
    elif isinstance(data, dict):
        ids = data.get(split_mode, None)
        if ids is None:
            return None
    else:
        return None

    return set(str(x) for x in ids)


class VideoMeshLatentDataset(Dataset):
    """
    Dataset of pre-computed (mesh latent, image embedding) sequences for
    training the video2mesh temporal DiT with flow matching.
    """

    def __init__(
        self,
        npz_dir: str,
        window: int = 16,
        split_json: Optional[str] = None,
        split_mode: str = "train",
        min_frames: Optional[int] = None,
        preload_all: bool = False,
        debug_limit: Optional[int] = None,
    ):
        super().__init__()

        self.npz_dir = npz_dir
        self.window = int(window)
        self.split_mode = split_mode
        self.preload_all = preload_all
        self.min_frames = min_frames if min_frames is not None else self.window

        all_npz = _find_all_npz(npz_dir)
        allowed = _load_split_ids(split_json, split_mode)

        items: List[Dict[str, Any]] = []
        for p in all_npz:
            rel = os.path.relpath(p, npz_dir)
            sample_id = os.path.splitext(rel)[0]  # drop .npz

            if allowed is not None and sample_id not in allowed and rel not in allowed:
                continue

            items.append({
                "path": p,
                "rel": rel,
                "sample_id": sample_id,
            })

            if debug_limit is not None and len(items) >= debug_limit:
                break

        # Filter + cache length metadata.
        self.items: List[Dict[str, Any]] = []
        self._preloaded: Dict[str, Dict[str, np.ndarray]] = {}

        for it in items:
            try:
                with np.load(it["path"], allow_pickle=False) as z:
                    n = z["latent"].shape[0]
                    ne = z["image_embed"].shape[0]
                    if n != ne:
                        continue
                    if n < self.min_frames:
                        continue
                    it["num_frames"] = int(n)
                    it["latent_shape"] = tuple(z["latent"].shape)
                    it["embed_shape"] = tuple(z["image_embed"].shape)
                    if preload_all:
                        self._preloaded[it["path"]] = {
                            "latent": z["latent"].astype(np.float32),
                            "image_embed": z["image_embed"].astype(np.float32),
                        }
                self.items.append(it)
            except Exception as e:
                # Skip corrupted or incomplete files silently.
                continue

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, it: Dict[str, Any]):
        if it["path"] in self._preloaded:
            arrs = self._preloaded[it["path"]]
            return arrs["latent"], arrs["image_embed"]
        with np.load(it["path"], allow_pickle=False) as z:
            latent = z["latent"].astype(np.float32)
            image_embed = z["image_embed"].astype(np.float32)
        return latent, image_embed

    def _pick_window(self, n: int) -> int:
        if n <= self.window:
            return 0
        if self.split_mode == "train":
            return random.randint(0, n - self.window)
        # Deterministic window for eval: center crop.
        return max(0, (n - self.window) // 2)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        latent, image_embed = self._load(it)
        n = latent.shape[0]

        start = self._pick_window(n)
        end = start + self.window

        lat_win = latent[start:end]            # [T, N, D_latent]
        img_win = image_embed[start:end]       # [T, C, D_cond]

        # Pad at the end if the clip is shorter than the window.
        if lat_win.shape[0] < self.window:
            pad = self.window - lat_win.shape[0]
            lat_win = np.concatenate(
                [lat_win, np.repeat(lat_win[-1:], pad, axis=0)], axis=0
            )
            img_win = np.concatenate(
                [img_win, np.repeat(img_win[-1:], pad, axis=0)], axis=0
            )

        return {
            "latent": torch.from_numpy(lat_win).float(),          # [T, N, D_latent]
            "image_embed": torch.from_numpy(img_win).float(),     # [T, C, D_cond]
            "sample_id": it["sample_id"],
            "start": start,
        }


def collate_video_mesh(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "latent": torch.stack([b["latent"] for b in batch], dim=0),           # [B, T, N, D]
        "image_embed": torch.stack([b["image_embed"] for b in batch], dim=0), # [B, T, C, D]
        "sample_id": [b["sample_id"] for b in batch],
        "start": torch.tensor([b["start"] for b in batch], dtype=torch.long),
    }
    return out

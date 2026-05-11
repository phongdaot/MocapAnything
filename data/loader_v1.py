### loader_v1.py ###
from torch.utils.data import Dataset
import torch.distributed as dist
import torch
import tqdm
import random
from typing import Optional, List, Dict, Tuple, Any
import os
import json
import pickle
import numpy as np
import time
from mmengine import Config
from mmengine.runner.checkpoint import load_checkpoint as mm_load_checkpoint
from utils.logger import logger
from utils.dist_utils import cleanup_distributed, is_main_process

MAX_JOINTS=150

# ============================================================
# Dataset
# ============================================================
class AnySpeciesPoseDataset(Dataset):
    def __init__(
            self,
            bvh_dir: str,
            window: int = 48,
            use_rot6d: bool = False,
            use_position: bool = True,
            mmap: bool = True,
            cache_scale: bool = True,
            limit_species_debug: Optional[List[str]] = None,
            split_json: Optional[str] = None,  # ← newly added
            split_mode: Optional[str] = None,  # ← 'train' or 'test'
            split_group: Optional[str] = None,
            mesh_dir: Optional[str] = None,  # add optional mesh replacement (not GT mesh)
            max_mesh_points: int = 1024  # ← number of sampled mesh points per frame
    ):
        self.bvh_dir = bvh_dir
        root = os.path.dirname(bvh_dir)
        self.pose_dir = os.path.join(root, "bvh_pose")
        self.train_dir = os.path.join(root, "npz_train")
        if mesh_dir is not None:
            self.mesh_dir = mesh_dir
        else:
            self.mesh_dir = os.path.join(root, "npz_mesh_normed")
        self.species_info_dict = np.load(os.path.join(os.path.dirname(bvh_dir), "species_info_dict.npy"),
                                         allow_pickle=True).item()

        self.window = window
        self.use_rot6d = use_rot6d
        self.use_position = use_position
        self.mmap = mmap
        self.cache_scale = cache_scale
        self.limit_species_debug = set(limit_species_debug) if limit_species_debug else None
        self.split_mode = split_mode
        self.max_mesh_points = max_mesh_points

        # === load test split JSON ===
        self.test_seq_set = set()
        self.test_angle_map = {}
        if split_json and os.path.isfile(split_json):
            with open(split_json, "r") as f:
                split_dict = json.load(f)
            for group in ["seen", "rare", "unseen"]:
                if group in split_dict:
                    for seq, angle in split_dict[group].items():
                        self.test_seq_set.add(seq)
                        if seq not in self.test_angle_map:
                            self.test_angle_map[seq] = []
                        # self.test_angle_map[seq].append(f'y{angle}')
                        # check if angle is a list
                        if isinstance(angle, list):
                            for a in angle:
                                self.test_angle_map[seq].append(f'y{a}')
                        else:
                            self.test_angle_map[seq].append(f'y{angle}')
            logger.info(f"Loaded split_json: {len(self.test_seq_set)} test sequences")
        else:
            logger.info("No split_json provided or file not found.")

        self.split_group = split_group
        self.group_seq_set = set()
        if split_json and os.path.isfile(split_json):
            with open(split_json, "r") as f:
                split_dict = json.load(f)
            if self.split_group is not None and self.split_group in split_dict:
                for seq, angle in split_dict[self.split_group].items():
                    self.group_seq_set.add(seq)

        # === scale cache ===
        self.scale_dict_path = os.path.join(root, "__mesh2pose1002_species_scale_cache.pkl")
        self.scale_dict = self._load_scale_cache() if cache_scale else {}

        # === meta file ===
        self.items = self._build_items()

        # === cache all species scale during initialization ===
        self._init_species_scale_cache()
        logger.info(f"Final usable sequences: {len(self.items)}")

    # ============================================================
    # Scale cache load / save
    # ============================================================
    def _load_scale_cache(self) -> Dict[str, float]:
        if os.path.isfile(self.scale_dict_path):
            with open(self.scale_dict_path, "rb") as f:
                self.scale_dict = pickle.load(f)
            logger.info(f"Loaded species scale cache ({len(self.scale_dict.keys())} species): {self.scale_dict_path}")
            return self.scale_dict
        return {}

    def _save_scale_cache(self):
        if self.cache_scale and self.scale_dict:
            with open(self.scale_dict_path, "wb") as f:
                pickle.dump(self.scale_dict, f)
            logger.info(f"Saved species scale cache ({len(self.scale_dict.keys())} species): {self.scale_dict_path}")

    # ============================================================
    # Build meta
    # ============================================================
    def _build_items(self) -> List[Dict[str, Any]]:
        items, skipped = [], []

        for dirpath, _, fnames in os.walk(self.bvh_dir):
            for f in fnames:
                if not f.endswith(".bvh"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, f), self.bvh_dir)
                stem = os.path.splitext(rel)[0]
                seq_name = stem.split('/')[0]
                angle_str = stem.split('/')[-1]

                # species filtering (match prefix before '#')
                if self.limit_species_debug:
                    sp_full = stem.split('/')[0]
                    sp_short = sp_full.split('#')[0]
                    if sp_short not in self.limit_species_debug:
                        continue

                # === filter according to split_mode ===
                if self.split_mode == "train":
                    if seq_name in self.test_seq_set:
                        continue

                elif self.split_mode == "test":
                    if self.split_group is not None:
                        if seq_name not in self.group_seq_set:
                            continue
                    else:
                        if seq_name not in self.test_seq_set:
                            continue
                    valid_angles = self.test_angle_map.get(seq_name, None)
                    if valid_angles is not None and angle_str not in valid_angles:
                        continue

                pose_p = os.path.join(self.pose_dir, f"{stem}.npz")
                train_p = os.path.join(self.train_dir, f"{stem}.npz")
                mesh_p = os.path.join(self.mesh_dir, f"{stem}.npz")
                if not (os.path.isfile(pose_p) and os.path.isfile(train_p)):
                    continue

                pose_npz = np.load(pose_p, mmap_mode='r' if self.mmap else None)
                F, J = pose_npz['position'].shape[:2]
                pose_npz.close()
                if F < self.window:
                    skipped.append((stem, F))
                    continue

                items.append({
                    "rel": stem,
                    "pose": pose_p,
                    "train": train_p,
                    "mesh": mesh_p if os.path.isfile(mesh_p) else "",
                    "F": int(F),
                    "J": int(J),
                })

        logger.info(f"Total sequences scanned: {len(items) + len(skipped)}")
        logger.info(f"Valid sequences: {len(items)} (F ≥ {self.window})")
        logger.info(f"Skipped short sequences: {len(skipped)}")
        if skipped:
            for s, f in skipped[:8]:
                logger.info(f"  - {s} (F={f})")

        if items:
            j_stats = {}
            for it in items:
                j_stats[it["J"]] = j_stats.get(it["J"], 0) + 1
            logger.info("Joint count distribution:")
            for jn, cnt in sorted(j_stats.items()):
                logger.info(f"  {jn:3d} joints: {cnt} sequences")

        return items

    # ============================================================
    # Initialize species scale cache
    # ============================================================
    def _init_species_scale_cache(self):
        species_to_mesh = {}
        for it in self.items:
            sp = it["rel"].split("/")[0].split("#")[0]
            if sp not in species_to_mesh and it["mesh"]:
                species_to_mesh[sp] = it["mesh"]

        new_scales = 0
        for sp, mesh_path in species_to_mesh.items():
            if sp in self.scale_dict:
                continue
            if os.path.isfile(mesh_path):
                mesh_npz = np.load(mesh_path, mmap_mode='r' if self.mmap else None)
                gscale = mesh_npz.get("global_scale", None)
                goffset = mesh_npz.get("bbox_center", None)
                mesh_npz.close()
                if gscale is not None:
                    self.scale_dict[sp] = {'global_scale': gscale, 'bbox_center': goffset}
                    new_scales += 1
                    logger.info(f"Loaded species {sp:<20s} global_scale = {gscale}")
                    logger.info(f"Loaded species {sp:<20s} bbox_center = {goffset}")
        if new_scales > 0:
            self._save_scale_cache()
        else:
            logger.info("All species scales are already cached")

    # ============================================================
    # Random window + reference frame sampling
    # ============================================================
    def _rand_window(self, F: int) -> Tuple[int, int]:
        if F <= self.window:
            return 0, F
        t0 = random.randint(0, F - self.window)
        return t0, t0 + self.window

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        pose_npz = np.load(it['pose'], mmap_mode='r' if self.mmap else None)
        train_npz = np.load(it['train'], mmap_mode='r' if self.mmap else None)
        mesh_npz = np.load(it['mesh'], mmap_mode='r' if self.mmap else None)

        position = pose_npz['position']
        rot6d = pose_npz['rot6d'] if ('rot6d' in pose_npz and self.use_rot6d) else None
        image_embed = train_npz['image_embed']
        vertices_normed = mesh_npz['vertices_normed']
        surface_normal = mesh_npz['normals']

        _, J = position.shape[:2]
        F, _ = vertices_normed.shape[:2]
        t0, t1 = self._rand_window(F)
        W = t1 - t0
        ref_idx = random.randint(0, F - 1)

        # === retrieve cached scale ===
        species_name = it["rel"].split("/")[0].split("#")[0]
        assert species_name in self.scale_dict, f"Species {species_name} missing global_scale!"
        gscale = self.scale_dict[species_name]['global_scale']
        goffset = self.scale_dict[species_name]['bbox_center']

        # === alignment & normalization ===
        position = (position - position[:, 0:1, :]) / gscale

        # === fixed mesh point sampling ===
        V = vertices_normed.shape[1]
        if V > self.max_mesh_points:
            idx = np.random.choice(V, self.max_mesh_points, replace=False)
            vertices_normed = vertices_normed[:, idx, :]
            surface_normal = surface_normal[:, idx, :]
        elif V < self.max_mesh_points:
            pad = np.zeros((F, self.max_mesh_points - V, 3), dtype=np.float32)
            vertices_normed = np.concatenate([vertices_normed, pad], axis=1)
            surface_normal = np.concatenate([surface_normal, pad], axis=1)

        # === extract window and reference ===
        pos_win = position[t0:t1]
        rot_win = rot6d[t0:t1] if rot6d is not None else None
        img_win = image_embed[t0:t1]
        mesh_win = vertices_normed[t0:t1]
        normals_win = surface_normal[t0:t1]

        ref_pos = position[ref_idx]
        ref_rot = pose_npz['rot6d'][ref_idx] if rot6d is not None else None
        ref_img = image_embed[ref_idx]
        ref_mesh = vertices_normed[ref_idx]
        ref_normals = surface_normal[ref_idx]

        pose_npz.close()
        train_npz.close()
        mesh_npz.close()

        # === topology structure extraction ===
        hop_mat = self.species_info_dict[species_name]['joints_distance']
        edge_mat = self.species_info_dict[species_name]['joint_relation']
        # assert hop_mat.shape[0] == J, "shape check for hop_mat"
        joint_t5embed = self.species_info_dict[species_name]['t5_embedding']

        # === load static joint mask ===
        static_joint_ids = self.species_info_dict[species_name].get("static_joints", [])
        static_mask = np.zeros((J,), dtype=np.bool_)
        static_mask[static_joint_ids] = True

        return {
            "rel": it["rel"],
            "species": species_name,
            "F": F,
            "J": J,
            "W": W,
            "global_scale": gscale,
            "position": pos_win.astype(np.float32),
            "rot6d": rot_win.astype(np.float32) if rot_win is not None else None,
            "image_embed": img_win.astype(np.float32),
            # "mesh": mesh_win.astype(np.float32),
            "mesh": mesh_win.astype(np.float32),
            "surface_normal": normals_win.astype(np.float32),
            "ref_position": ref_pos.astype(np.float32),
            "ref_rot6d": ref_rot.astype(np.float32) if ref_rot is not None else None,
            "ref_image_embed": ref_img.astype(np.float32),
            # "ref_mesh": ref_mesh.astype(np.float32),
            "ref_mesh": ref_mesh.astype(np.float32),
            "ref_surface_normal": ref_normals.astype(np.float32),
            "graph_hop": hop_mat.astype(np.int64), # J, J
            "graph_edge": edge_mat.astype(np.int64), # J, J
            "joint_t5embed": joint_t5embed.astype(np.float32), # J, D
            "static_mask": static_mask,
        }
        

def collate_anyspecies_padded(batch):
    """
    将任意物种的 pose / mesh 数据打包成统一 batch。
    支持不同关节数的序列，自动 padding 对齐。
    mesh 已在 Dataset 内部采样为固定点数，不再额外处理。
    """
    B = len(batch)
    J_max = MAX_JOINTS
    W = batch[0]["W"]

    # ====== 各项容器 ======
    pos_list, rot_list, img_list, mesh_list = [], [], [], []
    ref_pos_list, ref_rot_list, ref_img_list, ref_mesh_list = [], [], [], []
    joint_mask_list, scale_list, J_valid_list, species_list = [], [], [], []
    normals_list, ref_normals_list = [], []
    hop_list, edge_list = [], []
    joint_t5_list = []
    static_mask_list = []

    # ====== 遍历样本 ======
    for b in batch:
        J = b["J"]
        J_valid_list.append(J)
        species_list.append(b["species"])
        scale_list.append(b["global_scale"])

        # --- joints embedding ---
        joint_t5 = b["joint_t5embed"]
        if J < J_max:
            pad = np.zeros((J_max - J, joint_t5.shape[-1]), dtype=np.float32)
            joint_t5 = np.concatenate([joint_t5, pad], axis=0)
        joint_t5_list.append(joint_t5 if J <= J_max else joint_t5[:J_max])

        # --- static mask ---
        static_mask = b["static_mask"]
        if J < J_max:
            pad = np.zeros((J_max - J,), dtype=np.bool_)
            static_mask = np.concatenate([static_mask, pad])
        else:
            static_mask = static_mask[:J_max]
        static_mask_list.append(static_mask)

        # --- joints mask ---
        mask = np.zeros((J_max,), dtype=np.bool_)
        mask[:min(J, J_max)] = True
        joint_mask_list.append(mask)

        # --- hop/edge ---
        hop = b["graph_hop"]
        edge = b["graph_edge"]
        # pad 到 J_max
        hop_pad = np.full((MAX_JOINTS, MAX_JOINTS), fill_value=5, dtype=np.int64)  # 5: padding的最远距离
        edge_pad = np.full((MAX_JOINTS, MAX_JOINTS), fill_value=4, dtype=np.int64)  # 4: padding的边类型
        hop_pad[:J, :J] = hop
        edge_pad[:J, :J] = edge
        hop_list.append(hop_pad)
        edge_list.append(edge_pad)

        # --- position ---
        pos = b["position"]
        if J < J_max:
            pad = np.zeros((W, J_max - J, 3), dtype=np.float32)
            pos = np.concatenate([pos, pad], axis=1)
        pos_list.append(pos if J <= J_max else pos[:, :J_max])

        # --- rot6d ---
        rot = b["rot6d"]
        if rot is not None:
            if J < J_max:
                pad = np.zeros((W, J_max - J, 6), dtype=np.float32)
                rot = np.concatenate([rot, pad], axis=1)
        else:
            rot = np.zeros((W, J_max, 6), dtype=np.float32)
        rot_list.append(rot if J <= J_max else rot[:, :J_max])

        # --- image ---
        img_list.append(b["image_embed"])

        # --- mesh ---
        mesh_list.append(b["mesh"])  # [F, P, 3] 已经采样好了

        # --- surface_normal ---
        normals_list.append(b["surface_normal"])  # [F, P, 3]

        # --- ref position ---
        ref_pos = b["ref_position"]
        if J < J_max:
            pad = np.zeros((J_max - J, 3), dtype=np.float32)
            ref_pos = np.concatenate([ref_pos, pad], axis=0)
        ref_pos_list.append(ref_pos if J <= J_max else ref_pos[:J_max])

        # --- ref rot ---
        ref_rot = b["ref_rot6d"]
        if ref_rot is not None:
            if J < J_max:
                pad = np.zeros((J_max - J, 6), dtype=np.float32)
                ref_rot = np.concatenate([ref_rot, pad], axis=0)
        else:
            ref_rot = np.zeros((J_max, 6), dtype=np.float32)
        ref_rot_list.append(ref_rot if J <= J_max else ref_rot[:J_max])

        # --- ref image ---
        ref_img_list.append(b["ref_image_embed"])

        # --- ref mesh ---
        ref_mesh_list.append(b["ref_mesh"])  # [P, 3] 已经采样好了

        # --- ref surface_normal ---
        ref_normals_list.append(b["ref_surface_normal"])  # [P, 3]

    # ====== Stack 所有内容 ======
    return {
        "position": torch.from_numpy(np.stack(pos_list, 0)),  # [B, F, J, 3]
        "rot6d": torch.from_numpy(np.stack(rot_list, 0)),  # [B, F, J, 6]
        "image_embed": torch.from_numpy(np.stack(img_list, 0)),  # [B, F, P, D_img]
        "mesh": torch.from_numpy(np.stack(mesh_list, 0)),  # [B, F, P, 3]
        "surface_normal": torch.from_numpy(np.stack(normals_list, 0)),  # [B, F, P, 3]

        "ref_position": torch.from_numpy(np.stack(ref_pos_list, 0)),  # [B, J, 3]
        "ref_rot6d": torch.from_numpy(np.stack(ref_rot_list, 0)),  # [B, J, 6]
        "ref_image_embed": torch.from_numpy(np.stack(ref_img_list, 0)),  # [B, P, D_img]
        "ref_mesh": torch.from_numpy(np.stack(ref_mesh_list, 0)),  # [B, P, 3]
        "ref_surface_normal": torch.from_numpy(np.stack(ref_normals_list, 0)),  # [B, P, 3]

        "joint_mask": torch.from_numpy(np.stack(joint_mask_list, 0)),  # [B, J_max]
        "J_valid": torch.tensor(J_valid_list, dtype=torch.int32),
        # "global_scale": torch.tensor(scale_list, dtype=torch.float32),
        "global_scale": torch.from_numpy(np.array(scale_list)).float(),
        "species": species_list,

        "graph_hop": torch.from_numpy(np.stack(hop_list, 0)),  # [B, J_max, J_max]
        "graph_edge": torch.from_numpy(np.stack(edge_list, 0)),  # [B, J_max, J_max]
        "joint_t5embed": torch.from_numpy(np.stack(joint_t5_list, 0)),  # [B, J_max, D_t5]
        "static_mask": torch.from_numpy(np.stack(static_mask_list, 0)),  # [B, J_max]
    }

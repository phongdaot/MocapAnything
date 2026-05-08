### loader_v2.py ###
from torch.utils.data import Dataset
import torch.distributed as dist
import torch
from tqdm import tqdm
import random
from typing import Optional, List, Dict, Tuple, Any
import os
import json
import pickle
import numpy as np
import time
from utils.dist_utils import is_main_process
from utils.logger import logger
from utils.bvhreader import BVHReader

MAX_JOINTS=150
# ============================================================
# Dataset
# ============================================================
class AnySpeciesPoseDataset(Dataset):
    def __init__(
        self,
        bvh_dir: str,
        window: int = 48,
        use_position: bool = True,
        mmap: bool = True,
        cache_scale: bool = True,
        limit_species_debug: Optional[List[str]] = None,
        split_json: Optional[str] = None,
        split_mode: Optional[str] = None,   # 'train' or 'test'
        split_group: Optional[str] = None,  # 'seen' / 'rare' / 'unseen'
        max_mesh_points: int = 1024,
        memory_pkl_path: Optional[str] = None,
        preload_all: bool = True,
    ):
        self.bvh_dir = bvh_dir
        root = os.path.dirname(bvh_dir)
        self.pose_dir = os.path.join(root, "bvh_pose")
        self.train_dir = os.path.join(root, "npz_train_image_only")
        self.species_info_dict = np.load(
            os.path.join(os.path.dirname(bvh_dir), "species_info_dict.npy"),
            allow_pickle=True
        ).item()

        self.window = window
        self.use_rot6d = True
        self.use_position = use_position
        self.mmap = mmap
        self.cache_scale = cache_scale
        self.limit_species_debug = set(limit_species_debug) if limit_species_debug else None
        self.split_mode = split_mode
        self.split_group = split_group
        self.max_mesh_points = max_mesh_points
        self.preload_all = preload_all

        # =========================================================
        # just add memory pkl
        # pkl format:
        # {
        #   species_name: {
        #       "rot6d": [N, J, 6],
        #       "pose_normed": [N, J, 3]
        #   }
        # }
        # =========================================================
        self.memory_pkl_path = memory_pkl_path
        self.species_memory_dict = {}
        if self.memory_pkl_path is None:
            raise ValueError("memory_pkl_path is missing")
        with open(self.memory_pkl_path, "rb") as f:
            self.species_memory_dict = pickle.load(f)
        if not isinstance(self.species_memory_dict, dict):
            raise ValueError(f"memory_pkl_path is not a dict: {self.memory_pkl_path}")
        logger.info(
            f"loaded species memory pkl: {self.memory_pkl_path}, "
            f"species num = {len(self.species_memory_dict)}"
        )

        self.bvh_reader = BVHReader(
            max_num_joints=150,
            crop_size=600,
            no_pos=True,
            bvh_norm=False,
            reset_pose_prob=1.0,
        )

        self.bvh_reader_test = BVHReader(
            max_num_joints=150,
            crop_size=600,
            no_pos=True,
            bvh_norm=False,
            reset_pose_prob=0.0,
        )

        # =========================================================
        # split json
        # =========================================================
        self.test_seq_set = set()
        self.test_angle_map = {}
        self.group_seq_set = set()

        if split_json and os.path.isfile(split_json):
            with open(split_json, "r") as f:
                split_dict = json.load(f)

            for group in ["seen", "rare", "unseen"]:
                if group in split_dict:
                    for seq, angle in split_dict[group].items():
                        self.test_seq_set.add(seq)

                        if seq not in self.test_angle_map:
                            self.test_angle_map[seq] = []

                        if isinstance(angle, list):
                            for a in angle:
                                self.test_angle_map[seq].append(f"y{a}")
                        else:
                            self.test_angle_map[seq].append(f"y{angle}")

            if self.split_group is not None:
                if self.split_group not in split_dict:
                    raise ValueError(
                        f"split_group={self.split_group} is not in split_json, available: {list(split_dict.keys())}"
                    )
                for seq in split_dict[self.split_group].keys():
                    self.group_seq_set.add(seq)

            logger.info(f"Loaded split_json: {len(self.test_seq_set)} test sequences")
            if self.split_group is not None:
                logger.info(f"Using split_group={self.split_group}, seq num = {len(self.group_seq_set)}")
        else:
            logger.info("No split_json provided or file not found.")
        # =========================================================
        # scale cache
        # =========================================================
        self.scale_dict_path = os.path.join(root, "cache/__mesh2pose1002_species_scale_cache.pkl")
        self.scale_dict = self._load_scale_cache() if cache_scale else {}

        # =========================================================
        # meta cache
        # =========================================================
        self.items_cache_path = os.path.join(
            root,
            f"cache/__mesh2pose1002_{self.split_mode}_{self.split_group}_seqlen_{self.window}_items_cache.json"
        )

        # =========================================================
        # rank0 is responsible for building/caching, other ranks wait and read
        # =========================================================
        if is_main_process():
            if os.path.exists(self.items_cache_path):
                logger.info(f"Cache detected: {self.items_cache_path}, loading directly ✅")
                with open(self.items_cache_path, "r") as f:
                    self.items = json.load(f)
                logger.info(f"Loaded {len(self.items)} sequences from cache")
            else:
                logger.info("No cache detected, start scanning BVH files and building items ...")
                self.items = self._build_items()
                with open(self.items_cache_path, "w") as f:
                    json.dump(self.items, f)
                logger.info(f"Build complete and cache saved: {len(self.items)} sequences → {self.items_cache_path}")
        else:
            while not os.path.exists(self.items_cache_path):
                time.sleep(1)
            with open(self.items_cache_path, "r") as f:
                self.items = json.load(f)
            logger.info(f"rank{dist.get_rank()} loaded cache, total {len(self.items)} sequences")

        logger.info(f"Final available sequence count: {len(self.items)}")

        # =========================================================
        # build per-species static cache
        # =========================================================
        self.species_static_cache = self._build_species_static_cache()

        # =========================================================
        # preload all sequences into RAM
        # =========================================================
        self.data_cache = None
        if self.preload_all:
            self.data_cache = self._preload_all_data()
            logger.info(f"preload finished: {len(self.data_cache)} sequences in RAM")
        else:
            logger.info("preload_all=False, will still read on demand")

    # ============================================================
    # Scale cache load / save
    # ============================================================
    def _load_scale_cache(self) -> Dict[str, Any]:
        if os.path.isfile(self.scale_dict_path):
            with open(self.scale_dict_path, "rb") as f:
                self.scale_dict = pickle.load(f)
            logger.info(f"Loaded species scale cache ({len(self.scale_dict.keys())} species): {self.scale_dict_path}")
            return self.scale_dict
        logger.warning(f"[WARN] scale cache does not exist: {self.scale_dict_path}")
        return {}

    def _save_scale_cache(self):
        if self.cache_scale and self.scale_dict:
            with open(self.scale_dict_path, "wb") as f:
                pickle.dump(self.scale_dict, f)
            logger.info(f"Saved species scale cache ({len(self.scale_dict.keys())} species): {self.scale_dict_path}")

    # ============================================================
    # build meta
    # ============================================================
    def _build_items(self) -> List[Dict[str, Any]]:
        items, skipped = [], []

        all_bvh_files = []
        for dirpath, _, fnames in os.walk(self.bvh_dir):
            for f in fnames:
                if f.endswith(".bvh"):
                    all_bvh_files.append(os.path.join(dirpath, f))

        logger.info(f"Start scanning BVH files, total {len(all_bvh_files)} ...")

        for bvh_fpath in tqdm(all_bvh_files, desc="Building dataset meta", ncols=100):
            rel = os.path.relpath(bvh_fpath, self.bvh_dir)
            stem = os.path.splitext(rel)[0]
            seq_name = stem.split('/')[0]
            angle_str = stem.split('/')[-1]

            # species filtering (match before #)
            if self.limit_species_debug:
                sp_full = stem.split('/')[0]
                sp_short = sp_full.split('#')[0]
                if sp_short not in self.limit_species_debug:
                    continue

            # split filtering
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

            if not os.path.isfile(pose_p):
                continue
            if not os.path.isfile(train_p):
                continue

            pose_npz = np.load(pose_p, mmap_mode='r' if self.mmap else None)
            F, J = pose_npz['position'].shape[:2]
            pose_npz.close()

            if F < self.window:
                skipped.append((stem, F))
                continue

            items.append({
                "bvh": bvh_fpath,
                "rel": stem,
                "pose": pose_p,
                "train": train_p,
                "F": int(F),
                "J": int(J),
            })

        logger.info(f"Total scanned sequences: {len(items) + len(skipped)}")
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
    # build per-species static cache
    # ============================================================
    def _build_species_static_cache(self) -> Dict[str, Dict[str, Any]]:
        species_static_cache = {}
        skipped_species = []

        all_species = all_species = sorted({
            it["rel"].split("/")[0].split("#")[0]
            for it in self.items
        })
        if self.limit_species_debug is not None:
            all_species = [sp for sp in all_species if sp in self.limit_species_debug]

        for species_name in all_species:
            if species_name not in self.species_info_dict:
                skipped_species.append((species_name, "missing in species_info_dict"))
                continue

            if species_name not in self.scale_dict:
                skipped_species.append((species_name, "missing in scale_dict"))
                continue

            info = self.species_info_dict[species_name]

            hop_mat = info['joints_distance'].astype(np.int64)
            edge_mat = info['joint_relation'].astype(np.int64)
            joint_t5embed = info['t5_embedding'].astype(np.float32)

            J = hop_mat.shape[0]

            # static joints for rotation
            static_rot_joint_ids = info['static_rot_joints']
            static_rot_joint_mask = np.zeros((J,), dtype=np.bool_)
            static_rot_joint_mask[static_rot_joint_ids] = True

            # static joints for position
            static_pos_joint_ids = info['static_joints']
            static_pos_joint_mask = np.zeros((J,), dtype=np.bool_)
            static_pos_joint_mask[static_pos_joint_ids] = True

            memory_pose = None
            memory_rot6d = None
            has_memory = False
            if species_name in self.species_memory_dict:
                sp_mem = self.species_memory_dict[species_name]
                if ("pose_normed" in sp_mem) and ("rot6d" in sp_mem):
                    memory_pose = sp_mem["pose_normed"].astype(np.float32)
                    memory_rot6d = sp_mem["rot6d"].astype(np.float32)
                    if memory_pose.shape[0] > 0 and memory_rot6d.shape[0] > 0:
                        has_memory = True

            gscale = self.scale_dict[species_name]['global_scale']

            species_static_cache[species_name] = {
                "graph_hop": hop_mat,
                "graph_edge": edge_mat,
                "joint_t5embed": joint_t5embed,
                "static_rot_joint_mask": static_rot_joint_mask,
                "static_pos_joint_mask": static_pos_joint_mask,
                "memory_pose": memory_pose,
                "memory_rot6d": memory_rot6d,
                "has_memory": has_memory,
                "global_scale": np.float32(gscale),
            }

        logger.info(f"built species_static_cache: {len(species_static_cache)} species")
        if skipped_species:
            logger.warning(f"skipped species in static cache: {len(skipped_species)}")
            for sp, reason in skipped_species[:10]:
                logger.info(f"  - {sp}: {reason}")
                
        no_mem_species = [sp for sp, cache in species_static_cache.items() if not cache["has_memory"]]

        if no_mem_species:
            logger.warning(f"species without memory, will fallback to ref memory: {len(no_mem_species)}")
            for sp in no_mem_species[:10]:
                logger.info(f"  - {sp}")
        
        return species_static_cache

    # ============================================================
    # preload all sequences into RAM
    # ============================================================
    def _preload_all_data(self) -> List[Dict[str, Any]]:
        data_cache = []
        skipped = []

        iterator = tqdm(self.items, desc="Preloading all dataset to RAM", ncols=100)

        for it in iterator:
            bvh_fpath = it["bvh"]
            rel = it["rel"]
            species_name = rel.split("/")[0].split("#")[0]

            if species_name not in self.species_static_cache:
                skipped.append((rel, f"{species_name} missing in species_static_cache"))
                continue

            pose_npz = np.load(it["pose"], allow_pickle=False)
            train_npz = np.load(it["train"], allow_pickle=False)

            position = pose_npz["position"].astype(np.float32)   # [F, J, 3]
            rot6d = pose_npz["rot6d"].astype(np.float32)         # [F, J, 6]
            image_embed = train_npz["image_embed"].astype(np.float32)

            F_pose = position.shape[0]
            F_img  = image_embed.shape[0]
            if not F_pose == F_pose:
                logger.info(f'{bvh_fpath} frame cut')
            F = min(F_pose, F_img)

            position = position[:F]
            rot6d = rot6d[:F]
            image_embed = image_embed[:F]

            F, J = position.shape[:2]

            res = {'motion_path': bvh_fpath}
            res['joint_rename'] = False
            res = self.bvh_reader_test(res)

            parents = np.array(res['parents'], dtype=np.int64)[None, :]
            rest_pose = np.array(res['rest_pose'], dtype=np.float32)[None, :, :]

            gscale = float(self.species_static_cache[species_name]["global_scale"])

            position = (position - position[:, 0:1, :]) / gscale
            rest_pose = (rest_pose - rest_pose[:, 0:1, :]) / gscale

            data_cache.append({
                "rel": rel,
                "species": species_name,
                "F": int(F),
                "J": int(J),
                "position": position,
                "rot6d_a": rot6d,
                "image_embed": image_embed,
                "parent_a": parents,
                "offset_a": rest_pose,
            })

            pose_npz.close()
            train_npz.close()

        logger.info(f"preload completed: {len(data_cache)} sequences")
        if skipped:
            logger.warning(f"preload skipped: {len(skipped)}")
            for name, reason in skipped[:10]:
                logger.info(f"  - {name}: {reason}")

        return data_cache

    def _load_single_item_from_disk(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]

        bvh_fpath = it["bvh"]
        rel = it["rel"]
        species_name = rel.split("/")[0].split("#")[0]

        if species_name not in self.species_static_cache:
            raise KeyError(f"{species_name} missing in species_static_cache")

        pose_npz = np.load(it["pose"], mmap_mode='r' if self.mmap else None, allow_pickle=False)
        train_npz = np.load(it["train"], mmap_mode='r' if self.mmap else None, allow_pickle=False)

        position = pose_npz["position"].astype(np.float32)
        rot6d = pose_npz["rot6d"].astype(np.float32)
        image_embed = train_npz["image_embed"].astype(np.float32)

        F_pose = position.shape[0]
        F_img  = image_embed.shape[0]
        F = min(F_pose, F_img)

        position = position[:F]
        rot6d = rot6d[:F]
        image_embed = image_embed[:F]

        F, J = position.shape[:2]

        res = {'motion_path': bvh_fpath}
        res['joint_rename'] = False
        res = self.bvh_reader_test(res)

        parents = np.array(res['parents'], dtype=np.int64)[None, :]
        rest_pose = np.array(res['rest_pose'], dtype=np.float32)[None, :, :]

        gscale = float(self.species_static_cache[species_name]["global_scale"])

        position = (position - position[:, 0:1, :]) / gscale
        rest_pose = (rest_pose - rest_pose[:, 0:1, :]) / gscale

        pose_npz.close()
        train_npz.close()

        return {
            "rel": rel,
            "species": species_name,
            "F": int(F),
            "J": int(J),
            "position": position,
            "rot6d_a": rot6d,
            "image_embed": image_embed,
            "parent_a": parents,
            "offset_a": rest_pose,
        }

    # ============================================================
    # randomly sample window
    # ============================================================
    def _rand_window(self, F: int) -> Tuple[int, int]:
        if F < self.window:
            raise AssertionError(f"data {F} smaller than {self.window}")
        t0 = random.randint(0, F - self.window)
        return t0, t0 + self.window

    def __len__(self):
        if self.preload_all and self.data_cache is not None:
            return len(self.data_cache)
        return len(self.items)

    # ============================================================
    # keep the original function to prevent references elsewhere from breaking
    # ============================================================
    def _load_motion(self, bvh_fpath, pose_npz):
        res = {'motion_path': bvh_fpath}
        res['joint_rename'] = False
        res = self.bvh_reader_test(res)

        rot6d = pose_npz['rot6d']
        parents = torch.from_numpy(np.array(res['parents'])).unsqueeze(0)
        rest_pose = torch.from_numpy(res['rest_pose']).float().unsqueeze(0)
        return rot6d, parents, rest_pose

    def _load_motion_augmented(self, bvh_fpath, pose_npz):
        res = {'motion_path': bvh_fpath}
        res['joint_rename'] = False
        res = self.bvh_reader(res)

        motion = torch.from_numpy(res['motion']).float().unsqueeze(0)
        rest_pose = torch.from_numpy(res['rest_pose']).float().unsqueeze(0)
        num_joints = torch.tensor(res['num_joints']).long().squeeze(-1).unsqueeze(0)
        joint_mask = torch.from_numpy(res['joint_mask']).float().unsqueeze(0)
        joint_names = [[n] for n in res['joint_names']]
        parents = torch.from_numpy(np.array(res['parents'])).unsqueeze(0)

        owo_joint_feat = None

        frame_count, joint_count, _ = pose_npz['rot6d'].shape
        rot6d = res['motion'][:frame_count, :joint_count, 3:]
        return rot6d, parents, rest_pose, owo_joint_feat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.preload_all and self.data_cache is not None:
            data = self.data_cache[idx]
        else:
            data = self._load_single_item_from_disk(idx)

        F = data["F"]
        J = data["J"]

        if self.split_mode == "train":
            t0, t1 = self._rand_window(F)
            ref_idx = random.randint(0, F - 1)
        else:
            t0 = 0
            t1 = min(self.window, F)
            ref_idx = 0
        W = t1 - t0

        species_name = data["species"]
        sp_cache = self.species_static_cache[species_name]

        pos_win = data["position"][t0:t1]
        ref_pos = data["position"][ref_idx]

        rot_win_a = data["rot6d_a"][t0:t1]
        ref_rot_a = data["rot6d_a"][ref_idx]

        img_win = data["image_embed"][t0:t1]
        ref_img = data["image_embed"][ref_idx]

        memory_pose = sp_cache["memory_pose"]
        memory_rot6d = sp_cache["memory_rot6d"]

        if (memory_pose is None) or (memory_rot6d is None):
            memory_pose = ref_pos[None, ...].astype(np.float32)      # [1, J, 3]
            memory_rot6d = ref_rot_a[None, ...].astype(np.float32)   # [1, J, 6]
            
        return {
            "rel": data["rel"],
            "species": species_name,
            "F": F,
            "J": J,
            "W": W,
            "global_scale": sp_cache["global_scale"],

            # ===== shared =====
            "position": pos_win.astype(np.float32),
            "ref_position": ref_pos.astype(np.float32),
            "graph_hop": sp_cache["graph_hop"],
            "graph_edge": sp_cache["graph_edge"],
            "joint_t5embed": sp_cache["joint_t5embed"],
            "static_rot_joint_mask": sp_cache["static_rot_joint_mask"],
            "static_pos_joint_mask": sp_cache["static_pos_joint_mask"],

            # ===== memory =====
            "memory_pose": memory_pose,
            "memory_rot6d": memory_rot6d,

            # ===== view a =====
            "rot6d_a": rot_win_a.astype(np.float32),
            "ref_rot6d_a": ref_rot_a.astype(np.float32),
            "parent_a": data["parent_a"],
            "offset_a": data["offset_a"],

            # ===== video2pose =====
            "image_embed": img_win.astype(np.float32),
            "ref_image_embed": ref_img.astype(np.float32),
        }


# ============================================================
# Collate
# ============================================================
MAX_JOINTS = 150


def collate_anyspecies_padded(batch):
    J_max = MAX_JOINTS
    W = batch[0]["W"]

    # ===== shared =====
    pos_list, img_list = [], []
    ref_pos_list, ref_img_list = [], []
    joint_mask_list, ancestor_mask_list = [], []
    scale_list, J_valid_list, species_list = [], [], []
    hop_list, edge_list = [], []
    joint_t5_list = []
    static_rot_joint_mask_list = []
    static_pos_joint_mask_list = []

    # ===== memory =====
    memory_pose_list = []
    memory_rot6d_list = []

    # ===== a =====
    rot_a_list = []
    ref_rot_a_list = []
    parent_a_list = []
    offset_a_list = []

    for b in batch:
        J = b["J"]
        J_valid_list.append(J)
        species_list.append(b["species"])
        scale_list.append(b["global_scale"])

        # -------------------------
        # joint_t5
        # -------------------------
        joint_t5 = b["joint_t5embed"]
        if J < J_max:
            pad = np.zeros((J_max - J, joint_t5.shape[-1]), dtype=np.float32)
            joint_t5 = np.concatenate([joint_t5, pad], axis=0)
        joint_t5_list.append(joint_t5[:J_max])

        # -------------------------
        # static_rot_joint_mask
        # -------------------------
        static_rot_joint_mask = b["static_rot_joint_mask"]
        if J < J_max:
            pad = np.zeros((J_max - J,), dtype=np.bool_)
            static_rot_joint_mask = np.concatenate([static_rot_joint_mask, pad])
        static_rot_joint_mask_list.append(static_rot_joint_mask[:J_max])

        # -------------------------
        # static_pos_joint_mask
        # -------------------------
        static_pos_joint_mask = b["static_pos_joint_mask"]
        if J < J_max:
            pad = np.zeros((J_max - J,), dtype=np.bool_)
            static_pos_joint_mask = np.concatenate([static_pos_joint_mask, pad])
        static_pos_joint_mask_list.append(static_pos_joint_mask[:J_max])

        # -------------------------
        # joint mask
        # -------------------------
        mask = np.zeros((J_max,), dtype=np.bool_)
        mask[:min(J, J_max)] = True
        joint_mask_list.append(mask)

        # -------------------------
        # ancestor mask
        # -------------------------
        ancestor_mask = np.zeros((J_max, J_max), dtype=np.bool_)
        parent = b["parent_a"].squeeze(0)
        for i in range(min(J, J_max)):
            ancestor_mask[i, i] = True
            p = parent[i]
            while p != -1:
                ancestor_mask[i, p] = True
                p = parent[p]
        ancestor_mask_list.append(ancestor_mask)

        # -------------------------
        # hop / edge
        # -------------------------
        hop = b["graph_hop"]
        edge = b["graph_edge"]
        hop_pad = np.full((J_max, J_max), fill_value=5, dtype=np.int64)
        edge_pad = np.full((J_max, J_max), fill_value=4, dtype=np.int64)
        hop_pad[:J, :J] = hop
        edge_pad[:J, :J] = edge
        hop_list.append(hop_pad)
        edge_list.append(edge_pad)

        # -------------------------
        # position
        # -------------------------
        pos = b["position"]
        if J < J_max:
            pad = np.zeros((W, J_max - J, 3), dtype=np.float32)
            pos = np.concatenate([pos, pad], axis=1)
        pos_list.append(pos[:, :J_max])

        # -------------------------
        # rot a
        # -------------------------
        rot_a = b["rot6d_a"]
        if J < J_max:
            pad = np.zeros((W, J_max - J, 6), dtype=np.float32)
            rot_a = np.concatenate([rot_a, pad], axis=1)
        rot_a_list.append(rot_a[:, :J_max])

        # -------------------------
        # image_embed
        # -------------------------
        img_list.append(b["image_embed"])

        # -------------------------
        # memory pose
        # -------------------------
        memory_pose = b["memory_pose"]  # [N, J, 3]
        if J < J_max:
            pad = np.zeros((memory_pose.shape[0], J_max - J, 3), dtype=np.float32)
            memory_pose = np.concatenate([memory_pose, pad], axis=1)
        memory_pose_list.append(memory_pose[:, :J_max])

        # -------------------------
        # memory rot6d
        # -------------------------
        memory_rot6d = b["memory_rot6d"]  # [N, J, 6]
        if J < J_max:
            pad = np.zeros((memory_rot6d.shape[0], J_max - J, 6), dtype=np.float32)
            memory_rot6d = np.concatenate([memory_rot6d, pad], axis=1)
        memory_rot6d_list.append(memory_rot6d[:, :J_max])

        # -------------------------
        # ref position
        # -------------------------
        ref_pos = b["ref_position"]
        if J < J_max:
            pad = np.zeros((J_max - J, 3), dtype=np.float32)
            ref_pos = np.concatenate([ref_pos, pad], axis=0)
        ref_pos_list.append(ref_pos[:J_max])

        # -------------------------
        # ref rot a
        # -------------------------
        ref_rot_a = b["ref_rot6d_a"]
        if J < J_max:
            pad = np.zeros((J_max - J, 6), dtype=np.float32)
            ref_rot_a = np.concatenate([ref_rot_a, pad], axis=0)
        ref_rot_a_list.append(ref_rot_a[:J_max])

        # -------------------------
        # ref image_embed
        # -------------------------
        ref_img_list.append(b["ref_image_embed"])

        # -------------------------
        # parent / offset a
        # -------------------------
        parent_a_list.append(b["parent_a"].squeeze(0)[:J_max])
        offset_a_list.append(b["offset_a"].squeeze(0)[:J_max])

    return {
        # ===== shared =====
        "position": torch.from_numpy(np.stack(pos_list, 0)),                 # [B, W, J, 3]
        "ref_position": torch.from_numpy(np.stack(ref_pos_list, 0)),
        "joint_mask": torch.from_numpy(np.stack(joint_mask_list, 0)),
        "ancestor_mask": torch.from_numpy(np.stack(ancestor_mask_list, 0)),
        "J_valid": torch.tensor(J_valid_list, dtype=torch.int32),
        "global_scale": torch.from_numpy(np.array(scale_list)).float(),
        "species": species_list,
        "graph_hop": torch.from_numpy(np.stack(hop_list, 0)),
        "graph_edge": torch.from_numpy(np.stack(edge_list, 0)),
        "joint_t5embed": torch.from_numpy(np.stack(joint_t5_list, 0)),
        "static_rot_joint_mask": torch.from_numpy(np.stack(static_rot_joint_mask_list, 0)),
        "static_pos_joint_mask": torch.from_numpy(np.stack(static_pos_joint_mask_list, 0)),

        # ===== memory =====
        "memory_pose": torch.from_numpy(np.stack(memory_pose_list, 0)),
        "memory_rot6d": torch.from_numpy(np.stack(memory_rot6d_list, 0)),

        # ===== view a =====
        "rot6d_a": torch.from_numpy(np.stack(rot_a_list, 0)),
        "ref_rot6d_a": torch.from_numpy(np.stack(ref_rot_a_list, 0)),
        "parent_a": torch.from_numpy(np.stack(parent_a_list, 0)),
        "offset_a": torch.from_numpy(np.stack(offset_a_list, 0)),

        # ===== video2pose =====
        "image_embed": torch.from_numpy(np.stack(img_list, 0)),
        "ref_image_embed": torch.from_numpy(np.stack(ref_img_list, 0)),
    }
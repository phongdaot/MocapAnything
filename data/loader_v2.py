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
from utils.bvh_reader import BVHReader

MAX_JOINTS=150

import zipfile
from numpy.lib import format as _np_format

def _npz_shape(npz_path: str, key: str):
    """读取 .npz 中某数组的 shape,只解析 .npy header,不加载数据(快)。失败返回 None。"""
    try:
        with zipfile.ZipFile(npz_path) as z:
            with z.open(key + ".npy") as f:
                version = _np_format.read_magic(f)
                shape, _fortran, _dtype = _np_format._read_array_header(f, version)
                return tuple(int(x) for x in shape)
    except Exception:
        return None

def _npz_leading_dim(npz_path: str, key: str) -> Optional[int]:
    """某数组首维长度(header-only)。"""
    s = _npz_shape(npz_path, key)
    return s[0] if s else None

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
        blocklist: Optional[str] = None,          # 每行一个 mid,排除坏几何/坏渲染(mobjaverse 用)
        pose_jumps_path: Optional[str] = None,    # {seq:[接缝帧]} 切分,窗口不跨接缝(mobjaverse 用)
        raw_position: bool = False,               # mobjaverse: 用 position_before(原始,与 rest_pose 同尺度)
        epoch_sample_ratio: Optional[float] = None,  # 每 epoch 只随机采样这一比例的序列(mobjaverse 用 0.25 均衡);None=全用
        ref_enhance: Optional[str] = None,        # None | "cross_seq"(同物种跨序列跨角度,zoo用) | "cross_angle"(同序列跨角度,obj/mob用)
    ):
        self.ref_enhance = ref_enhance
        # mobjaverse 的 position 在 build 时已 ×scale(0.5×),与 bvh rest_pose(原始)差 ~2× → 骨长不一致;
        # raw_position=True 改用 position_before(原始) + gscale=bbox,使 pos/gscale 与 rest/gscale 同尺度。
        self.pos_field = "position_before" if raw_position else "position"
        self.epoch_sample_ratio = epoch_sample_ratio
        self._active_idx = None      # 子采样时:本 epoch 生效的位置索引(numpy 数组);None=全用
        self._n_keep = None          # 子采样保留条数(固定,保证 DistributedSampler 长度稳定)
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
        # fps memory bank 已弃用:memory_pkl_path 可选;不提供则所有 species
        # has_memory=False(走现成 fallback),与 multidata 主线一致。
        self.memory_pkl_path = memory_pkl_path
        self.species_memory_dict = {}
        if self.memory_pkl_path:
            with open(self.memory_pkl_path, "rb") as f:
                self.species_memory_dict = pickle.load(f)
            if not isinstance(self.species_memory_dict, dict):
                raise ValueError(f"memory_pkl_path is not a dict: {self.memory_pkl_path}")
            logger.info(
                f"loaded species memory pkl: {self.memory_pkl_path}, "
                f"species num = {len(self.species_memory_dict)}"
            )
        else:
            logger.info("memory_pkl_path not set — fps memory bank disabled (deprecated)")

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

            # 遍历 split_dict 里的所有组(不再硬编码 seen/rare/unseen),
            # 以支持官方形态分组(biped_seen / quadruped_unseen / misc_seen ...);
            # test_seq_set 收集全部组 → 保证所有测试对象都排除出 train(防泄漏)。
            for group in split_dict:
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
        # blocklist(排除坏 mid)+ pose_jumps 切分 manifest(mobjaverse 专用;zoo/obj 不传即空)
        # =========================================================
        self.blocklist_ids = set()
        if blocklist and os.path.isfile(blocklist):
            self.blocklist_ids = set(open(blocklist).read().split())
            logger.info(f"blocklist: {len(self.blocklist_ids)} ids will be excluded")
        self.pose_jumps = {}
        if pose_jumps_path and os.path.isfile(pose_jumps_path):
            with open(pose_jumps_path) as f:
                self.pose_jumps = json.load(f)
            logger.info(f"pose_jumps: {len(self.pose_jumps)} seqs have seams; windows won't cross seams")

        # =========================================================
        # scale cache
        # =========================================================
        # scale cache 优先找 {root}/cache/,回退 {root}/(兼容不同数据集的布局,如 obj1k)
        _scale_in_cache = os.path.join(root, "cache/__mesh2pose1002_species_scale_cache.pkl")
        _scale_in_root = os.path.join(root, "__mesh2pose1002_species_scale_cache.pkl")
        self.scale_dict_path = _scale_in_cache if os.path.isfile(_scale_in_cache) else _scale_in_root
        self.scale_dict = self._load_scale_cache() if cache_scale else {}

        # =========================================================
        # meta cache
        # =========================================================
        # 缓存 key 纳入 blocklist + pose_jumps + pos_field 内容哈希:任一变化则缓存失效重建
        import hashlib as _hashlib
        # 纳入 test_seq_set(即 split 内容):换 split(如拓扑→官方形态)后旧 train 缓存会失效重建,
        # 否则会复用旧缓存把官方 val 对象当训练用。仅影响 hash 分支(mobj 有 blocklist),zoo/obj 仍 base。
        _angsig = sorted((k, tuple(sorted(v))) for k, v in self.test_angle_map.items())  # 视角选择变了也失效重建
        _tag_src = json.dumps([sorted(self.blocklist_ids), sorted(self.pose_jumps.keys()), self.pos_field, sorted(self.test_seq_set), _angsig])
        _tag = _hashlib.md5(_tag_src.encode()).hexdigest()[:8] if (self.blocklist_ids or self.pose_jumps) else "base"
        self.items_cache_path = os.path.join(
            root,
            f"cache/__mesh2pose1002_{self.split_mode}_{self.split_group}_seqlen_{self.window}_{_tag}_items_cache.json"
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
                os.makedirs(os.path.dirname(self.items_cache_path), exist_ok=True)
                # 原子写:先写临时文件再 os.replace,避免其他 rank 读到半成品缓存(JSONDecodeError)
                _tmp = f"{self.items_cache_path}.tmp.{os.getpid()}"
                with open(_tmp, "w") as f:
                    json.dump(self.items, f)
                os.replace(_tmp, self.items_cache_path)
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

        # 过滤掉 static info 缺失(species_info_dict / scale_dict 未覆盖)的序列,
        # 否则 __getitem__ 会 KeyError。这样训练集只含有完整 static info 的物种。
        _before = len(self.items)
        self.items = [
            it for it in self.items
            if it["rel"].split("/")[0].split("#")[0] in self.species_static_cache
        ]
        _dropped = _before - len(self.items)
        if _dropped > 0:
            logger.warning(
                f"dropped {_dropped}/{_before} sequences whose species lacks static info "
                f"(missing in species_info_dict/scale cache); {len(self.items)} remain"
            )

        # =========================================================
        # preload all sequences into RAM
        # =========================================================
        self.data_cache = None
        if self.preload_all:
            self.data_cache = self._preload_all_data()
            logger.info(f"preload finished: {len(self.data_cache)} sequences in RAM")
        else:
            logger.info("preload_all=False, will still read on demand")

        # =========================================================
        # 每-epoch 子采样(mobjaverse 均衡):固定保留 N*ratio 条,__len__ 恒定(DistributedSampler 稳定),
        # set_epoch 只重洗“哪 N*ratio 条”生效。所有 rank 用同一 epoch 种子 → 子集一致。
        # =========================================================
        self._full_n = len(self.data_cache) if (self.preload_all and self.data_cache is not None) else len(self.items)
        if self.epoch_sample_ratio is not None and 0.0 < self.epoch_sample_ratio < 1.0 and self._full_n > 0:
            self._n_keep = max(1, int(round(self._full_n * self.epoch_sample_ratio)))
            self.set_epoch(0)
            logger.info(f"epoch_sample_ratio={self.epoch_sample_ratio}: 每 epoch 用 {self._n_keep}/{self._full_n} 条")

        # ref-enhance 索引(对齐发布配方):按物种 / 按 anim(species#anim)归组各自的 (pose,train) 路径,
        # 供训练时从更广的池采样参考帧。cross_seq=同物种任意序列/角度;cross_angle=同序列任意角度。
        self._sp_rels = {}
        self._base_rels = {}
        if self.ref_enhance and self.split_mode == "train" and self.items:
            from collections import defaultdict as _dd
            _spd, _bad = _dd(list), _dd(list)
            for _it in self.items:
                _rel = _it["rel"]
                _entry = (_it["pose"], _it["train"])
                _spd[_rel.split("/")[0].split("#")[0]].append(_entry)
                _bad[_rel.rsplit("/", 1)[0]].append(_entry)
            self._sp_rels, self._base_rels = dict(_spd), dict(_bad)
            logger.info(f"[ref_enhance={self.ref_enhance}] 索引: {len(self._sp_rels)} 物种 / {len(self._base_rels)} anim")

    def set_epoch(self, epoch: int):
        """子采样模式下,每 epoch 重新随机抽取 _n_keep 条生效序列(所有 rank 一致)。非子采样则 no-op。"""
        if self._n_keep is None:
            return
        rng = np.random.RandomState(2024 + int(epoch))
        self._active_idx = rng.permutation(self._full_n)[:self._n_keep]

    def _map_idx(self, idx: int) -> int:
        """把外部索引(0.._n_keep-1)映射到真实位置;非子采样则原样返回。"""
        if self._active_idx is not None:
            return int(self._active_idx[idx])
        return idx

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

            # blocklist 过滤(坏几何/坏渲染,按 mid = # 号前)
            if self.blocklist_ids and seq_name.split('#')[0] in self.blocklist_ids:
                continue

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

            pose_shape = _npz_shape(pose_p, self.pos_field)   # [F, J, 3] header-only
            if pose_shape is None:
                continue
            F_pose, J = pose_shape[0], pose_shape[1]

            # image_embed 帧数可能 < pose(video-only 抽特征长度差异);加载时 F=min(pose,img),
            # 这里也按 min 过滤;只读 npy header 取帧数,不加载整块(否则建库巨慢)。
            F_img = _npz_leading_dim(train_p, "image_embed")
            if F_img is None:
                continue
            F = min(int(F_pose), int(F_img))

            # pose_jumps 切分:窗口不跨接缝。无接缝(zoo/obj)时 pieces=[[0,F]],等价原逻辑。
            pieces = [p for p in self._pieces_for(seq_name, int(F)) if p[1] - p[0] >= self.window]
            if not pieces:
                skipped.append((stem, F))
                continue

            items.append({
                "bvh": bvh_fpath,
                "rel": stem,
                "pose": pose_p,
                "train": train_p,
                "F": int(F),
                "J": int(J),
                "pieces": pieces,
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

            position = pose_npz[self.pos_field].astype(np.float32)   # [F, J, 3]
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

        position = pose_npz[self.pos_field].astype(np.float32)
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
    def _pieces_for(self, seq_name: str, F: int) -> List[List[int]]:
        """按 pose_jumps 接缝把 [0,F) 切成连续片段;无接缝则整段。越界接缝忽略。"""
        seams = self.pose_jumps.get(seq_name, [])
        if not seams:
            return [[0, F]]
        bnds = [0] + sorted({int(s) for s in seams if 0 < int(s) < F}) + [F]
        return [[bnds[i], bnds[i + 1]] for i in range(len(bnds) - 1)]

    def _rand_window(self, pieces: List[List[int]]) -> Tuple[int, int, int, int]:
        """在 ≥window 的片段中按长度加权随机选一个,窗口只落在该片段内(不跨接缝)。
        返回 (t0, t1, piece_start, piece_end)。"""
        valid = [(s, e) for s, e in pieces if e - s >= self.window]
        if not valid:
            s, e = pieces[0]
            return s, min(s + self.window, e), s, e
        lengths = [e - s for s, e in valid]
        s, e = random.choices(valid, weights=lengths, k=1)[0]
        t0 = random.randint(s, e - self.window)
        return t0, t0 + self.window, s, e

    def __len__(self):
        if self._active_idx is not None:      # 子采样:长度=保留条数(恒定)
            return len(self._active_idx)
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
        # 子采样:外部索引先映射到真实位置(只在入口映射一次;重采样时直接用真实位置,不再二次映射)
        real = self._map_idx(idx)
        # 坏样本(损坏/截断 npz、缺 key 等)自动跳过并随机重采样,避免整轮训练崩溃
        last_err = None
        for _ in range(16):
            try:
                return self._getitem_impl(real)
            except Exception as e:
                last_err = e
                real = random.randint(0, self._full_n - 1)
        raise last_err

    def _getitem_impl(self, idx: int) -> Dict[str, Any]:
        if self.preload_all and self.data_cache is not None:
            data = self.data_cache[idx]
        else:
            data = self._load_single_item_from_disk(idx)

        F = data["F"]
        J = data["J"]

        if self.split_mode == "train":
            # 片段感知:窗口 + ref 帧都落在同一连续片段内(不跨接缝)。旧缓存无 pieces 时兜底整段。
            pieces = self.items[idx].get("pieces") or [[0, F]]
            t0, t1, ps, pe = self._rand_window(pieces)
            ref_idx = random.randint(ps, pe - 1)
        else:
            t0 = 0
            t1 = min(self.window, F)
            ref_idx = min(int(os.environ.get("EVAL_REF_IDX", "0")), F - 1)
        W = t1 - t0

        species_name = data["species"]
        sp_cache = self.species_static_cache[species_name]

        pos_win = data["position"][t0:t1]
        ref_pos = data["position"][ref_idx]

        rot_win_a = data["rot6d_a"][t0:t1]
        ref_rot_a = data["rot6d_a"][ref_idx]

        img_win = data["image_embed"][t0:t1]
        ref_img = data["image_embed"][ref_idx]

        # ref-enhance(对齐发布配方):训练时把参考帧从更广的池重采(zoo=同物种跨序列跨角度;obj=同序列跨角度)。
        # 从池里随机挑一条 (pose,train) npz,再随机取一帧,按本 dataset 同样的归一化(减该帧root/÷scale)。
        # ref 必须是统一的一帧:pose/rot/图片都取同一 _fi(在 pose 与 image 的公共帧范围内),保证严格对齐。
        # J 不一致或加载失败则回退到默认同序列 ref。
        if self.split_mode == "train" and self.ref_enhance:
            _pool = (self._sp_rels.get(species_name) if self.ref_enhance == "cross_seq"
                     else self._base_rels.get(data["rel"].rsplit("/", 1)[0]))
            if _pool:
                _rp_path, _rt_path = random.choice(_pool)
                try:
                    _pn = np.load(_rp_path, mmap_mode='r' if self.mmap else None, allow_pickle=False)
                    _tn = np.load(_rt_path, mmap_mode='r' if self.mmap else None, allow_pickle=False)
                    _rposA = _pn[self.pos_field]
                    _ie = _tn["image_embed"]
                    _F = min(_rposA.shape[0], _ie.shape[0])
                    if _F > 0:
                        _fi = random.randint(0, _F - 1)
                        _rp1 = _rposA[_fi].astype(np.float32)
                        if _rp1.shape[0] == J:                        # 同物种应同 J
                            _gsc = float(sp_cache["global_scale"])
                            ref_pos = (_rp1 - _rp1[0:1, :]) / _gsc     # [J,3] 与 data["position"] 同归一化
                            ref_rot_a = _pn["rot6d"][_fi].astype(np.float32)
                            ref_img = _ie[_fi].astype(np.float32)      # 同一帧的图片,严格对齐
                    if hasattr(_tn, "close"): _tn.close()
                    if hasattr(_pn, "close"): _pn.close()
                except Exception:
                    pass    # 保持默认同序列 ref

        # 防御:部分 npz 帧数不一致(mobjaverse 抽取未完/损坏 partial npz)→ 窗口长度不齐,
        # collate 拼接会崩。校验不齐就 raise,由 __getitem__ 的重试包裹重采样掉。
        if pos_win.shape[0] != W or rot_win_a.shape[0] != W or img_win.shape[0] != W:
            raise ValueError(
                f"window frame mismatch pos={pos_win.shape[0]} rot={rot_win_a.shape[0]} "
                f"img={img_win.shape[0]} W={W}"
            )

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

    # 诊断:任何 batch 内样本的时间/关节维不一致都会在下面 concatenate 崩;先明确报出来。
    for _bi, _b in enumerate(batch):
        _Jb = _b["J"]
        for _k, _tdim, _jdim in (("position", 0, 1), ("rot6d_a", 0, 1), ("image_embed", 0, None)):
            _arr = _b[_k]
            if _arr.shape[_tdim] != W or (_jdim is not None and _arr.shape[_jdim] != _Jb):
                logger.error(
                    f"[collate BAD] sample{_bi} field={_k} shape={_arr.shape} "
                    f"expect W={W} J={_Jb} species={_b.get('species')} rel-cache 有坏样本"
                )

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
"""Build `species_info_dict.npy` from a preprocessed dataset root.

Consumes the stage-3 / stage-4 artifacts (`bvh/` and `bvh_pose/`) and produces the
per-species static info that training and inference load:

    joints_name       list[str], length J          BVH hierarchy order
    joints_distance   (J, J) float64               tree-hop distance, capped at max_path_len
    joint_relation    (J, J) float64               edge-type id (see EDGE_TYPES)
    rename_clean      list[str], length J          human-readable joint name fed to T5
    t5_embedding      (J, 768) float32             mean-pooled t5-base encoding of rename_clean
    static_joints     list[int]                    joints that never translate (root-relative)
    static_rot_joints list[int]                    joints that never rotate

`rename_clean` is either looked up in a per-species joint-name map (`--joint_name_map`)
or derived automatically from the raw BVH joint names. The zoo1030 release uses curated
canonical names ("quadruped spine 01") that cannot be derived from the raw names, so it
needs the map; obj1k was built with the automatic path and reproduces exactly without one.

Usage:
    # build
    python preprocess/build_species_info.py --dataset_root <root> [--joint_name_map map.json]

    # verify a rebuild against an existing file without writing anything
    python preprocess/build_species_info.py --dataset_root <root> \
        --joint_name_map map.json --verify_against <root>/species_info_dict.npy --sample 10

    # export the orig -> canonical map out of an existing species_info_dict.npy
    python preprocess/build_species_info.py --export_map_from <root>/species_info_dict.npy \
        --output joint_name_map.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import bvh as BVH


EDGE_TYPES = {
    'self':          0,
    'parent':        1,
    'child':         2,
    'sibling':       3,
    'no_relation':   4,
    'end_effector':  5,
    'ts_token_conn': 6,
}

# A joint is static when its speed stays below this for every frame of every sequence.
STATIC_EPS = 1e-6

# `static_joints` is measured on the four cardinal yaws only; `static_rot_joints` on all of them.
POS_YAWS = {'y0.npz', 'y90.npz', 'y180.npz', 'y270.npz'}


# --------------------------------------------------------------------------- topology

def create_topology_edge_relations(parents, max_path_len=5):
    """Build (edge_rel, topo_rel) for a skeleton given its `parents` array.

    parents : array-like of int, length J. parents[0] is typically -1.
    """
    n = len(parents)
    topo_rel = np.zeros((n, n), dtype=np.float64)
    edge_rel = np.full((n, n), EDGE_TYPES['no_relation'], dtype=np.float64)

    for i in range(n):
        parent_i = parents[i]
        is_ee = True
        for j in range(n):
            parent_j = parents[j]

            # --- edge type ---
            if i == j:
                edge_rel[i, j] = EDGE_TYPES['self']
            elif parent_j == i:                # j is a child of i
                edge_rel[i, j] = EDGE_TYPES['child']
                is_ee = False
            elif j == parent_i:                # j is the parent of i
                edge_rel[i, j] = EDGE_TYPES['parent']
            elif parent_j == parent_i:         # same parent => sibling
                edge_rel[i, j] = EDGE_TYPES['sibling']
            # else: stays no_relation

            # --- tree-hop distance ---
            if i == j:
                topo_rel[i, j] = 0
            elif j < i:                        # symmetric, reuse
                topo_rel[i, j] = topo_rel[j, i]
            elif parent_j == i:
                topo_rel[i, j] = 1
            else:
                topo_rel[i, j] = topo_rel[i, parent_j] + 1

        if is_ee:
            edge_rel[i, i] = EDGE_TYPES['end_effector']

    topo_rel[topo_rel > max_path_len] = max_path_len
    return edge_rel, topo_rel


def species_from_path(path):
    """`bvh/Alligator#AlligatorALL-Bite1/y0.bvh` -> `Alligator`."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if '#' in stem:
        return stem.split('#', 1)[0]
    parent = os.path.basename(os.path.dirname(path))
    if '#' in parent:
        return parent.split('#', 1)[0]
    return stem


def collect_skeletons(bvh_root, max_path_len=5, verbose=True):
    """One skeleton per species. When a species has several rig variants across its
    sequences, pick the MAJORITY skeleton rather than whichever file sorts first."""
    # One yaw per sequence is enough: rotating a BVH never changes its hierarchy.
    paths = sorted(glob(os.path.join(bvh_root, '*#*', 'y0.bvh')))
    if not paths:
        paths = sorted(glob(os.path.join(bvh_root, '**', '*.bvh'), recursive=True))
    if not paths:
        raise SystemExit(f'No .bvh files found under {bvh_root}')

    by_species = defaultdict(list)
    for p in paths:
        by_species[species_from_path(p)].append(p)

    out = OrderedDict()
    iterator = sorted(by_species)
    if verbose:
        iterator = tqdm(iterator, desc='skeletons')

    for species in iterator:
        sig_count = Counter()
        sig_data = {}
        for bvh_pth in by_species[species]:
            try:
                anim, names, _ = BVH.load(bvh_pth)
            except Exception:
                continue
            sig = tuple(names)
            sig_count[sig] += 1
            if sig not in sig_data:
                sig_data[sig] = (list(names), np.asarray(anim.parents))

        if not sig_count:
            print(f'[skip] {species}: no readable BVH')
            continue

        best_sig, best_n = sig_count.most_common(1)[0]
        if len(sig_count) > 1:
            print(f'[multi-rig] {species}: {len(sig_count)} skeletons '
                  f'{[(len(s), c) for s, c in sig_count.most_common()]}, '
                  f'using J={len(best_sig)} (n={best_n})')

        names, parents = sig_data[best_sig]
        # BVH `ROOT __0` is a placeholder name; the released files call it Root.
        names = ['Root' if n == '__0' else n for n in names]
        edge_rel, topo_rel = create_topology_edge_relations(parents, max_path_len=max_path_len)
        out[species] = {
            'joints_name':     names,
            'joints_distance': topo_rel,
            'joint_relation':  edge_rel,
        }

    return out


# ------------------------------------------------------------------- joint name cleanup

REMOVE_PREFIXES = ['BN_Bip01', 'Bip01', 'BN', 'NPC', 'jt', 'Sabrecat', 'Elk',
                   'mixamorig:', 'mixamorig']
JAPANESE_WORDS = {
    'momo': 'Thigh', 'sippo': 'Tail', 'mune': 'Chest', 'hiza': 'Knee', 'hara': 'Stomach',
    'ashi': 'Leg', 'hiji': 'Elbow', 'koshi': 'Hips', 'te': 'Hand', 'kubi': 'Neck',
    'atama': 'Head', 'ago': 'Jaw', 'kata': 'Shoulder',
}


def _remove_prefix(s):
    orig = s
    for prefix in REMOVE_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s or orig


def _split_and_replace(s):
    new_splitted = []
    for part in re.split('(?=[A-Z]|_)', s):
        clean = re.sub(r'[\d_]+', '', part)
        if clean == '':
            continue
        elif clean in ('L', 'l'):
            new_splitted.append('Left')
        elif clean in ('R', 'r'):
            new_splitted.append('Right')
        elif clean in JAPANESE_WORDS:
            new_splitted.append(JAPANESE_WORDS[clean])
        elif clean == 'Tai':
            new_splitted.append('Tail')
        else:
            new_splitted.append(clean)
    return ' '.join(new_splitted) if new_splitted else s


def auto_clean(name):
    """Strip rig prefixes and split CamelCase/underscores into words T5 can read."""
    return _split_and_replace(_remove_prefix(name)) if name else ''


def canonical_names(species, joints_name, name_map):
    """Look joint names up in `name_map[species]`, falling back to `auto_clean`."""
    if not name_map:
        return [auto_clean(n) for n in joints_name], 0
    per_species = name_map.get(species, {})
    out, missing = [], 0
    for n in joints_name:
        if n in per_species:
            out.append(per_species[n])
        else:
            missing += 1
            out.append(auto_clean(n))
    return out, missing


# ------------------------------------------------------------------------------- T5

def make_t5_encoder(model_name='t5-base', device=None):
    """Mean-pool t5-base over the joint-name tokens, masking padding and empty names."""
    import torch
    from transformers import T5EncoderModel, T5Tokenizer

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading T5 encoder: {model_name} on {device}')
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    encoder = T5EncoderModel.from_pretrained(model_name).to(device).eval()

    def embed(entries):
        inputs = tokenizer(entries, return_tensors='pt', padding=True, truncation=True).to(device)
        mask = inputs['attention_mask'].clone()
        for i, e in enumerate(entries):
            if e == '':
                mask[i] = 0
        with torch.no_grad():
            out = encoder(**inputs).last_hidden_state
            pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return pooled.float().cpu().numpy()

    return embed


# ---------------------------------------------------------------------------- static

def static_indices(files, key, subtract_traj):
    """Indices of joints whose per-frame speed never exceeds STATIC_EPS."""
    cat, num_joints = [], None
    for f in files:
        try:
            data = np.load(f)
        except Exception:
            continue
        if key not in data:
            continue
        a = data[key]
        if a.ndim != 3 or a.shape[0] == 0:
            continue
        if subtract_traj and 'traj' in data:
            a = a - data['traj'][:, None, :]        # root-relative
        if num_joints is None:
            num_joints = a.shape[1]
        elif a.shape[1] != num_joints:
            continue
        cat.append(a)

    if not cat:
        return []
    cat = np.concatenate(cat, 0)
    if cat.shape[0] < 2:
        return list(range(cat.shape[1]))
    speed = np.linalg.norm(np.diff(cat, axis=0), axis=-1)     # [T-1, J]
    return np.where((speed < STATIC_EPS).all(axis=0))[0].tolist()


def species_static(pose_root, species):
    all_npz = sorted(glob(os.path.join(pose_root, f'{species}#*', 'y*.npz')))
    cardinal = [f for f in all_npz if os.path.basename(f) in POS_YAWS]
    return (static_indices(cardinal, 'position', subtract_traj=True),
            static_indices(all_npz, 'rot6d', subtract_traj=False))


# ----------------------------------------------------------------------------- build

def build(dataset_root, name_map=None, t5_model='t5-base', max_path_len=5,
          device=None, only_species=None):
    bvh_root = os.path.join(dataset_root, 'bvh')
    pose_root = os.path.join(dataset_root, 'bvh_pose')

    skeletons = collect_skeletons(bvh_root, max_path_len=max_path_len)
    if only_species is not None:
        skeletons = OrderedDict((s, v) for s, v in skeletons.items() if s in only_species)
    print(f'{len(skeletons)} species')

    embed = make_t5_encoder(t5_model, device=device)

    total_missing = 0
    out = OrderedDict()
    for species, info in tqdm(skeletons.items(), total=len(skeletons), desc='species_info'):
        rename_clean, missing = canonical_names(species, info['joints_name'], name_map)
        total_missing += missing
        static_pos, static_rot = species_static(pose_root, species)
        out[species] = {
            'joints_name':       info['joints_name'],
            'joints_distance':   info['joints_distance'],
            'joint_relation':    info['joint_relation'],
            'rename_clean':      rename_clean,
            't5_embedding':      embed(rename_clean).astype(np.float32),
            'static_joints':     static_pos,
            'static_rot_joints': static_rot,
        }

    if name_map and total_missing:
        print(f'[warn] {total_missing} joints were absent from the name map and fell back '
              f'to automatic cleanup')
    return out


# ---------------------------------------------------------------------------- verify

def cosine_rows(a, b):
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.where(den == 0, 1.0, den)
    return num / den


def verify(built, reference, min_cosine=0.9999, max_abs=1e-2):
    exact_fields = ['joints_name', 'joints_distance', 'joint_relation',
                    'rename_clean', 'static_joints', 'static_rot_joints']
    failures = []
    worst_cos, worst_abs = 1.0, 0.0

    for species, got in sorted(built.items()):
        if species not in reference:
            failures.append((species, 'missing', 'not in reference'))
            continue
        ref = reference[species]
        for field in exact_fields:
            a, b = got[field], ref[field]
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                same = np.array_equal(np.asarray(a), np.asarray(b))
            else:
                same = [str(x) for x in a] == [str(x) for x in b]
            if not same:
                failures.append((species, field, f'{_brief(a)} != {_brief(b)}'))

        a, b = np.asarray(got['t5_embedding'], np.float64), np.asarray(ref['t5_embedding'], np.float64)
        if a.shape != b.shape:
            failures.append((species, 't5_embedding', f'shape {a.shape} != {b.shape}'))
            continue
        cos, dif = cosine_rows(a, b).min(), np.abs(a - b).max()
        worst_cos, worst_abs = min(worst_cos, cos), max(worst_abs, dif)
        if cos < min_cosine or dif > max_abs:
            failures.append((species, 't5_embedding', f'cos={cos:.6f} maxabs={dif:.4g}'))

    print(f'\nchecked {len(built)} species')
    print(f'  t5_embedding worst cosine  : {worst_cos:.8f}  (threshold {min_cosine})')
    print(f'  t5_embedding worst max-abs : {worst_abs:.6g}  (threshold {max_abs})')
    if failures:
        print(f'\nFAIL — {len(failures)} mismatch(es):')
        for species, field, detail in failures[:40]:
            print(f'  {species:32s} {field:18s} {detail}')
        if len(failures) > 40:
            print(f'  ... and {len(failures) - 40} more')
    else:
        print('\nPASS — every field matches the reference')
    return not failures


def _brief(v):
    if isinstance(v, np.ndarray):
        return f'ndarray{v.shape}'
    s = str(list(v)[:4])
    return s + (' ...' if len(v) > 4 else '')


# ------------------------------------------------------------------------------ main

def pick_sample(names, n):
    """Evenly spaced across the sorted list, so the tail gets covered too."""
    names = sorted(names)
    if n <= 0 or n >= len(names):
        return set(names)
    step = len(names) / n
    return {names[int(i * step)] for i in range(n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset_root', help='dataset root holding bvh/ and bvh_pose/')
    ap.add_argument('--joint_name_map',
                    help='per-species {orig: canonical} JSON. '
                         'Defaults to <dataset_root>/joint_name_map.json when present.')
    ap.add_argument('--t5_model', default='t5-base')
    ap.add_argument('--max_path_len', type=int, default=5)
    ap.add_argument('--device', default=None)
    ap.add_argument('--output', help='where to write (default <dataset_root>/species_info_dict.npy)')
    ap.add_argument('--verify_against', help='compare against this species_info_dict.npy; writes nothing')
    ap.add_argument('--sample', type=int, default=0, help='verify only N species (0 = all)')
    ap.add_argument('--export_map_from', help='dump orig->canonical map from this species_info_dict.npy and exit')
    args = ap.parse_args()

    if args.export_map_from:
        src = np.load(args.export_map_from, allow_pickle=True).item()
        name_map = {sp: dict(zip((str(n) for n in info['joints_name']),
                                 (str(c) for c in info['rename_clean'])))
                    for sp, info in src.items()}
        out = args.output or 'joint_name_map.json'
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(name_map, f, ensure_ascii=False, indent=1, sort_keys=True)
        joints = sum(len(v) for v in name_map.values())
        print(f'{len(name_map)} species / {joints} joints -> {out}')
        return

    if not args.dataset_root:
        ap.error('--dataset_root is required unless --export_map_from is given')

    map_path = args.joint_name_map
    if map_path is None:
        default_map = os.path.join(args.dataset_root, 'joint_name_map.json')
        map_path = default_map if os.path.exists(default_map) else None

    name_map = None
    if map_path:
        with open(map_path, encoding='utf-8') as f:
            name_map = json.load(f)
        print(f'joint name map: {map_path} ({len(name_map)} species)')
    else:
        print('[warn] no joint name map given — joint names will be cleaned automatically.\n'
              '       That is correct for obj1k-style data, but it will NOT reproduce the\n'
              '       curated canonical names used by the released zoo1030 species_info.')

    only = None
    reference = None
    if args.verify_against:
        reference = np.load(args.verify_against, allow_pickle=True).item()
        if args.sample:
            only = pick_sample(reference.keys(), args.sample)
            print(f'verifying a sample of {len(only)} / {len(reference)} species')

    built = build(args.dataset_root, name_map=name_map, t5_model=args.t5_model,
                  max_path_len=args.max_path_len, device=args.device, only_species=only)

    if reference is not None:
        sys.exit(0 if verify(built, reference) else 1)

    out = args.output or os.path.join(args.dataset_root, 'species_info_dict.npy')
    np.save(out, dict(built))
    print(f'{len(built)} species -> {out}')


if __name__ == '__main__':
    main()

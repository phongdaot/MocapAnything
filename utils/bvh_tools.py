from pathlib import Path

import numpy as np
import torch

from . import animation as animation
from . import bvh as BVH
from .transforms3d import *


# ---------------------------------------------------------------------------
# Vendored numpy quaternion helpers (Hamilton convention, (w, x, y, z))
# ---------------------------------------------------------------------------
def qmul_np(q0, q1):
    """Hamilton product of two quaternions. Inputs broadcast as numpy arrays;
    the last axis must be 4 in (w, x, y, z) order."""
    w0, x0, y0, z0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    return np.stack([w, x, y, z], axis=-1)


def qrot_np(q, v):
    """Rotate 3D vector v by unit quaternion q. q is (..., 4) (w, x, y, z),
    v is (..., 3); q broadcasts against v on all but the last axis."""
    s = q[..., 0:1]
    r = q[..., 1:]
    cross1 = np.cross(r, v)
    return v + 2.0 * (s * cross1 + np.cross(r, cross1))


def qbetween_np(v0, v1):
    """Quaternion that rotates 3-vector v0 onto v1 (single vectors)."""
    v0 = np.asarray(v0, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v0n = v0 / n0
    v1n = v1 / n1
    dot = float(np.dot(v0n, v1n))
    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-8:
        axis = np.cross(v0n, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(v0n, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    s = float(np.sqrt((1.0 + dot) * 2.0))
    axis = np.cross(v0n, v1n) / s
    return np.array([s / 2.0, axis[0], axis[1], axis[2]])


def face_forward_bvh_with_scale(
    path: str,
    data: animation.Animation,
    names: list,
    frametime: float,
    scale: float = 1.
):
    """
    Rotate the BVH sequence to face the +Z axis, apply scaling, and save to file.

    Args:
        path (str): Output path to save the adjusted BVH file.
        data (Animation): Motion data object containing offsets and joint transforms.
        names (List[str]): Joint names in order.
        frametime (float): Frame time for the output BVH file (1 / FPS).
        scale (float, optional): Scale factor for joint positions. Defaults to 1.0.
    """
    global_xforms = animation.transforms_global(data)
    bvh_offset = data.offsets.copy()
    root_to_pevis = bvh_offset[0]

    local_xforms = animation.get_local_transform_from_global(
        global_xforms, data
    )
    rot_mat = torch.from_numpy(local_xforms[:, :, :3, :3])
    rotations_q = matrix_to_quaternion(rot_mat).numpy()

    global_orientation = rotations_q[:, 0].copy()
    seq_len = global_orientation.shape[0]

    direction = qrot_np(global_orientation[0].copy(), np.array([0, 0, 1]))
    direction_xz = direction.copy()
    direction_xz[1] = 0
    q_rot_inv = qbetween_np(direction_xz, np.array([0, 0, 1]))
    gr_face_forward = qmul_np(
        np.tile(q_rot_inv, (seq_len, 1)), global_orientation
    )
    rotations_q[:, 0] = gr_face_forward
    rotation_mat = quaternion_to_matrix(
        torch.from_numpy(rotations_q)
    )
    rotation_euler = matrix_to_euler_angles(rotation_mat,
                                                       'ZYX').numpy()

    trans = torch.from_numpy(
        global_xforms[:, 0, :3, 3] / global_xforms[:, 0, 3, 3:4]
    )
    trans[:, [0, 2]] -= trans[0, [0, 2]]
    trans = qrot_np(np.tile(q_rot_inv, (seq_len, 1)), trans.numpy())

    positions = np.tile(bvh_offset * scale, (data.shape[0], 1, 1))
    positions[:, 0] += (trans - root_to_pevis) * scale
    bvh_data = {
        'rotations': np.degrees(rotation_euler),
        'positions': positions,
        'offsets': bvh_offset * scale,
        'parents': data.parents,
        'names': names,
        'order': 'zyx',
        'frametime': frametime,
    }
    BVH.save_dict(path, bvh_data, fps=int(1 / frametime))


MOD = 1000000007
EQ_THR = 0.01
ZERO_THR = 1e-8

R = np.random.randint(MOD, size=(1000,))


def L2(d: np.ndarray) -> float:
    return float(np.inner(d, d))


def almost_eq(a: float, b: float) -> bool:
    if abs(a) < ZERO_THR:
        return abs(b) < ZERO_THR
    return abs(a - b) / a < EQ_THR


def compress_array(arr: list[float]) -> list[int]:
    values = []
    for i, v in enumerate(arr):
        values.append((v, i))
    values = sorted(values)
    l = 0
    k = 0
    res = [0] * len(arr)
    while l < len(values):
        r = l + 1
        while r < len(values) and almost_eq(values[l][0], values[r][0]):
            r += 1
        for i in range(l, r):
            res[values[i][1]] = k
        k += 1
        l = r
    return res


def find_symmetry(bvh_path: Path) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Find symmetric joint pairs in a BVH skeleton based on structure and offset similarity.

    Args:
        bvh_path (Path): Path to the input BVH file.

    Returns:
        tuple:
            - center (list[int]): List of center joints for each symmetric subtree.
            - symmetry (list[tuple[int, int]]): List of symmetric joint index pairs.
    """
    anim, names, frametime = BVH.load(str(bvh_path))

    pos_rel = anim.offsets
    par = anim.parents

    pos_rel = compress_array([L2(v) for v in pos_rel])

    n = len(par)
    assert n <= 1000

    dep = [0] * n
    ch = [[] for _ in range(n)]
    for i in range(1, n):
        ch[par[i]].append(i)
        dep[i] = dep[par[i]] + 1

    hs = [1] * n
    hs_w = [1] * n
    for i in reversed(range(n)):
        for j in ch[i]:
            hs[i] *= int(R[dep[i]]) + hs[j]
            hs[i] %= MOD
            hs_w[i] *= int(R[dep[i]]) * hs_w[j] + pos_rel[j]
            hs_w[i] %= MOD

    center = []
    symmetry = []

    def add_symmetry(u, v):
        symmetry.append((u, v))
        assert len(ch[u]) == len(ch[v])
        pairs = []
        used = set()
        for cu in ch[u]:
            cand = []
            for cv in ch[v]:
                if hs_w[cu] == hs_w[cv] and pos_rel[cu] == pos_rel[cv]:
                    cand.append(cv)
            # print(cu, cand)
            assert len(cand) == 1
            assert cand[0] not in used
            used.add(cand[0])
            pairs.append((cu, cand[0]))
        for cu, cv in pairs:
            add_symmetry(cu, cv)

    def subtree_eq(u, v):
        assert u > 0 and v > 0
        return hs_w[u] == hs_w[v] and (
            pos_rel[u] == pos_rel[v] or len(ch[u]) > 0
        )

    def dfs(u):
        center.append(u)
        ls = []
        used = []
        for v in ch[u]:
            for pv in used:
                if subtree_eq(pv, v):
                    raise ValueError("Symmetry of size >= 3 found")
            found = False
            for i, pv in enumerate(ls):
                if subtree_eq(pv, v):
                    # if pos_rel[pv] != pos_rel[v]:
                    # print(f"Suspicious symmetry: {pv}, {v}")
                    found = True
                    add_symmetry(pv, v)
                    used.append(pv)
                    ls.pop(i)
                    break
                # if hs[pv] == hs[v]:
                # print(f"Potentially symmetry: {pv}, {v}")
            if not found:
                ls.append(v)
        for v in ls:
            dfs(v)

    dfs(0)

    return center, symmetry


def get_diameter(parents, bone_length):
    """
    Compute the longest path (tree diameter) in a skeleton tree.

    Args:
        parents (list[int]): Parent index list for each joint (-1 for root).
        bone_length (list[float]): Bone lengths for each joint.

    Returns:
        tuple:
            - float: Total length of the longest path.
            - list[int]: Joint indices along the longest path.
    """
    tree = [
        {
            'bone_length': l,
            'parent': None,
            'children': [],
            'explored': False
        } for l in bone_length
    ]
    for i, p, in enumerate(parents):
        node = tree[i]
        if p != -1:
            node['parent'] = p
            tree[p]['children'].append(i)

    # return tree
    def get_one_end(idx, path=[], length=0.):
        node = tree[idx]
        cur_length = length + node['bone_length']
        path.append(idx)
        node['explored'] = True

        if node['parent']:
            candidate = [
                i for i in node['children'] + [node['parent']]
                if not tree[i]['explored']
            ]
        else:
            candidate = [i for i in node['children'] if not tree[i]['explored']]

        if not candidate:
            return cur_length, path

        m_length = cur_length
        m_path = path[:]
        for c in candidate:
            c_length, c_path = get_one_end(c, path[:], cur_length)
            if c_length >= m_length:
                m_length = c_length
                m_path = c_path

        return m_length, m_path

    a_l, a_p = get_one_end(0)
    for node in tree:
        node['explored'] = False

    b_l, b_p = get_one_end(a_p[-1], [])
    return b_l, b_p


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Batch face-forward + rescale every BVH in a directory."
    )
    parser.add_argument("--src", required=True, help="folder of input .bvh files")
    parser.add_argument("--dst", required=True, help="folder to write rescaled .bvh files")
    parser.add_argument("--scale", type=float, default=100.0)
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for filename in os.listdir(args.src):
        if not filename.lower().endswith(".bvh"):
            continue
        src = os.path.join(args.src, filename)
        dst = os.path.join(args.dst, filename)
        print(f"Scaling {src} -> {dst}")
        anim, joint_names, frametime = BVH.load(src)
        face_forward_bvh_with_scale(dst, anim, joint_names, frametime, args.scale)
    print("Done. All BVH files processed.")

import operator
import random
from typing import Dict, List, Optional, Union

import numpy as np

from .quaternions import Quaternions


class Animation:
    """
    Animation is a numpy-like wrapper for animation data

    Animation data consists of several arrays consisting
    of F frames and J joints.

    The animation is specified by

        rotations : (F, J) Quaternions | Joint Rotations
        positions : (F, J, 3) ndarray  | Joint Positions

    The base pose is specified by

        orients   : (J) Quaternions    | Joint Orientations
        offsets   : (J, 3) ndarray     | Joint Offsets

    And the skeletal structure is specified by

        parents   : (J) ndarray        | Joint Parents
    """

    def __init__(
        self, rotations, positions, orients, offsets, parents, end_site_offsets
    ):

        self.rotations = rotations
        self.positions = positions
        self.orients = orients
        self.offsets = offsets
        self.parents = parents
        self.end_site_offsets = end_site_offsets

    def __op__(self, op, other):
        return Animation(
            op(self.rotations, other.rotations),
            op(self.positions, other.positions),
            op(self.orients, other.orients), op(self.offsets, other.offsets),
            op(self.parents, other.parents),
            op(self.end_site_offsets, other.end_site_offsets)
        )

    def __iop__(self, op, other):
        self.rotations = op(self.roations, other.rotations)
        self.positions = op(self.roations, other.positions)
        self.orients = op(self.orients, other.orients)
        self.offsets = op(self.offsets, other.offsets)
        self.parents = op(self.parents, other.parents)
        self.end_site_offsets = op(
            self.end_site_offsets, other.end_site_offsets
        )
        return self

    def __sop__(self, op):
        return Animation(
            op(self.rotations), op(self.positions), op(self.orients),
            op(self.offsets), op(self.parents), op(self.end_site_offsets)
        )

    def __add__(self, other):
        return self.__op__(operator.add, other)

    def __sub__(self, other):
        return self.__op__(operator.sub, other)

    def __mul__(self, other):
        return self.__op__(operator.mul, other)

    def __div__(self, other):
        return self.__op__(operator.div, other)

    def __abs__(self):
        return self.__sop__(operator.abs)

    def __neg__(self):
        return self.__sop__(operator.neg)

    def __iadd__(self, other):
        return self.__iop__(operator.iadd, other)

    def __isub__(self, other):
        return self.__iop__(operator.isub, other)

    def __imul__(self, other):
        return self.__iop__(operator.imul, other)

    def __idiv__(self, other):
        return self.__iop__(operator.idiv, other)

    def __len__(self):
        return len(self.rotations)

    def __getitem__(self, k):
        if isinstance(k, tuple):
            return Animation(
                self.rotations[k], self.positions[k], self.orients[k[1:]],
                self.offsets[k[1:]], self.parents[k[1:]],
                self.end_site_offsets[k[1:]]
            )
        else:
            return Animation(
                self.rotations[k], self.positions[k], self.orients,
                self.offsets, self.parents, self.end_site_offsets
            )

    def __setitem__(self, k, v):
        if isinstance(k, tuple):
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k[1:], v.orients)
            self.offsets.__setitem__(k[1:], v.offsets)
            self.parents.__setitem__(k[1:], v.parents)
            self.end_site_offsets.__setitem__(k[1:], v.end_site_offsets)
        else:
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k, v.orients)
            self.offsets.__setitem__(k, v.offsets)
            self.parents.__setitem__(k, v.parents)
            self.end_site_offsets.__setitem__(k, v.end_site_offsets)

    def random_rest_pose(self, save_name=None):
        frame_cnt, joint_cnt = self.rotations.shape
        idx = random.randint(1, frame_cnt - 2)

        global_positions = positions_global(self)
        rest_pos = global_positions[idx]
        new_offsets = rest_pos.copy()
        for joint_idx in range(joint_cnt):
            parent = self.parents[joint_idx]
            if parent >= 0:
                new_offsets[joint_idx] = rest_pos[joint_idx] - rest_pos[parent]
            else:
                new_offsets[joint_idx] = rest_pos[joint_idx]

        glob_t = transforms_global(self)[:, :, :3, :3]
        # local_t = transforms_local(self)[:, :, :3, :3]

        # my_glob = local2global(local_t, self.parents)

        glob_t_inv = np.linalg.inv(glob_t)
        glob_x_inv = glob_t_inv[idx]  # J*3*3

        new_rot_local = np.zeros_like(glob_t)
        new_rot_global = np.zeros_like(glob_t)

        for joint_idx in range(joint_cnt):
            parent = self.parents[joint_idx]

            if parent >= 0:
                par_glob_inv = np.linalg.inv(new_rot_global[:, parent])
                old_global = glob_t[:, joint_idx]

                new_rot_local[:, joint_idx
                             ] = par_glob_inv @ old_global @ glob_x_inv[
                                 joint_idx]

                new_rot_global[:, joint_idx
                              ] = new_rot_global[:, parent
                                                ] @ new_rot_local[:, joint_idx]
            else:
                old_global = glob_t[:, joint_idx]  # F*3*3
                new_rot_local[:, joint_idx] = old_global @ glob_x_inv[joint_idx]
                new_rot_global[:, joint_idx] = new_rot_local[:, joint_idx]

        new_rots_quat = Quaternions.from_transforms(new_rot_local)

        self.rotations = new_rots_quat
        self.offsets = new_offsets
        self.positions = self.offsets[None, :, :].repeat(frame_cnt, axis=0)
        self.positions[:, 0] = global_positions[:, 0]

        # import utils.bvh as BVH
        # BVH.save(save_name, self, frametime=1.0 / 30.0)
        # pass

    def apply_rot_on_offset(self, rot_mat):
        if rot_mat.shape[0] == self.offsets.shape[0] and len(
            rot_mat.shape
        ) == 3:
            C = rot_mat
        else:
            C = np.eye(3)[None, ...].repeat(self.offsets.shape[0], axis=0)
            C[:] = rot_mat

        global_xforms = transforms_global(self)
        joints_pos = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
        root_pos = joints_pos[:, 0]
        local_rots = get_local_transform_from_global(global_xforms,
                                                     self)[:, :, :3, :3]
        offs_new, rots_new, root_new = apply_rest_rebase_bvh(
            self.parents,
            self.offsets,
            local_rots,
            C,
            root_pos=root_pos  # root_pos 可为 None
        )

        positions = np.tile(offs_new, (global_xforms.shape[0], 1, 1))
        positions[:, 0] = root_new

        self.rotations = Quaternions.from_transforms(rots_new)
        self.positions = positions
        self.offsets = offs_new

        # rotation_euler = transforms.matrix_to_euler_angles(torch.from_numpy(rots_new),
        #                                                 'ZYX').numpy()
        # bvh_data = {
        #     'rotations': np.degrees(rotation_euler),
        #     'positions': positions,
        #     'offsets': offs_new,
        #     'parents': self.parents,
        #     'names': names,
        #     'order': 'zyx',
        #     'frametime': ft,
        # }

        # if not save_pth:
        #     BVH.save_dict(bvh_pth.replace('.bvh', '_aug.bvh'), bvh_data, fps=int(1 / ft))
        # else:
        #     BVH.save_dict(save_pth, bvh_data, fps=int(1 / ft))

    def apply_rot_on_animation(self, rot_mat):
        if rot_mat.shape[0] == self.offsets.shape[0] and len(
            rot_mat.shape
        ) == 3:
            C = rot_mat
        else:
            C = np.eye(3)[None, ...].repeat(self.offsets.shape[0], axis=0)
            C[:] = rot_mat

        global_xforms = transforms_global(self)
        joints_pos = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
        root_pos = joints_pos[:, 0]
        local_rots = get_local_transform_from_global(global_xforms,
                                                     self)[:, :, :3, :3]
        offs_new, rots_new, root_new = apply_rest_fix(
            self.parents,
            self.offsets,
            local_rots,
            C,
            root_pos=root_pos  # root_pos 可为 None
        )

        positions = np.tile(offs_new, (global_xforms.shape[0], 1, 1))
        positions[:, 0] = root_new

        self.rotations = Quaternions.from_transforms(rots_new)
        self.positions = positions
        self.offsets = offs_new

        # rotation_euler = transforms.matrix_to_euler_angles(torch.from_numpy(rots_new),
        #                                                 'ZYX').numpy()

        # bvh_data = {
        #     'rotations': np.degrees(rotation_euler),
        #     'positions': positions,
        #     'offsets': offs_new,
        #     'parents': self.parents,
        #     'names': names,
        #     'order': 'zyx',
        #     'frametime': ft,
        # }
        # if not save_pth:
        #     BVH.save_dict(bvh_pth.replace('.bvh', '_rot.bvh'), bvh_data, fps=int(1 / ft))
        # else:
        #     BVH.save_dict(save_pth, bvh_data, fps=int(1 / ft))

    def joint_rest_global_pos(self):
        global_pos = self.offsets.copy()
        for idx in range(1, len(self.parents)):
            parent = self.parents[idx]
            global_pos[idx] = global_pos[parent] + self.offsets[idx]
        return global_pos

    @property
    def shape(self):
        return (self.rotations.shape[0], self.rotations.shape[1])

    def copy(self):
        return Animation(
            self.rotations.copy(), self.positions.copy(), self.orients.copy(),
            self.offsets.copy(), self.parents.copy(),
            self.end_site_offsets.copy()
        )

    def repeat(self, *args, **kw):
        return Animation(
            self.rotations.repeat(*args, **kw),
            self.positions.repeat(*args, **kw), self.orients, self.offsets,
            self.parents, self.end_site_offsets
        )

    def ravel(self):
        return np.hstack(
            [
                self.rotations.log().ravel(),
                self.positions.ravel(),
                self.orients.log().ravel(),
                self.offsets.ravel(),
                self.end_site_offsets.ravel()
            ]
        )

    @classmethod
    def unravel(cls, anim, shape, parents):
        nf, nj = shape
        rotations = anim[nf * nj * 0:nf * nj * 3]
        positions = anim[nf * nj * 3:nf * nj * 6]
        orients = anim[nf * nj * 6 + nj * 0:nf * nj * 6 + nj * 3]
        offsets = anim[nf * nj * 6 + nj * 3:nf * nj * 6 + nj * 6]
        end_site_offsets = anim[nf * nj * 6 + nj * 6 + nj * 0:nf * nj * 6 +
                                nj * 6 + nj * 3]
        return cls(
            Quaternions.exp(rotations), positions, Quaternions.exp(orients),
            offsets, parents.copy(), end_site_offsets
        )


def apply_rest_fix(
    parents: np.ndarray,  # (J,), parent index per joint, root parent = -1
    offsets: np.ndarray,  # (J, 3), old rest offsets expressed in parent-local
    local_rots: np.ndarray,  # (T, J, 3, 3), per-frame local rotation matrices
    C: np.ndarray,  # (J, 3, 3), per-joint rest augmentation rotations
    root_pos: Optional[Union[np.ndarray, None]
                      ] = None,  # (T, 3) optional, root translation channels
):
    """
    Apply per-joint rest-fix rotations C_j to a kinematic chain.

    Conventions (column vectors, left-multiply):
      R'_j(t) = C_j @ R_j(t) @ C_j.T
      o'_j    = C_parent(j) @ o_j           (parent(root) := identity)
      root'(t)= C_root @ root(t)            if root translation exists

    Returns:
      offsets_new: (J, 3)
      local_rots_new: (T, J, 3, 3)
      root_pos_new: (T, 3) or None
    """
    T, J = local_rots.shape[0], local_rots.shape[1]
    assert offsets.shape == (J, 3)
    assert C.shape == (J, 3, 3)
    assert local_rots.shape[2:] == (3, 3)
    if root_pos is not None:
        assert root_pos.shape == (T, 3)

    # 1) New offsets: o'_j = C_parent(j) @ o_j
    Cp = np.eye(3, dtype=offsets.dtype)[None, ...].repeat(J, axis=0)  # (J,3,3)
    valid = parents >= 0
    Cp[valid] = C[parents[valid]]  # parent rotation for each joint
    offsets_new = (Cp @ offsets[..., None]).squeeze(-1)  # (J,3)

    # 2) New local rotations: R'_j(t) = C_j @ R_j(t) @ C_j^T
    #    Broadcast over T automatically.
    Ct = np.transpose(C, (0, 2, 1))  # (J,3,3)
    left = C[None, ...]  # (T,J,3,3) after broadcast
    right = Ct[None, ...]
    local_rots_new = left @ local_rots @ right  # (T,J,3,3)

    # 3) Root translation channels (optional)
    root_pos_new = None
    if root_pos is not None:
        root_idx = int(np.where(parents == -1)[0][0]) if (parents
                                                          == -1).any() else 0
        C_root = C[root_idx]  # (3,3)
        root_pos_new = (
            root_pos @ C_root.T
        )  # (T,3)  ← column-vector convention

    return offsets_new, local_rots_new, root_pos_new


def apply_rest_rebase_bvh(
    parents: np.ndarray,  # (J,), parent index per joint, root parent = -1
    offsets: np.ndarray,  # (J, 3), original offsets in parent frame
    local_rots: np.ndarray,  # (T, J, 3, 3), original local rotation matrices
    C: np.ndarray,  # (J, 3, 3), per-joint rest change rotations
    root_pos: Optional[Union[np.ndarray, None]] = None,  # (T, 3) or None
):
    """
    模式A：重定基（世界不变）。假设BVH评估顺序为 G_j = G_p * T(o_j) * R_j。
    返回 (offsets_new, local_rots_new, root_pos_new)
    """
    T, J = local_rots.shape[0], local_rots.shape[1]
    assert offsets.shape == (J, 3)
    assert C.shape == (J, 3, 3)
    assert local_rots.shape[2:] == (3, 3)
    if root_pos is not None:
        assert root_pos.shape == (T, 3)

    # 父节点的C
    Cp = np.repeat(np.eye(3, dtype=offsets.dtype)[None, ...], J, axis=0)
    mask = parents >= 0
    Cp[mask] = C[parents[mask]]

    # 1) offsets：o'_j = C_parent * o_j
    offsets_new = (Cp @ offsets[..., None]).squeeze(-1)

    # 2) rotations：R'_j = C_parent * R_j * C_child^{-1}
    Cinv = np.transpose(C, (0, 2, 1))  # 旋转的逆就是转置
    local_rots_new = (Cp[None, ...] @ local_rots) @ Cinv[None, ...]

    # 3) root translation（可选）
    root_pos_new = None
    if root_pos is not None:
        root_idx = int(np.where(parents == -1)[0][0]) if (parents
                                                          == -1).any() else 0
        C_root = C[root_idx]
        # 列向量约定：v' = C * v  ⇒ 数组右乘 C^T
        root_pos_new = root_pos @ C_root.T

    return offsets_new, local_rots_new, root_pos


def local2global(rots, parents):
    glob_rots = rots.copy()
    for i in range(1, len(parents)):
        glob_rots[:, i] = glob_rots[:, parents[i]] @ rots[:, i]
    return glob_rots


def transforms_local(anim):
    """
    Computes Animation Local Transforms

    As well as a number of other uses this can
    be used to compute global joint transforms,
    which in turn can be used to compete global
    joint positions

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    transforms : (F, J, 4, 4) ndarray

        For each frame F, joint local
        transforms for each joint J
    """

    # quaternion to rotation matrix, get shape of F * J * 3 * 3
    transforms = anim.rotations.transforms()
    # then transform the rotation matrix to homogeneous matrix
    # first expand the shape from F * J * 3 * 3 to F * J * 3 * 4
    transforms = np.concatenate(
        [transforms, np.zeros(transforms.shape[:2] + (3, 1))], axis=-1
    )
    # then expand the shape from F * J * 3 * 4 to F * J * 4 * 4
    transforms = np.concatenate(
        [transforms, np.zeros(transforms.shape[:2] + (1, 4))], axis=-2
    )
    transforms[:, :, 0:3, 3] = anim.positions
    transforms[:, :, 3:4, 3] = 1.0
    return transforms


def transforms_multiply(t0s, t1s):
    """
    Transforms Multiply

    Multiplies two arrays of animation transforms

    Parameters
    ----------

    t0s, t1s : (F, J, 4, 4) ndarray
        Two arrays of transforms
        for each frame F and each
        joint J

    Returns
    -------

    transforms : (F, J, 4, 4) ndarray
        Array of transforms for each
        frame F and joint J multiplied
        together
    """

    # return ut.matrix_multiply(t0s, t1s)
    return t0s @ t1s


def transforms_inv(ts):
    fts = ts.reshape(-1, 4, 4)
    fts = np.array(list(map(lambda x: np.linalg.inv(x), fts)))
    return fts.reshape(ts.shape)


def transforms_blank(anim):
    """
    Blank Transforms

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    transforms : (F, J, 4, 4) ndarray
        Array of identity transforms for
        each frame F and joint J
    """
    # print(anim.shape, anim.rotations.shape, anim.positions.shape)
    ts = np.zeros(anim.shape + (4, 4))
    ts[:, :, 0, 0] = 1.0
    ts[:, :, 1, 1] = 1.0
    ts[:, :, 2, 2] = 1.0
    ts[:, :, 3, 3] = 1.0
    return ts


def transforms_global(anim):
    """
    Global Animation Transforms

    This relies on joint ordering
    being incremental. That means a joint
    J1 must not be a ancestor of J0 if
    J0 appears before J1 in the joint
    ordering.

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    ------

    transforms : (F, J, 4, 4) ndarray
        Array of global transforms for
        each frame F and joint J
    """

    joints = np.arange(anim.shape[1])
    parents = np.arange(anim.shape[1])
    locals = transforms_local(anim)
    globals = transforms_blank(anim)

    globals[:, 0] = locals[:, 0]

    for i in range(1, anim.shape[1]):
        globals[:, i] = transforms_multiply(
            globals[:, anim.parents[i]], locals[:, i]
        )

    return globals


def positions_global(anim):
    """
    Global Joint Positions

    Given an animation compute the global joint
    positions at at every frame

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    positions : (F, J, 3) ndarray
        Positions for every frame F
        and joint position J
    """

    positions = transforms_global(anim)[:, :, :, 3]
    return positions[:, :, :3] / positions[:, :, 3, np.newaxis]


def get_local_transform_from_global(global_xforms, templete_anim):
    """
    Added by hsy on 20210616
    the templete_anim has the same skeleton as anim of global_xforms,
    which means that they share the same orients, offsets and parents.

    Parameters
    ----------

    global_xforms : (F, J, 4, 4) ndarray
        Array of global transforms for
        each frame F and joint J

    templete_anim : Animation
        Input templete animation

    Returns
    -------

    local_xforms : (F, J, 4, 4) ndarray
        Array of local transforms for
        each frame F and joint J
    """
    local_xforms = np.zeros(global_xforms.shape)
    local_xforms[:, :, 0, 0] = 1.0
    local_xforms[:, :, 1, 1] = 1.0
    local_xforms[:, :, 2, 2] = 1.0
    local_xforms[:, :, 3, 3] = 1.0

    # first set the local transforms of root
    local_xforms[:, 0] = global_xforms[:, 0]

    for i in range(1, global_xforms.shape[1]):
        local_xforms[:, i] = transforms_multiply(
            transforms_inv(global_xforms[:, templete_anim.parents[i]]),
            global_xforms[:, i]
        )

    return local_xforms


def get_anim_from_local_transforms(local_xforms, anim):
    """
    Added by hsy on 20210616
    the orients, offsets and parents of the anim is already done.

    Parameters
    ----------

    local_xforms : (F, J, 4, 4) ndarray
        Array of local transforms for
        each frame F and joint J

    anim : Animation
        Input animation without rotations and positions

    Returns
    -------

    anim : Animation
        Input animation with rotations and positions
    """

    anim.positions = local_xforms[:, :, 0:3, 3]
    anim.rotations = Quaternions.from_transforms(local_xforms[:, :, 0:3, 0:3])

    # tmp_rot = Quaternions.from_transforms(local_xforms[:, :, 0:3, 0:3])
    # # euler = np.degrees(tmp_rot[0, 0].euler())
    # # rec_rot = Quaternions.from_euler(np.radians(euler))
    # # print(tmp_rot[0, 0], rec_rot, tmp_rot[0, 0].qs - rec_rot.qs)
    # euler = np.degrees(tmp_rot.euler('xyz'))
    # anim.rotations = Quaternions.from_euler(np.radians(euler))

    # # drop the first frame
    # anim.positions = anim.positions[1:]
    # anim.rotations = anim.rotations[1:]

    return anim

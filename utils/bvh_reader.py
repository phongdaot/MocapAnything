import random

import numpy as np
import torch

from . import animation as animation
from . import bvh as bvh
from .common import get_diameter
from .transforms3d import matrix_to_rotation_6d, quaternion_to_matrix
from .config_utils import load_json

class BVHReader(object):
    """Load BVH file.
    """

    def __init__(
        self,
        max_num_joints=-1,
        crop_size=300,
        rot_type='rot6d',
        reset_pose_prob=0.,
        random_rest_pose=0.,
        trans_only=False,
        trans_mean=None,
        trans_std=None,
        norm_info=None,
        no_pos=False,
        bvh_norm=True,
    ):
        self.max_num_joints = max_num_joints
        self.rot_type = rot_type
        self.crop_size = crop_size
        self.reset_pose_prob = reset_pose_prob
        self.random_rest_pose = random_rest_pose
        self.trans_only = trans_only
        self.trans_mean = np.array(trans_mean)
        self.trans_std = np.array(trans_std)
        if norm_info:
            self.norm_info = load_json(norm_info)
        else:
            self.norm_info = False
        self.no_pos = no_pos
        self.bvh_norm = bvh_norm

    def __call__(self, results):

        if 'motion_path' in results.keys():
            motion_path = results['motion_path']
            try:
                anim, names, frametime = bvh.load(motion_path)
            except Exception as e:
                print(motion_path, e)
        elif 'motion' in results.keys():
            raw_bvh_str = results['motion']
            anim, names, frametime = bvh.load_from_str(raw_bvh_str)
        else:
            raise KeyError(results)

        if random.random() < self.reset_pose_prob:
            anim.random_rest_pose()
        if random.random() < self.random_rest_pose:
            # Generate random rotation matrices with shape (n, 3, 3)
            n = anim.offsets.shape[0]  # Assuming n is the number of joints
            rot_mat = np.random.randn(n, 3, 3)
            # Orthogonalize each 3x3 matrix to make it a valid rotation matrix
            for i in range(n):
                Q, _ = np.linalg.qr(rot_mat[i])
                rot_mat[i] = Q

            # global_positions_before = animation.positions_global(anim)

            anim.apply_rot_on_offset(rot_mat)
            # global_positions_after = animation.positions_global(anim)

            # # Check if global positions before and after are different
            # diff = np.abs(global_positions_before - global_positions_after)
            # max_diff = np.max(diff)
            # print(f'Max difference: {max_diff}')
            # if max_diff > 1e-4:
            #     print(
            #         'ERROR: Global positions changed after applying rotation to offsets'
            #     )

        if ('joint_rename' in results.keys()) and results['joint_rename']:
            names = results['joint_new_names'].copy()

        num_frames, num_joints = anim.rotations.shape
        num_paddings = self.max_num_joints - num_joints

        results['joint_mask'] = np.concatenate(
            (np.ones(num_joints), np.zeros(num_paddings)), axis=-1
        )

        parents = anim.parents.tolist()
        if self.max_num_joints != -1:
            for _ in range(num_paddings):
                parents.append(-2)
                names.append('invalid')

        results['num_joints'] = num_joints
        results['parents'] = parents
        results['joint_names'] = names

        # Do Scaling on bvh and base mesh here, into [0,1]^3?
        # if 'mesh_verts' in results.keys():
        #     verts = results['mesh_verts']
        #     v_max = verts.max(dim=0).values.numpy()
        #     v_min = verts.min(dim=0).values.numpy()
        #     scale_factor = (v_max - v_min).max()

        #     anim.offsets /= scale_factor
        #     anim.offsets[0] -= v_min / scale_factor
        #     anim.positions /= scale_factor

        #     results['mesh_verts'
        #            ] = (results['mesh_verts'] - v_min) / scale_factor
        if self.bvh_norm:
            bone_length = np.linalg.norm(anim.offsets, axis=1)
            diameter, path = get_diameter(anim.parents, bone_length)
            anim.offsets /= diameter
            anim.positions /= diameter

        # Scaling ends

        # global_positions = animation.positions_global(anim)
        # if self.max_num_joints != -1:
        #     global_positions = np.concatenate(
        #         (global_positions, np.zeros((num_frames, num_paddings, 3))),
        #         axis=1
        #     )
        # results['initial_positions'] = global_positions[0]

        global_positions = anim.joint_rest_global_pos()
        if self.max_num_joints != -1:
            global_positions = np.concatenate(
                (global_positions, np.zeros((num_paddings, 3))), axis=0
            )
        results['initial_positions'] = global_positions

        rest_pose = anim.offsets
        if self.max_num_joints != -1:
            rest_pose = np.concatenate(
                (rest_pose, np.zeros((num_paddings, 3))), axis=0
            )
        results['rest_pose'] = rest_pose

        positions = anim.positions
        if self.max_num_joints != -1:
            positions = np.concatenate(
                (positions, np.zeros((num_frames, num_paddings, 3))), axis=1
            )
        results['positions'] = positions

        if self.rot_type == 'rot6d':
            rotations = anim.rotations.qs
            rotations = torch.from_numpy(rotations)
            rotations = matrix_to_rotation_6d(quaternion_to_matrix(rotations))
            rotations = rotations.numpy()

            if self.max_num_joints != -1:
                rotations = np.concatenate(
                    (rotations, np.zeros((num_frames, num_paddings, 6))),
                    axis=1
                )
        else:
            raise NotImplementedError()

        results['rotations'] = rotations

        if self.max_num_joints != -1:
            rot_q = torch.from_numpy(anim.rotations.qs).float()
            rot_6d = matrix_to_rotation_6d(quaternion_to_matrix(rot_q))
            global_xforms = animation.transforms_global(anim)
            joints_pos = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
            root_vel = joints_pos[1:, 0] - joints_pos[:-1, 0]
            if (self.trans_mean != None).all() and (self.trans_std
                                                    != None).all():
                root_vel = (root_vel -
                            self.trans_mean[None]) / self.trans_std[None]

            pos_to_root = joints_pos[:, 1:].copy()
            pos_to_root -= joints_pos[:, :1]

            feature = np.zeros((num_frames, self.max_num_joints, 9))
            feature[1:, 0, :3] = root_vel
            if not self.trans_only:
                if not self.no_pos:
                    feature[:, 1:num_joints, :3] = pos_to_root
                feature[:, :num_joints, 3:] = rot_6d

            if self.norm_info:
                feature[:, 0] = (
                    feature[:, 0] -
                    np.array(self.norm_info['root']['mean'])[None]
                ) / np.array(self.norm_info['root']['std'])[None]

                feature[:, 1:] = (
                    feature[:, 1:] -
                    np.array(self.norm_info['joints']['mean'])[None, None]
                ) / np.array(self.norm_info['joints']['std'])[None, None]

            length = len(feature)
            if length >= self.crop_size:
                idx = random.randint(0, length - self.crop_size)
                feature = feature[idx:idx + self.crop_size]
                results['motion_length'] = self.crop_size
            else:
                padding_length = self.crop_size - length
                D = feature.shape[1:]
                padding_zeros = np.zeros((padding_length, *D), dtype=np.float32)
                feature = np.concatenate([feature, padding_zeros], axis=0)
                results['motion_length'] = length

            assert len(feature) == self.crop_size
            results['motion'] = feature
            results['motion_shape'] = feature.shape
            if length >= self.crop_size:
                results['motion_mask'] = torch.ones(self.crop_size).numpy()
            else:
                results['motion_mask'] = torch.cat(
                    (torch.ones(length), torch.zeros(self.crop_size - length))
                ).numpy()

        return results

    def denormalize_trans(self, motion):
        root_vel = motion[:, 0, :3]
        root_vel_dn = root_vel * self.trans_std[None] + self.trans_mean[None]
        motion[:, 0, :3] = root_vel_dn
        return motion

    def denormalize(self, motion):
        feature = motion.copy()
        feature[:, 0] = (
            feature[:, 0] * np.array(self.norm_info['root']['std'])[None]
        ) + np.array(self.norm_info['root']['mean'])[None]
        feature[:, 1:] = (
            feature[:, 1:] * np.array(self.norm_info['joints']['std'])[None]
        ) + np.array(self.norm_info['joints']['mean'])[None]
        return feature

import copy
import re

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .animation import Animation
from .quaternions import Quaternions
from .transforms3d import euler_angles_to_matrix, matrix_to_euler_angles
import torch
from typing import Dict, Any

channelmap = {'Xrotation': 'x', 'Yrotation': 'y', 'Zrotation': 'z'}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}


def load(filename, start=None, end=None, order=None, world=False):
    """
    Reads a BVH file and constructs an animation

    Parameters
    ----------
    filename: str
        File to be opened

    start : int
        Optional Starting Frame

    end : int
        Optional Ending Frame

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space

    Returns
    -------

    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """

    f = open(filename, 'r')

    i = 0
    active = -1
    end_site = False

    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # added by hsy on 20211007
    end_site_offsets = np.array([]).reshape((0, 3))

    line_ind = 0

    for line in f:

        if 'HIERARCHY' in line:
            continue
        if 'MOTION' in line:
            continue

        # rmatch = re.match(r'ROOT (\w+)', line)
        rmatch = re.match(r'\s*ROOT\s+([^\s\{\}]+)', line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            end_site_offsets = np.append(
                end_site_offsets, np.array([[0, 0, 0]]), axis=0
            )
            active = (len(parents) - 1)
            continue

        if '{' in line:
            continue

        if '}' in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(
            r'\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)', line
        )
        if offmatch:
            if not end_site:
                offsets[active] = np.array(
                    [list(map(float, offmatch.groups()))]
                )
            else:
                end_site_offsets[active] = np.array(
                    [list(map(float, offmatch.groups()))]
                )
            continue

        chanmatch = re.match(r'\s*CHANNELS\s+(\d+)', line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = ''.join([channelmap[p] for p in parts])
                if order == '':
                    order = None
            continue

        # jmatch = re.match('\s*JOINT\s+(\w+)', line)
        jmatch = re.match(r'\s*JOINT\s+([^\s\{\}]+)', line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            end_site_offsets = np.append(
                end_site_offsets, np.array([[0, 0, 0]]), axis=0
            )
            active = (len(parents) - 1)
            continue

        if 'End Site' in line:
            end_site = True
            continue

        fmatch = re.match('\s*Frames:\s+(\d+)', line)
        if fmatch:
            # print(fmatch, int(fmatch.group(1)))
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match('\s*Frame Time:\s+([\d\.]+)', line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        # print(line, line_ind)
        if len(dmatch) <= 1:
            continue
        line_ind += 1
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception('Too many channels! %i' % channels)

            i += 1

    f.close()
    # print("loading order", order)
    # euler = copy.deepcopy(rotations)
    rotations = Quaternions.from_euler(
        np.radians(rotations), order=order, world=world
    )
    # rec_euler = np.degrees(rotations.euler(order[::-1]))
    # print("euler and rec_euler diff", np.mean(np.abs(euler) - np.abs(rec_euler)))
    # print(len(rotations), len(positions))

    return (
        Animation(
            rotations, positions, orients, offsets, parents, end_site_offsets
        ), names, frametime
    )


# TODO: merge with normal load
def load_from_str(raw_bvh_str, start=None, end=None, order=None, world=False):
    i = 0
    active = -1
    end_site = False

    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # added by hsy on 20211007
    end_site_offsets = np.array([]).reshape((0, 3))

    line_ind = 0

    f = raw_bvh_str.split('\n')
    for line in f:

        if 'HIERARCHY' in line:
            continue
        if 'MOTION' in line:
            continue

        # rmatch = re.match(r'ROOT (\w+)', line)
        rmatch = re.match(r'\s*ROOT\s+([^\s\{\}]+)', line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            end_site_offsets = np.append(
                end_site_offsets, np.array([[0, 0, 0]]), axis=0
            )
            active = (len(parents) - 1)
            continue

        if '{' in line:
            continue

        if '}' in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(
            r'\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)', line
        )
        if offmatch:
            if not end_site:
                offsets[active] = np.array(
                    [list(map(float, offmatch.groups()))]
                )
            else:
                end_site_offsets[active] = np.array(
                    [list(map(float, offmatch.groups()))]
                )
            continue

        chanmatch = re.match(r'\s*CHANNELS\s+(\d+)', line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = ''.join([channelmap[p] for p in parts])
                if order == '':
                    order = None
            continue

        # jmatch = re.match('\s*JOINT\s+(\w+)', line)
        jmatch = re.match(r'\s*JOINT\s+([^\s\{\}]+)', line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            end_site_offsets = np.append(
                end_site_offsets, np.array([[0, 0, 0]]), axis=0
            )
            active = (len(parents) - 1)
            continue

        if 'End Site' in line:
            end_site = True
            continue

        fmatch = re.match('\s*Frames:\s+(\d+)', line)
        if fmatch:
            # print(fmatch, int(fmatch.group(1)))
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match('\s*Frame Time:\s+([\d\.]+)', line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        # print(line, line_ind)
        if len(dmatch) <= 1:
            continue
        line_ind += 1
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception('Too many channels! %i' % channels)

            i += 1

    # print("loading order", order)
    # euler = copy.deepcopy(rotations)
    rotations = Quaternions.from_euler(
        np.radians(rotations), order=order, world=world
    )
    # rec_euler = np.degrees(rotations.euler(order[::-1]))
    # print("euler and rec_euler diff", np.mean(np.abs(euler) - np.abs(rec_euler)))
    # print(len(rotations), len(positions))

    return (
        Animation(
            rotations, positions, orients, offsets, parents, end_site_offsets
        ), names, frametime
    )


def interpolation_anim(anim: Animation, fps: float) -> Animation:
    """
    Interpolate animation to a new frame rate.
    该功能未经验证，没有增加额外参数故无需其它适配

    Args:
        anim: Animation object to be interpolated.
        fps: Target frames per second.

    Returns:
        Interpolated animation object.
    """
    now_fps = 1 / anim.frametime
    if abs(now_fps - fps) < 0.01:
        return anim
    joint_num = anim.rotations.shape[1]
    now_fnum = len(anim.rotations)
    new_fnum = round((fps * now_fnum) / now_fps)
    key_times = [i for i in range(now_fnum)]
    interp_times = [
        i * ((now_fnum - 1) / (new_fnum - 1)) for i in range(new_fnum)
    ]

    # interpolating rotation by slerp
    quaternions = anim.rotations
    mats = quaternions.transforms()
    interp_mats = np.zeros((new_fnum, joint_num, 3, 3))
    for j_id in range(joint_num):
        mats_j = mats[:, j_id]
        key_rots = R.from_matrix(mats_j)
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(interp_times)
        interp_mats[:, j_id] = interp_rots.as_matrix()
    interp_quat = Quaternions.from_transforms(interp_mats)
    # interpolating trans by linear
    trans = anim.positions[:, 0]
    new_trans = np.zeros((new_fnum, trans.shape[-1]))
    for dim in range(trans.shape[-1]):
        new_trans[:, dim] = np.interp(interp_times, key_times, trans[:, dim])
    interp_positions = np.zeros((new_fnum, joint_num, trans.shape[1]))
    interp_positions[:, 0] = new_trans
    anim.frametime = 1 / fps
    anim.rotations = interp_quat
    anim.positions = interp_positions
    return anim


def save(
    filename,
    anim,
    names=None,
    frametime=1.0 / 24.0,
    order='zyx',
    positions=False,
    mirror=False,
    skd=None,
):
    """
    Saves an Animation to file as BVH

    Parameters
    ----------
    filename: str
        File to be saved to

    anim : Animation
        Animation to save

    names : [str]
        List of joint names

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    frametime : float
        Optional Animation Frame time

    positions : bool
        Optional specfier to save bone
        positions for each frame

    """
    # anim = interpolation_anim(anim, frametime)

    if names is None:
        names = ['joint_' + str(i) for i in range(len(anim.parents))]

    with open(filename, 'w') as f:

        t = ''
        f.write('%sHIERARCHY\n' % t)
        f.write('%sROOT %s\n' % (t, names[0]))
        f.write('%s{\n' % t)
        t += '\t'

        f.write(
            '%sOFFSET %f %f %f\n' %
            (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2])
        )
        f.write(
            '%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n' % (
                t, channelmap_inv[order[0]], channelmap_inv[order[1]],
                channelmap_inv[order[2]]
            )
        )

        save_order = [0]

        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t = save_joint(
                    f,
                    anim,
                    names,
                    t,
                    i,
                    save_order,
                    order=order,
                    positions=positions
                )

        t = t[:-1]
        f.write('%s}\n' % t)

        f.write('MOTION\n')
        f.write('Frames: %i\n' % anim.shape[0])
        f.write('Frame Time: %f\n' % frametime)

        rots_tmp = np.degrees(anim.rotations.euler(order=order[::-1]))
        poss_tmp = anim.positions

        rots = copy.deepcopy(rots_tmp)
        poss = copy.deepcopy(poss_tmp)

        if mirror:
            left_inds = skd.left_inds
            right_inds = skd.right_inds
            body_inds = skd.body_inds
            rots[:,
                 left_inds] = rots_tmp[:,
                                       right_inds] * np.array([[[1, -1, -1]]])
            rots[:,
                 right_inds] = rots_tmp[:,
                                        left_inds] * np.array([[[1, -1, -1]]])
            rots[:,
                 body_inds] = rots_tmp[:,
                                       body_inds] * np.array([[[1, -1, -1]]])
            poss[:, :, 0] *= -1.0

        for i in range(anim.shape[0]):
            for j in save_order:

                if positions or j == 0:

                    f.write(
                        '%f %f %f %f %f %f ' % (
                            poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                            rots[i, j, ordermap[order[0]]],
                            rots[i, j, ordermap[order[1]]],
                            rots[i, j, ordermap[order[2]]]
                        )
                    )

                else:

                    f.write(
                        '%f %f %f ' % (
                            rots[i, j, ordermap[order[0]]],
                            rots[i, j, ordermap[order[1]]],
                            rots[i, j, ordermap[order[2]]]
                        )
                    )

            f.write('\n')


def interpolation_dict(data: Dict[str, Any], fps: float) -> Dict[str, Any]:
    """
    Interpolate animation data in a dictionary to a new frame rate.

    Args:
        data: Dictionary containing animation data with keys 'frametime', 'rotations', and 'positions'.
        fps: Target frames per second.

    Returns:
        Dictionary with interpolated animation data.
    """
    now_fps = 1 / data['frametime']
    if abs(now_fps - fps) < 0.01:
        return data
    joint_num = data['rotations'].shape[1]
    mats = euler_angles_to_matrix(
        torch.from_numpy(np.deg2rad(data['rotations'])), 'ZYX'
    ).numpy()
    now_fnum = len(mats)
    new_fnum = round((fps * now_fnum) / now_fps)
    key_times = [i for i in range(now_fnum)]
    interp_times = [
        i * ((now_fnum - 1) / (new_fnum - 1)) for i in range(new_fnum)
    ]
    interp_times[-1] -= 1e-9

    # interpolating rotation by slerp
    interp_mats = np.zeros((new_fnum, joint_num, 3, 3))
    for j_id in range(joint_num):
        mats_j = mats[:, j_id]
        key_rots = R.from_matrix(mats_j)
        slerp = Slerp(key_times, key_rots)
        interp_rots = slerp(interp_times)
        interp_mats[:, j_id] = interp_rots.as_matrix()

    # interpolating trans by linear
    trans = data['positions'][:, 0]
    new_trans = np.zeros((new_fnum, trans.shape[-1]))
    for dim in range(trans.shape[-1]):
        new_trans[:, dim] = np.interp(interp_times, key_times, trans[:, dim])
    interp_positions = np.zeros((new_fnum, joint_num, trans.shape[1]))
    interp_positions[:, 0] = new_trans

    data['frametime'] = 1 / fps
    data['rotations'] = np.degrees(
        matrix_to_euler_angles(torch.from_numpy(interp_mats), 'ZYX').numpy()
    )
    data['positions'] = interp_positions
    return data


def save_dict(filename, data, fps=30, save_positions=False):
    """ Save a joint hierarchy to a file.

    Args:
        filename (str): The output will save on the bvh file.
        data (dict): The data to save.(rotations, positions, offsets, parents, names, order, frametime)
        save_positions (bool): Whether to save all of joint positions on MOTION. (False is recommended.)
    """

    data = interpolation_dict(data.copy(), fps)
    order = data['order']
    frametime = data['frametime']

    with open(filename, 'w') as f:

        t = ''
        f.write('%sHIERARCHY\n' % t)
        f.write('%sROOT %s\n' % (t, data['names'][0]))
        f.write('%s{\n' % t)
        t += '\t'

        f.write(
            '%sOFFSET %f %f %f\n' % (
                t, data['offsets'][0, 0], data['offsets'][0, 1],
                data['offsets'][0, 2]
            )
        )
        f.write(
            '%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n' % (
                t, channelmap_inv[order[0]], channelmap_inv[order[1]],
                channelmap_inv[order[2]]
            )
        )

        save_order = [0]

        for i in range(len(data['parents'])):
            if data['parents'][i] == 0:
                t = save_joint_dict(
                    f,
                    data,
                    t,
                    i,
                    save_order,
                    order=order,
                    save_positions=save_positions
                )

        t = t[:-1]
        f.write('%s}\n' % t)

        rots, poss = data['rotations'], data['positions']

        f.write('MOTION\n')
        f.write('Frames: %i\n' % len(rots))
        f.write('Frame Time: %f\n' % frametime)

        for i in range(rots.shape[0]):
            for j in save_order:

                if save_positions or j == 0:

                    f.write(
                        '%f %f %f %f %f %f ' % (
                            poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                            rots[i, j, 0], rots[i, j, 1], rots[i, j, 2]
                        )
                    )

                else:

                    f.write(
                        '%f %f %f ' %
                        (rots[i, j, 0], rots[i, j, 1], rots[i, j, 2])
                    )

            f.write('\n')


def save_joint_dict(
    f, data, t, i, save_order, order='zyx', save_positions=False
):

    save_order.append(i)

    f.write('%sJOINT %s\n' % (t, data['names'][i]))
    f.write('%s{\n' % t)
    t += '\t'

    f.write(
        '%sOFFSET %f %f %f\n' % (
            t, data['offsets'][i, 0], data['offsets'][i, 1], data['offsets'][i,
                                                                             2]
        )
    )

    if save_positions:
        f.write(
            '%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n' % (
                t, channelmap_inv[order[0]], channelmap_inv[order[1]],
                channelmap_inv[order[2]]
            )
        )
    else:
        f.write(
            '%sCHANNELS 3 %s %s %s\n' % (
                t, channelmap_inv[order[0]], channelmap_inv[order[1]],
                channelmap_inv[order[2]]
            )
        )

    end_site = True

    for j in range(len(data['parents'])):
        if data['parents'][j] == i:
            t = save_joint_dict(
                f,
                data,
                t,
                j,
                save_order,
                order=order,
                save_positions=save_positions
            )
            end_site = False

    if end_site:
        f.write('%sEnd Site\n' % t)
        f.write('%s{\n' % t)
        t += '\t'
        f.write('%sOFFSET %f %f %f\n' % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write('%s}\n' % t)

    t = t[:-1]
    f.write('%s}\n' % t)

    return t


def save_joint(f, anim, names, t, i, save_order, order='zyx', positions=False):

    save_order.append(i)

    f.write('%sJOINT %s\n' % (t, names[i]))
    f.write('%s{\n' % t)
    t += '\t'

    f.write(
        '%sOFFSET %f %f %f\n' %
        (t, anim.offsets[i, 0], anim.offsets[i, 1], anim.offsets[i, 2])
    )

    if positions:
        f.write(
            '%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n' % (
                t, channelmap_inv[order[0]], channelmap_inv[order[1]],
                channelmap_inv[order[2]]
            )
        )
    else:
        f.write(
            '%sCHANNELS 3 %s %s %s\n' % (
                t, channelmap_inv[order[0]], channelmap_inv[order[1]],
                channelmap_inv[order[2]]
            )
        )

    end_site = True

    for j in range(anim.shape[1]):
        if anim.parents[j] == i:
            t = save_joint(
                f,
                anim,
                names,
                t,
                j,
                save_order,
                order=order,
                positions=positions
            )
            end_site = False

    if end_site:
        f.write('%sEnd Site\n' % t)
        f.write('%s{\n' % t)
        t += '\t'
        # f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        f.write(
            '%sOFFSET %f %f %f\n' % (
                t, anim.end_site_offsets[i, 0], anim.end_site_offsets[i, 1],
                anim.end_site_offsets[i, 2]
            )
        )
        t = t[:-1]
        f.write('%s}\n' % t)

    t = t[:-1]
    f.write('%s}\n' % t)

    return t


def load_eulers(
    filename, start=None, end=None, order=None, world=False, skd=None
):
    """
    Reads a BVH file and constructs an animation

    Parameters
    ----------
    filename: str
        File to be opened

    start : int
        Optional Starting Frame

    end : int
        Optional Ending Frame

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space

    Returns
    -------

    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """

    f = open(filename, 'r')

    i = 0
    active = -1
    end_site = False

    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # added by hsy on 20211007
    end_site_offsets = np.array([]).reshape((0, 3))

    line_ind = 0

    for line in f:

        if 'HIERARCHY' in line:
            continue
        if 'MOTION' in line:
            continue

        rmatch = re.match(r'ROOT (\w+)', line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            end_site_offsets = np.append(
                end_site_offsets, np.array([[0, 0, 0]]), axis=0
            )
            active = (len(parents) - 1)
            continue

        if '{' in line:
            continue

        if '}' in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(
            r'\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)', line
        )
        if offmatch:
            if not end_site:
                offsets[active] = np.array(
                    [list(map(float, offmatch.groups()))]
                )
            else:
                end_site_offsets[active] = np.array(
                    [list(map(float, offmatch.groups()))]
                )
            continue

        chanmatch = re.match(r'\s*CHANNELS\s+(\d+)', line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = ''.join([channelmap[p] for p in parts])
            continue

        jmatch = re.match('\s*JOINT\s+(\w+)', line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            end_site_offsets = np.append(
                end_site_offsets, np.array([[0, 0, 0]]), axis=0
            )
            active = (len(parents) - 1)
            continue

        if 'End Site' in line:
            end_site = True
            continue

        fmatch = re.match('\s*Frames:\s+(\d+)', line)
        if fmatch:
            # print(fmatch, int(fmatch.group(1)))
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match('\s*Frame Time:\s+([\d\.]+)', line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        # print(line, line_ind)
        if len(dmatch) <= 1:
            continue
        line_ind += 1
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception('Too many channels! %i' % channels)

            i += 1

    f.close()
    # print("loading order", order)
    euler = copy.deepcopy(rotations)

    rotations = Quaternions.from_euler(
        np.radians(rotations), order=order, world=world
    )

    return (
        Animation(
            rotations, positions, orients, offsets, parents, end_site_offsets
        ), names, frametime, euler
    )


def save_eulers(
    filename,
    anim,
    euler,
    names=None,
    frametime=1.0 / 24.0,
    order='zyx',
    positions=False,
    orients=True,
    mirror=False,
    skd=None,
):
    """
    Saves an Animation to file as BVH

    Parameters
    ----------
    filename: str
        File to be saved to

    anim : Animation
        Animation to save

    names : [str]
        List of joint names

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    frametime : float
        Optional Animation Frame time

    positions : bool
        Optional specfier to save bone
        positions for each frame

    orients : bool
        Multiply joint orients to the rotations
        before saving.

    """

    if names is None:
        names = ['joint_' + str(i) for i in range(len(anim.parents))]

    with open(filename, 'w') as f:

        t = ''
        f.write('%sHIERARCHY\n' % t)
        f.write('%sROOT %s\n' % (t, names[0]))
        f.write('%s{\n' % t)
        t += '\t'

        f.write(
            '%sOFFSET %f %f %f\n' %
            (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2])
        )
        f.write(
            '%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n' % (
                t, channelmap_inv[order[0]], channelmap_inv[order[1]],
                channelmap_inv[order[2]]
            )
        )

        save_order = [0]

        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t = save_joint(
                    f,
                    anim,
                    names,
                    t,
                    i,
                    save_order,
                    order=order,
                    positions=positions
                )

        t = t[:-1]
        f.write('%s}\n' % t)

        f.write('MOTION\n')
        f.write('Frames: %i\n' % anim.shape[0])
        f.write('Frame Time: %f\n' % frametime)

        # 这里一定要保证eulers的顺序是zyx，如果eulers是直接从bvh中读取的，那么其顺序是xyz
        rots_tmp = euler[:, :, ::-1]
        poss_tmp = anim.positions

        rots = copy.deepcopy(rots_tmp)
        poss = copy.deepcopy(poss_tmp)

        if mirror:
            left_inds = skd.left_inds
            right_inds = skd.right_inds
            body_inds = skd.body_inds
            rots[:,
                 left_inds] = rots_tmp[:,
                                       right_inds] * np.array([[[1, -1, -1]]])
            rots[:,
                 right_inds] = rots_tmp[:,
                                        left_inds] * np.array([[[1, -1, -1]]])
            rots[:,
                 body_inds] = rots_tmp[:,
                                       body_inds] * np.array([[[1, -1, -1]]])
            poss[:, :, 0] *= -1.0

        for i in range(euler.shape[0]):
            for j in save_order:

                if positions or j == 0:

                    f.write(
                        '%f %f %f %f %f %f ' % (
                            poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                            rots[i, j, ordermap[order[0]]],
                            rots[i, j, ordermap[order[1]]],
                            rots[i, j, ordermap[order[2]]]
                        )
                    )

                else:

                    f.write(
                        '%f %f %f ' % (
                            rots[i, j, ordermap[order[0]]],
                            rots[i, j, ordermap[order[1]]],
                            rots[i, j, ordermap[order[2]]]
                        )
                    )

            f.write('\n')

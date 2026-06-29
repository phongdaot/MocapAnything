### mesh.py ###

from glob import glob
import shutil
import subprocess
from typing import Tuple, Optional
import numpy as np
from .transforms3d import quaternion_to_axis_angle
from . import bvh as BVH
import trimesh
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from utils.visualization import add_background_to_image_folder, convert_images_to_video
from .common import get_diameter, sm_loop, interchange_y_z_axis, rot_y
from .transforms3d import axis_angle_to_matrix

def rotate_mesh_sequence_y_axis(
    vertices: np.ndarray,
    angles: float | np.ndarray,
    faces: np.ndarray,
    root_positions: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate a mesh sequence around each frame's root position along the Y axis,
    and compute per-vertex normals after rotation.

    Args:
        vertices (np.ndarray): Mesh vertices of shape (F, N, 3).
        root_positions (np.ndarray): Root positions of shape (F, 3).
        angles (float or np.ndarray): Rotation angles in degrees.
            Can be a single float or an array of shape (F,).
        faces (np.ndarray): Face indices of shape (T, 3), shared across frames.

    Returns:
        tuple:
            - np.ndarray: Rotated mesh sequence of shape (F, N, 3).
            - np.ndarray: Per-frame vertex normals of shape (F, N, 3).
    """
    F, N, _ = vertices.shape
    if isinstance(angles, (float, int)):
        angles = np.full((F,), angles)
    else:
        angles = np.asarray(angles)
        assert angles.shape == (F,), "angles must be a float or shape (F,)"

    # Create batched rotation matrices
    rot = R.from_euler('y', angles, degrees=True)
    rot_matrices = rot.as_matrix()  # (F, 3, 3)

    # Rotate mesh
    if root_positions:
        assert root_positions.shape == (F, 3), "root_positions must be (F, 3)"
        centered = vertices - root_positions[:, None, :]  # (F, N, 3)
        rotated = rot_matrices @ centered.transpose(0, 2, 1)  # (F, 3, N)
        rotated = rotated.transpose(0, 2, 1)  # (F, N, 3)
        rotated += root_positions[:, None, :]  # (F, N, 3s)
    else:
        rotated = rot_matrices @ vertices.transpose(0, 2, 1)
        rotated = rotated.transpose(0, 2, 1)

    # Compute normals
    normals = np.zeros_like(rotated)  # (F, N, 3)
    v0 = rotated[:, faces[:, 0]]  # (F, T, 3)
    v1 = rotated[:, faces[:, 1]]
    v2 = rotated[:, faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # (F, T, 3)

    for f in range(F):
        for i in range(3):
            np.add.at(normals[f], faces[:, i], face_normals[f])

    norm = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8
    normals /= norm  # (F, N, 3)

    return rotated, normals

def compute_rest_joints(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """
    Compute global joint positions in T-pose from BVH offsets and parent structure.

    Args:
        offsets (np.ndarray): Local offsets of each joint in BVH hierarchy.
        parents (np.ndarray): Parent indices for each joint.

    Returns:
        np.ndarray: Global joint positions in rest (T) pose.
    """
    J = offsets.shape[0]
    joints = np.zeros((J, 3))
    for j in range(J):
        if parents[j] < 0:
            joints[j] = offsets[j]
        else:
            joints[j] = joints[parents[j]] + offsets[j]
    return joints

def lbs(
    pose: torch.Tensor,
    v_template: torch.Tensor,
    rest_joints: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Linear Blend Skinning with the given pose and rest parameters.

    Args:
        pose (torch.Tensor): (B, (J+1)*3) Pose parameters in axis-angle format.
        v_template (torch.Tensor): (V, 3) Template mesh vertices.
        rest_joints (torch.Tensor): (B, J+1, 3) Rest pose global joint positions.
        parents (torch.Tensor): (J,) Parent joint indices.
        lbs_weights (torch.Tensor): (V, J+1) Skinning weights.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - verts (B, V, 3): Deformed mesh vertices.
            - joints (B, J+1, 3): Transformed joint locations.
    """
    batch_size = pose.shape[0]
    device, dtype = pose.device, pose.dtype

    # Get the joints
    # NxJx3 array
    J = rest_joints
    rot_mats = axis_angle_to_matrix(pose).view([batch_size, -1, 3, 3])
    v_posed = v_template.repeat(batch_size, 1, 1)

    # Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)

    # Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J.shape[1]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones(
        [batch_size, v_posed.shape[1], 1], dtype=dtype, device=device
    )
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]
    return verts, J_transformed

def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Create homogeneous transformation matrices from rotation and translation.

    Args:
        R (torch.Tensor): (B, 3, 3) Rotation matrices.
        t (torch.Tensor): (B, 3, 1) Translation vectors.
    """
    return torch.cat(
        [F.pad(R, [0, 0, 0, 1]),
         F.pad(t, [0, 0, 0, 1], value=1)], dim=2
    )


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rigid body transformations down a kinematic tree.

    Args:
        rot_mats (torch.Tensor): (B, J, 3, 3) Rotation matrices for each joint.
        joints (torch.Tensor): (B, J, 3) Joint positions.
        parents (torch.Tensor): (J,) Parent joint indices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - posed_joints (B, J, 3): Transformed joints.
            - rel_transforms (B, J, 4, 4): Relative transform matrices.
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(
            transform_chain[parents[i]], transforms_mat[:, i]
        )
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms

def read_obj_mesh(
    file_path: str,
) -> Tuple[
    np.ndarray,           # vertices: (V, 3)
    np.ndarray,           # faces: (F, 3)
    Optional[np.ndarray], # uvs: (VT, 2) or None
    Optional[np.ndarray], # face_uvs: (F, 3) or None
]:
    """
    Read an OBJ file and extract vertex positions, face indices,
    UV coordinates, and per-face UV indices.
    """
    vertices = []
    uvs = []
    faces = []
    face_uvs = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                vertices.append([
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                ])

            elif line.startswith("vt "):
                parts = line.split()
                u = float(parts[1])
                v = float(parts[2])
                uvs.append([u, v])

            elif line.startswith("vn "):
                continue

            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                face_uv = []

                for part in parts:
                    tokens = part.split("/")

                    v_idx = int(tokens[0]) - 1
                    face.append(v_idx)

                    if len(tokens) > 1 and tokens[1] != "":
                        vt_idx = int(tokens[1]) - 1
                        face_uv.append(vt_idx)
                    else:
                        face_uv.append(-1)

                if len(face) != 3:
                    raise ValueError(
                        f"Only triangular faces are supported, got face with {len(face)} vertices in {file_path}"
                    )

                faces.append(face)

                if any(idx != -1 for idx in face_uv):
                    face_uvs.append(face_uv)
                else:
                    face_uvs.append([-1, -1, -1])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    uvs_arr = np.asarray(uvs, dtype=np.float32) if len(uvs) > 0 else None

    if len(face_uvs) > 0 and any(idx != -1 for tri in face_uvs for idx in tri):
        face_uvs_arr = np.asarray(face_uvs, dtype=np.int64)
    else:
        face_uvs_arr = None

    return vertices, faces, uvs_arr, face_uvs_arr


def extract_mesh_from_bvh(
    bvh_pth: str,
    template_pth: str,
    save_root: str,
    lbs_weights_pth: str = "...",
    unicube: bool = False,
    vis_skeleton: bool = False,
    return_root_pos: bool = False,
    mesh_format: str = "obj",
    scale: float = 1.0,
    mesh_scale: float = 1.0,
    azim: float = 0.0,
):
    """
    Extract mesh animation from a BVH file and save as mesh sequence.
    """
    skin_weights = torch.from_numpy(np.load(lbs_weights_pth))
    temp_vertices, temp_faces, temp_uvs, temp_face_uvs = read_obj_mesh(template_pth)

    if unicube:
        v_max = temp_vertices.max(axis=0)
        v_min = temp_vertices.min(axis=0)
        scale_factor = (v_max - v_min).max()
        temp_vertices = temp_vertices - v_min
        temp_vertices = temp_vertices / scale_factor

    anim, names, ft = BVH.load(bvh_pth)

    bvh_offset = anim.offsets.copy()
    root_to_pelvis = bvh_offset[0]
    pelvis_pos = anim.positions[:, 0]
    trans = pelvis_pos - root_to_pelvis

    pose = quaternion_to_axis_angle(torch.from_numpy(np.array(anim.rotations)))
    v_template = torch.from_numpy(temp_vertices) * mesh_scale
    rest_joints = torch.from_numpy(
        compute_rest_joints(anim.offsets, anim.parents)
    )

    verts, J_transformed = lbs(
        pose.float(),
        v_template.float(),
        rest_joints.repeat(pose.shape[0], 1, 1).float(),
        anim.parents,
        skin_weights.float(),
    )

    trans_torch = torch.from_numpy(trans).to(dtype=verts.dtype, device=verts.device)

    J_transformed = J_transformed + trans_torch[:, None]
    verts = verts + trans_torch[:, None]

    verts = verts * scale
    pelvis_pos_scaled = pelvis_pos * scale

    if azim != 0:
        R = rot_y(azim).astype(np.float32)
        R_torch = torch.from_numpy(R).to(device=verts.device, dtype=verts.dtype)

        verts = torch.matmul(verts, R_torch.T)
        pelvis_pos_scaled = pelvis_pos_scaled @ R.T

    os.makedirs(save_root, exist_ok=True)
    has_uv = temp_uvs is not None and temp_face_uvs is not None

    for i in tqdm(range(verts.shape[0]), desc="exporting mesh"):
        frame_vertices = verts[i].detach().cpu().numpy()
        out_path = os.path.join(save_root, f"{i:04d}.{mesh_format}")

        if has_uv and mesh_format.lower() == "obj":
            with open(out_path, "w") as f:
                for v in frame_vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")

                for uv in temp_uvs:
                    f.write(f"vt {uv[0]} {uv[1]}\n")

                for face, face_uv in zip(temp_faces, temp_face_uvs):
                    f.write(
                        f"f "
                        f"{face[0] + 1}/{face_uv[0] + 1} "
                        f"{face[1] + 1}/{face_uv[1] + 1} "
                        f"{face[2] + 1}/{face_uv[2] + 1}\n"
                    )
        else:
            if has_uv:
                visual = trimesh.visual.texture.TextureVisuals(uv=temp_uvs)
                m_mesh = trimesh.Trimesh(
                    vertices=frame_vertices,
                    faces=temp_faces,
                    visual=visual,
                    process=False,
                )
                m_mesh.metadata["face_uvs"] = temp_face_uvs
            else:
                m_mesh = trimesh.Trimesh(
                    vertices=frame_vertices,
                    faces=temp_faces,
                    process=False,
                )
            m_mesh.export(out_path)

    if return_root_pos:
        return pelvis_pos_scaled


def blender_visualize_character_motion(
    output_dir: str,
    motion_path: str,
    base_mesh_path: str = None,
    skin_weight_path: str = None,
    character_folder: str = None,
    unicube: bool = False,
    fps: int = 30,
    blender_path: str = None,
    scene: Literal["origin", "blank"] = "origin",
    audio_path: str = None,
    hdri_path: str = None,
    view_scale: float = 1.0,
    object_position: float = 0.35,
    bg_color: Tuple = (255, 255, 255),
    mesh_format: str = "obj",
    camera_trace: bool = True,
    traj_smooth: float = 0.8,
    auto_scale: Literal[None, "bvh", "base_mesh"] = None,
    mesh_scale: float = 1.0,
    azim: float = 0.0,
) -> None:
    """
    Visualize character motion by converting BVH animation to a mesh sequence,
    rotating it by azim inside mesh extraction, then rendering it in Blender.
    """

    mesh_path = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_path, exist_ok=True)

    motion_name = motion_path.split("/")[-1][:-4]

    if character_folder:
        base_mesh_path = os.path.join(character_folder, "base_mesh.obj")
        skin_weight_path = os.path.join(character_folder, "skinning_weights.npy")
        if not os.path.exists(skin_weight_path):
            skin_weight_path = os.path.join(character_folder, "skin_weights.npy")

    if auto_scale == "bvh":
        anim, names, frametime = BVH.load(motion_path)
        bone_length = np.linalg.norm(anim.offsets, axis=1)
        diameter, path = get_diameter(anim.parents, bone_length)
        scale = 1 / diameter
    elif auto_scale == "base_mesh":
        temp_vertices, temp_faces, temp_uvs, temp_face_uvs = read_obj_mesh(base_mesh_path)
        v_max = temp_vertices.max(axis=0)
        v_min = temp_vertices.min(axis=0)
        scale_factor = (v_max - v_min).max()
        scale = 1 / scale_factor
    else:
        scale = 1.0

    root_pos = extract_mesh_from_bvh(
        motion_path,
        base_mesh_path,
        mesh_path,
        skin_weight_path,
        unicube=unicube,
        vis_skeleton=False,
        return_root_pos=True,
        mesh_format=mesh_format,
        scale=scale,
        mesh_scale=mesh_scale,
        azim=azim,
    )

    root_pos_sm = sm_loop(root_pos, traj_smooth)
    height = object_position

    if camera_trace:
        root_pos_sm[:, 2] = -root_pos_sm[:, 2]
        camera_traj = root_pos_sm + np.array([0, height, -3.8 / view_scale])
    else:
        camera_traj = np.tile(
            np.array([0, height, -3.8 / view_scale]),
            (root_pos_sm.shape[0], 1),
        )

    camera_traj_xyz = interchange_y_z_axis(camera_traj)
    camera_traj_pth = os.path.join(output_dir, "camera_trajectory.npy")
    np.save(camera_traj_pth, camera_traj_xyz)

    if hdri_path:
        scene = "blank"

    blender_visualize_single_mesh_sequence(
        output_dir=output_dir,
        object_path=mesh_path,
        fps=fps,
        blender_path=blender_path,
        scene=scene,
        audio_path=audio_path,
        character_folder=character_folder,
        hdri_path=hdri_path,
        camera_traj=camera_traj_pth,
        bg_color=bg_color,
        video_name=motion_name,
    )


def blender_visualize_single_mesh_sequence(
    output_dir: str,
    object_path: str = None,
    fps: int = 30,
    blender_path: str = None,
    scene: Literal["origin", "blank"] = "origin",
    audio_path: str = None,
    character_folder: str = None,
    hdri_path: str = None,
    camera_traj: str = None,
    bg_color=(255, 255, 255),
    video_name: str = "video",
    log_file: str = None,
) -> None:
    """
    Render a sequence of mesh frames using Blender and generate a video.
    """
    output_image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(output_image_dir):
        os.mkdir(output_image_dir)

    current_dir = os.path.dirname(__file__)
    blender_py_path = os.path.join(current_dir, "blender_mesh_utils.py")
    scene_blend_path = os.path.join(current_dir, scene + ".blend")
    if not os.path.exists(scene_blend_path):
        scene_blend_path = os.path.join(
            os.path.dirname(current_dir), "preprocess", scene + ".blend"
        )

    blender_cmd = [
        blender_path,
        "-P", blender_py_path,
        "-b",
        "--",
        "--folder", object_path,
        "--scene", scene_blend_path,
        "--out-folder", output_image_dir,
    ]

    if character_folder:
        blender_cmd += ["--use-mtl", "--character-folder", character_folder]
    if hdri_path:
        blender_cmd += ["--hdri", hdri_path]
    if camera_traj:
        blender_cmd += ["--camera-traj", camera_traj]

    if log_file is None:
        log_file = os.path.join(output_dir, "blender.log")

    with open(log_file, "w") as lf:
        subprocess.run(blender_cmd, stdout=lf, stderr=lf, check=True)

    add_background_to_image_folder(output_image_dir, bg_color)
    video_pth = os.path.join(output_dir, f"{video_name}.mp4")
    convert_images_to_video(output_image_dir, video_pth, fps)
    

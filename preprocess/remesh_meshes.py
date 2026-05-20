"""
Voxel-remesh per-character animated meshes via Blender.

Input  : npz files under --input_root, each containing
           - vertices : (T, N, 3) per-frame deformed vertices
           - faces    : (F, 3)    triangle indices shared across frames
Output : --output_root/{name}.npz with
           - vertices : (T, target_num_points, 3) float16
           - normals  : (T, target_num_points, 3) float16

Run with Blender's bundled Python so `bpy` is available, e.g.
    blender --background --python preprocess/remesh_meshes.py -- \\
        --input_root zoo/anim_meshes --output_root zoo/remesh_npz
"""
import argparse
import os
import sys
from glob import glob
from multiprocessing import Pool, cpu_count

import bpy
import numpy as np


def _compute_vertex_normals_smooth(verts: np.ndarray, faces_tri: np.ndarray) -> np.ndarray:
    v0 = verts[faces_tri[:, 0]]
    v1 = verts[faces_tri[:, 1]]
    v2 = verts[faces_tri[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    n = np.zeros_like(verts, dtype=np.float32)
    np.add.at(n, faces_tri[:, 0], fn)
    np.add.at(n, faces_tri[:, 1], fn)
    np.add.at(n, faces_tri[:, 2], fn)
    lens = np.linalg.norm(n, axis=1)
    mask = lens > 1e-20
    n[mask] /= lens[mask][:, None]
    return n


def voxelize_mesh_numpy(vertices, faces,
                       voxel_size=0.001, adaptivity=0.8,
                       remove_disconnected=True, disconnected_threshold=0.0001):
    mesh = bpy.data.meshes.new("tmp_input_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.validate(clean_customdata=True)
    mesh.update()

    obj = bpy.data.objects.new("tmp_input_obj", mesh)
    bpy.context.scene.collection.objects.link(obj)

    remesh = obj.modifiers.new("Z_VoxelRemesh", 'REMESH')
    remesh.mode = 'VOXEL'
    remesh.voxel_size = float(voxel_size)
    remesh.adaptivity = float(adaptivity)
    if hasattr(remesh, "use_remove_disconnected"):
        remesh.use_remove_disconnected = bool(remove_disconnected)
    if hasattr(remesh, "threshold"):
        remesh.threshold = float(disconnected_threshold)

    tri = obj.modifiers.new("Z_Triangulate", 'TRIANGULATE')
    if hasattr(tri, "keep_custom_normals"):
        tri.keep_custom_normals = True

    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()

    mesh_eval.calc_loop_triangles()
    loop_tris = mesh_eval.loop_triangles
    faces_np = np.empty((len(loop_tris), 3), dtype=np.int32)
    loop_tris.foreach_get("vertices", faces_np.ravel())

    verts_np = np.empty((len(mesh_eval.vertices), 3), dtype=np.float32)
    mesh_eval.vertices.foreach_get("co", verts_np.ravel())

    normals_np = np.empty((len(mesh_eval.vertices), 3), dtype=np.float32)
    mesh_eval.vertices.foreach_get("normal", normals_np.ravel())

    if not np.any(np.abs(normals_np) > 1e-12):
        normals_np = _compute_vertex_normals_smooth(verts_np, faces_np)

    obj_eval.to_mesh_clear()
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh, do_unlink=True)

    return verts_np, faces_np, normals_np


def remesh_one(in_npz, out_npz, target_num_points):
    if os.path.exists(out_npz):
        print(f"[SKIP] {out_npz}")
        return

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    print(f"[INFO] {in_npz} -> {out_npz}")

    data = np.load(in_npz, allow_pickle=True)
    vertices = data["vertices"]  # (T, N, 3)
    faces = data["faces"]

    new_v, new_n = [], []
    for i in range(vertices.shape[0]):
        v_re, _, n_re = voxelize_mesh_numpy(vertices[i], faces)
        if v_re.shape[0] < target_num_points:
            idx = np.random.choice(v_re.shape[0], target_num_points, replace=True)
        else:
            idx = np.random.choice(v_re.shape[0], target_num_points, replace=False)
        new_v.append(v_re[idx][None])
        new_n.append(n_re[idx][None])

    np.savez(
        out_npz,
        vertices=np.concatenate(new_v, axis=0).astype(np.float16),
        normals=np.concatenate(new_n, axis=0).astype(np.float16),
    )


def parse_args():
    # When invoked as `blender -P script.py -- --input_root ...`, real args
    # come after the standalone `--`.
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = argv[1:]

    p = argparse.ArgumentParser()
    p.add_argument("--input_root", default="zoo/anim_meshes",
                   help="folder of per-character mesh-animation npz files")
    p.add_argument("--output_root", default="zoo/remesh_npz",
                   help="folder to write remeshed npz files into")
    p.add_argument("--target_num_points", type=int, default=20000)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    in_files = sorted(glob(os.path.join(args.input_root, "*.npz")))
    print(f"[INFO] Found {len(in_files)} npz under {args.input_root}")

    for in_npz in in_files:
        out_npz = os.path.join(args.output_root, os.path.basename(in_npz))
        try:
            remesh_one(in_npz, out_npz, args.target_num_points)
        except Exception as e:
            print(f"[ERROR] {in_npz}: {e}")

import argparse
import csv
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector


def _argv_after_separator():
    argv = sys.argv
    return [] if "--" not in argv else argv[argv.index("--") + 1 :]


def _import_obj(filepath):
    before = set(bpy.data.objects.keys())
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=filepath)
    else:
        bpy.ops.import_scene.obj(filepath=filepath)
    new_objects = [obj for obj in bpy.data.objects if obj.name not in before]
    if not new_objects:
        raise RuntimeError(f"No object imported from {filepath}")
    return new_objects[0]


def _set_render_engine(engine, samples, image_format, jpeg_quality):
    scene = bpy.context.scene
    try:
        scene.render.engine = engine
    except Exception:
        if engine == "BLENDER_EEVEE_NEXT":
            scene.render.engine = "BLENDER_EEVEE"
        else:
            raise

    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = image_format
    scene.render.image_settings.quality = jpeg_quality
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
    if hasattr(scene, "eevee_next"):
        scene.eevee_next.taa_render_samples = samples
        if hasattr(scene.eevee_next, "use_gtao"):
            scene.eevee_next.use_gtao = True


def _set_hdri_background(hdri_path):
    if not hdri_path or hdri_path == "transparent" or not os.path.exists(hdri_path):
        bpy.context.scene.render.film_transparent = True
        return

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    background = nodes.new(type="ShaderNodeBackground")
    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    output = nodes.new(type="ShaderNodeOutputWorld")
    env_tex.image = bpy.data.images.load(hdri_path, check_existing=True)
    links.new(env_tex.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    bpy.context.scene.render.film_transparent = False


def _smooth_if_no_normal_map(obj):
    has_normal_map = False
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "NORMAL_MAP":
                has_normal_map = True
                break
        if has_normal_map:
            break
    if not has_normal_map:
        for poly in obj.data.polygons:
            poly.use_smooth = True


def _load_vertices_from_obj(path, expected_count):
    vertices = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z, *_ = line.split()
                vertices.extend((float(x), float(y), float(z)))
    count = len(vertices) // 3
    if count != expected_count:
        raise RuntimeError(f"Vertex count mismatch for {path}: {count} != {expected_count}")
    return vertices


def _update_mesh_vertices_from_obj(target_obj, obj_path):
    mesh = target_obj.data
    coords = _load_vertices_from_obj(obj_path, len(mesh.vertices))
    mesh.vertices.foreach_set("co", coords)
    _smooth_if_no_normal_map(target_obj)
    mesh.update()


def _prepare_base_object(character_folder):
    base_obj_path = os.path.join(character_folder, "base_mesh.obj")
    if not os.path.exists(base_obj_path):
        raise FileNotFoundError(base_obj_path)

    base_obj = _import_obj(base_obj_path)
    base_obj.rotation_euler = (math.radians(90), 0, 0)
    if not base_obj.data.materials:
        return base_obj

    for mat in base_obj.data.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue

        tex_node = next(
            (node for node in nodes if node.type == "TEX_IMAGE" and node.image is not None),
            None,
        )
        if tex_node is not None and tex_node.image is not None:
            tex_name = os.path.basename(bpy.path.abspath(tex_node.image.filepath))
            tex_path = os.path.join(character_folder, tex_name)
            if os.path.exists(tex_path):
                tex_node.image = bpy.data.images.load(tex_path, check_existing=True)
                base_color = principled.inputs.get("Base Color")
                if base_color is not None:
                    for link in list(base_color.links):
                        links.remove(link)
                    links.new(tex_node.outputs["Color"], base_color)

        roughness = principled.inputs.get("Roughness")
        if roughness is not None:
            roughness.default_value = 0.7
        normal = principled.inputs.get("Normal")
        if normal is not None:
            for link in list(normal.links):
                links.remove(link)
        for node in list(nodes):
            if node.type in {"NORMAL_MAP", "BUMP"}:
                nodes.remove(node)

    return base_obj


def _read_jobs(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader)


def _frame_extension(image_format):
    return ".jpg" if image_format.upper() in {"JPEG", "JPG"} else ".png"


def _render_job(base_obj, job, image_format):
    mesh_folder = job["mesh_folder"]
    output_dir = job["output_dir"]
    camera_traj = np.load(job["camera_traj"])
    os.makedirs(output_dir, exist_ok=True)
    mesh_files = sorted(
        file_name
        for file_name in os.listdir(mesh_folder)
        if file_name.lower().endswith(".obj")
    )
    if len(mesh_files) != len(camera_traj):
        raise RuntimeError(
            f"Frame count mismatch for {mesh_folder}: {len(mesh_files)} meshes, "
            f"{len(camera_traj)} camera positions"
        )

    camera = bpy.data.objects.get("Camera")
    if camera is None:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
    bpy.context.scene.camera = camera
    ext = _frame_extension(image_format)

    for frame_idx, file_name in enumerate(mesh_files):
        _update_mesh_vertices_from_obj(base_obj, os.path.join(mesh_folder, file_name))
        camera.location = Vector(camera_traj[frame_idx])
        bpy.context.scene.render.filepath = os.path.join(output_dir, f"{frame_idx:05d}{ext}")
        bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser(description="Render many BVH mesh sequences in one Blender process.")
    parser.add_argument("--jobs-file", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--character-folder", required=True)
    parser.add_argument("--hdri", default="transparent")
    parser.add_argument("--engine", default=os.environ.get("BLENDER_RENDER_ENGINE", "BLENDER_EEVEE_NEXT"))
    parser.add_argument("--samples", type=int, default=int(os.environ.get("BLENDER_EEVEE_SAMPLES", "64")))
    parser.add_argument("--image-format", default=os.environ.get("BLENDER_IMAGE_FORMAT", "PNG"))
    parser.add_argument("--jpeg-quality", type=int, default=int(os.environ.get("BLENDER_JPEG_QUALITY", "95")))
    args = parser.parse_args(_argv_after_separator())

    bpy.ops.wm.open_mainfile(filepath=args.scene)
    _set_render_engine(args.engine, args.samples, args.image_format, args.jpeg_quality)
    _set_hdri_background(args.hdri)
    base_obj = _prepare_base_object(args.character_folder)

    jobs = _read_jobs(args.jobs_file)
    for index, job in enumerate(jobs, start=1):
        print(f"[BLENDER_BATCH] {index}/{len(jobs)} {job['output_dir']}", flush=True)
        _render_job(base_obj, job, args.image_format)


if __name__ == "__main__":
    main()

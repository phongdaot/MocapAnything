import argparse
import json
import math
import os

import numpy as np

import bpy
from mathutils import Vector

scene_material_combination = {
    'default':
        {
            'sbj': (0.904, 0.545, 0.695, 1),
            'obj': (0.802, 0.535, 0.186, 1)
        },
    'blank': {
        'sbj': (0.904, 0.545, 0.695, 1),
        'obj': (0.802, 0.535, 0.186, 1)
    },
    'grey_pink': {
        'sbj': (0.904, 0.545, 0.695, 1),
        'obj': (0.904, 0.545, 0.695, 1)
    },
    'grey_pink_lzy': {
        'sbj': (0.904, 0.545, 0.695, 1),
        'obj': (0.863, 0.0784, 0.235, 1)
    }
}

def lerp_color(color_start, color_end, t):
    r1, g1, b1 = color_start
    r2, g2, b2 = color_end
    r = (1 - t) * r1 + t * r2
    g = (1 - t) * g1 + t * g2
    b = (1 - t) * b1 + t * b2
    print(r,b,g)
    return (r, g, b)

def set_hdri_background(hdri_path: str):
    """
    Set HDRI image as the background environment of the current world.

    Args:
        hdri_path (str): Path to the .hdr or .exr image file.
    """
    if not os.path.exists(hdri_path):
        print('Use transparent background')
        return

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create new environment background nodes
    node_background = nodes.new(type='ShaderNodeBackground')
    node_env_tex = nodes.new(type='ShaderNodeTexEnvironment')
    node_output = nodes.new(type='ShaderNodeOutputWorld')

    node_env_tex.image = bpy.data.images.load(hdri_path, check_existing=True)

    links.new(node_env_tex.outputs['Color'], node_background.inputs['Color'])
    links.new(
        node_background.outputs['Background'], node_output.inputs['Surface']
    )
    bpy.context.scene.render.film_transparent = False


def fix_normals_if_needed(obj):
    """
    If the object does not use a normal map, apply smooth shading.

    Args:
        obj (Object): The Blender mesh object.
    """
    has_normal_map = False
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'NORMAL_MAP':
                for link in node.outputs[0].links:
                    target = link.to_node
                    if target.type == 'BSDF_PRINCIPLED':
                        has_normal_map = True
                        break
            if has_normal_map:
                break
        if has_normal_map:
            break

    if not has_normal_map:
        for poly in obj.data.polygons:
            poly.use_smooth = True


def update_mesh_vertices(target_obj, ref_obj):
    """
    Update the vertex positions of the target mesh to match the reference mesh.
    Smooth normals are applied if no normal map is used.

    Args:
        target_obj (Object): The object to be modified.
        ref_obj (Object): The object providing reference vertex positions.
    """
    target_mesh = target_obj.data
    ref_mesh = ref_obj.data
    if len(target_mesh.vertices) != len(ref_mesh.vertices):
        print(
            f"Vertex count mismatch: {len(target_mesh.vertices)} vs {len(ref_mesh.vertices)}"
        )
        return
    for i in range(len(target_mesh.vertices)):
        target_mesh.vertices[i].co = ref_mesh.vertices[i].co.copy()

    fix_normals_if_needed(target_obj)
    target_mesh.update()


if __name__ == '__main__':
    import sys
    argv = sys.argv

    if '--' not in argv:
        argv = []
    else:
        argv = argv[argv.index('--') + 1:]

    print('argsv:{0}'.format(argv))
    parser = argparse.ArgumentParser(
        description='Render Motion in 3D Environment.'
    )
    parser.add_argument(
        '--folder',
        type=str,
        metavar='PATH',
        help=
        'path to specific folder which include folders containing .obj files',
        default=''
    )
    parser.add_argument(
        '--out-folder',
        type=str,
        metavar='PATH',
        help='path to output folder which include rendered img files',
        default=''
    )
    parser.add_argument(
        '--use-mtl',
        action='store_true',
        help='If set, will preserve imported .obj materials from .mtl file.'
    )
    parser.add_argument(
        '--character-folder',
        type=str,
        metavar='PATH',
        help='path to character folder which include base mesh and mtl',
        default=''
    )
    # ------------------ Camera and Scene --------------------------
    parser.add_argument(
        '--scene',
        type=str,
        metavar='PATH',
        default='',
        help='path to specific .blend path for 3D scene'
    )

    parser.add_argument(
        '--hdri',
        type=str,
        default='',
        help=
        'Path to .hdr or .exr environment map file to be used as world background'
    )

    parser.add_argument(
        '--camera-traj',
        type=str,
        default='',
        help='Path to .npy file include camera trajectory'
    )

    args = parser.parse_args(argv)
    print('args:{0}'.format(args))

    # Load the world
    WORLD_FILE = args.scene
    bpy.ops.wm.open_mainfile(filepath=WORLD_FILE)
    bpy.context.scene.render.engine = 'CYCLES'

    # Render Optimizations
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.cycles.samples = 128

    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'
                                  ].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for d in bpy.context.preferences.addons['cycles'].preferences.devices:
        d['use'] = 1  # Using all devices, include GPU and CPU

    scene_name = args.scene.split('/')[-1].replace('.blend', '')
    print('scene name:{0}'.format(scene_name))
    if scene_name not in scene_material_combination:
        scene_name = 'default'
    material_info = scene_material_combination[scene_name]

    mesh_folder = args.folder
    output_dir = args.out_folder
    print('mesh_folder:{0}'.format(mesh_folder))
    print('output dir:{0}'.format(output_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare ply paths
    mesh_files = sorted(os.listdir(mesh_folder))

    camera_trans = Vector((0., -4., 1.25))
    if args.camera_traj:
        camera_traj = np.load(args.camera_traj)
    if args.hdri and os.path.exists(args.hdri):
        set_hdri_background(args.hdri)

    if not os.path.exists(os.path.join(args.character_folder, "base_mesh.mtl")):
        args.use_mtl = False

    if args.use_mtl:
        exist = set(bpy.data.objects.keys())
        base_obj_path = os.path.join(args.character_folder, "base_mesh.obj")
        bpy.ops.wm.obj_import(filepath=base_obj_path)
        new_objs = [obj for obj in bpy.data.objects if obj.name not in exist]
        base_obj = new_objs[0]
        base_obj.rotation_euler = (math.radians(90), 0, 0)

        if not base_obj.data.materials:
            args.use_mtl = False
        else:
            for mat in base_obj.data.materials:
                if mat is None:
                    continue

                mat.use_nodes = True
                nt = mat.node_tree
                nodes = nt.nodes
                links = nt.links

                principled = None
                for n in nodes:
                    if n.type == 'BSDF_PRINCIPLED':
                        principled = n
                        break

                if principled is None:
                    print(f"No Principled BSDF found in material {mat.name}")
                    continue

                tex_node = None
                for n in nodes:
                    if n.type == 'TEX_IMAGE' and n.image is not None:
                        tex_node = n
                        break

                # texture is optional
                if tex_node is not None and tex_node.image is not None:
                    tex_name = os.path.basename(bpy.path.abspath(tex_node.image.filepath))
                    tex_path = os.path.join(args.character_folder, tex_name)

                    if os.path.exists(tex_path):
                        img = bpy.data.images.load(tex_path, check_existing=True)
                        tex_node.image = img

                        base_color_input = principled.inputs['Base Color']
                        for link in list(base_color_input.links):
                            links.remove(link)
                        links.new(tex_node.outputs['Color'], base_color_input)
                    else:
                        print(f"Texture not found on disk: {tex_path}, keeping imported material values.")

                # common cleanup is still okay
                principled.inputs['Roughness'].default_value = 0.7

                normal_in = principled.inputs.get('Normal')
                if normal_in is not None:
                    for link in list(normal_in.links):
                        links.remove(link)

                for n in list(nodes):
                    if n.type in {'NORMAL_MAP', 'BUMP'}:
                        for out_sock in n.outputs:
                            for link in list(out_sock.links):
                                links.remove(link)
                        nodes.remove(n)
    for frame_idx in range(len(mesh_files)):
        # if frame_idx%5!=0 or frame_idx//5>5:continue
        obj_path = os.path.join(mesh_folder, mesh_files[frame_idx])
        file_name = obj_path.split('/')[-1]

        # Load object mesh and set material
        bpy.ops.wm.obj_import(filepath=obj_path)
        obj_object = bpy.data.objects[file_name[:-4]]
        # The default seems 90, 0, 0 while importing .obj into blender
        obj_object.rotation_euler = (math.radians(90), 0, 0)
        mesh = obj_object.data

        if args.use_mtl:
            update_mesh_vertices(base_obj, obj_object)
            bpy.data.objects.remove(obj_object, do_unlink=True)
        else:
            for f in mesh.polygons:
                f.use_smooth = True
            mat = bpy.data.materials.new(name='ObjMaterial')
            obj_object.data.materials.append(mat)
            mat.use_nodes = True
            principled_bsdf = mat.node_tree.nodes['Principled BSDF']
            if principled_bsdf is not None:
                t = frame_idx / (len(mesh_files)-1)
                r,g,b = lerp_color((0.8, 0.8, 0.8), material_info['obj'][:3], t)
                principled_bsdf.inputs[0].default_value = (r,g,b, 1)
            obj_object.active_material = mat

        # Set Camera position
        camera = bpy.data.objects['Camera']
        if args.camera_traj:
            camera_trans = Vector(camera_traj[frame_idx])
        camera.location = camera_trans
        bpy.context.scene.camera = camera
        print('camera_trans: ', camera_trans, camera.location)

        bpy.data.scenes['Scene'].render.filepath = os.path.join(
            output_dir, ('%05d' % frame_idx) + '.jpg'
        )
        bpy.ops.render.render(write_still=True)

        # Delet materials
        if not args.use_mtl:
            for block in bpy.data.materials:
                if block.users == 0:
                    bpy.data.materials.remove(block)

            bpy.data.objects.remove(obj_object, do_unlink=True)

    bpy.ops.wm.quit_blender()

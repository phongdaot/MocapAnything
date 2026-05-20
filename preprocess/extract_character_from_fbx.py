import os
import sys
import shutil
from glob import glob

import bpy
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.bvh_tools import face_forward_bvh_with_scale


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------
def _cleanup_scene():
    """Delete every object and purge orphan datablocks so the next import is clean."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                       bpy.data.armatures, bpy.data.actions):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _read_bvh_joint_names(bvh_path):
    """Parse a BVH file header and return joint names in declaration order
    (matches the column order of `anim.rotations`).
    """
    names = []
    with open(bvh_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('ROOT ') or stripped.startswith('JOINT '):
                names.append(stripped.split(None, 1)[1])
            elif stripped.startswith('MOTION'):
                break
    return names


# ---------------------------------------------------------------------------
# Character extraction (mesh + weights + textures, NO bvh)
# ---------------------------------------------------------------------------
def extract_character_data(fbx_file_path, save_path, bvh_joint_names=None):
    """Import an FBX and dump base_mesh.obj/.mtl, blender_vertices.npy,
    skinning_weights.npy, and texture images into `save_path`.

    If `bvh_joint_names` is given, the skinning_weights matrix has one column
    per BVH joint in that exact order (joints with no matching vertex group —
    e.g. Blender's synthetic root — get a zero column). Otherwise it falls
    back to using the armature's own bone list.
    """
    os.makedirs(save_path, exist_ok=True)
    bpy.ops.import_scene.fbx(filepath=fbx_file_path)

    mesh_objs = []
    armature_obj = None
    for obj_imported in bpy.context.selected_objects:
        if obj_imported.type == 'MESH':
            mesh_objs.append(obj_imported)
        elif obj_imported.type == 'ARMATURE':
            armature_obj = obj_imported

    if not mesh_objs:
        print(f"❌ No mesh objects found: {fbx_file_path}")
        _cleanup_scene()
        return False

    default_cube = bpy.data.objects.get('Cube')
    if default_cube:
        bpy.data.objects.remove(default_cube, do_unlink=True)

    # Merge meshes
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.join()
    merged_mesh = bpy.context.view_layer.objects.active

    # Triangulate
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"✅ Merged {len(mesh_objs)} meshes → {merged_mesh.name}, triangulated")

    # Vertices
    mesh_vertices = np.array([v.co[:] for v in merged_mesh.data.vertices])
    np.save(os.path.join(save_path, 'blender_vertices.npy'), mesh_vertices)

    # Skinning weights — aligned to BVH joint order when available
    if armature_obj:
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.pose.transforms_clear()
        bpy.ops.object.mode_set(mode='OBJECT')

        armature_bone_names = [b.name for b in armature_obj.data.bones]

        if bvh_joint_names is not None:
            joint_order = bvh_joint_names
            missing_in_bvh = [b for b in armature_bone_names if b not in joint_order]
            extra_in_bvh = [j for j in joint_order if j not in armature_bone_names]
            print(f"BVH joints ({len(joint_order)}): {joint_order}")
            if missing_in_bvh:
                print(f"⚠️ Bones with no BVH joint (weights dropped): {missing_in_bvh}")
            if extra_in_bvh:
                print(f"ℹ️ BVH joints with no vertex group (zero column): {extra_in_bvh}")
        else:
            joint_order = armature_bone_names
            print(f"Bone names ({len(joint_order)}): {joint_order}")

        # Map vertex-group name → column index in `joint_order`
        joint_index = {name: i for i, name in enumerate(joint_order)}

        skinning_weights = np.zeros((len(merged_mesh.data.vertices), len(joint_order)),
                                    dtype=np.float32)
        for v_i, vertex in enumerate(merged_mesh.data.vertices):
            for g in vertex.groups:
                group_name = merged_mesh.vertex_groups[g.group].name
                col = joint_index.get(group_name)
                if col is None:
                    # Vertex group has no matching joint (rare); silently skip
                    continue
                skinning_weights[v_i, col] = g.weight
            s = skinning_weights[v_i].sum()
            if s > 1e-8:
                skinning_weights[v_i] /= s

        np.save(os.path.join(save_path, "skinning_weights.npy"), skinning_weights)
        print(f"✅ Skinning weights saved, shape={skinning_weights.shape}")
    else:
        print(f"⚠️ No armature in {fbx_file_path}, skipping skinning weights")

    # Textures
    print("--- Extracting textures ---")
    extracted_textures = set()
    for material in merged_mesh.data.materials:
        if not material or not material.use_nodes or not material.node_tree:
            continue
        for node in material.node_tree.nodes:
            if node.type != 'TEX_IMAGE' or node.image is None:
                continue
            image = node.image
            if image.filepath and os.path.exists(bpy.path.abspath(image.filepath)):
                image_path = bpy.path.abspath(image.filepath)
                fname = os.path.basename(image_path)
                tgt_path = os.path.join(save_path, fname)
                if tgt_path not in extracted_textures:
                    shutil.copy(image_path, tgt_path)
                    print(f"📄 Copied texture: {fname}")
                    extracted_textures.add(tgt_path)
            elif image.has_data:
                fname = f"{image.name}.png"
                tgt_path = os.path.join(save_path, fname)
                if tgt_path not in extracted_textures:
                    image.filepath_raw = tgt_path
                    image.file_format = 'PNG'
                    image.save()
                    print(f"🖼️ Exported embedded texture: {fname}")
                    extracted_textures.add(tgt_path)

    # OBJ + MTL
    bpy.ops.object.select_all(action='DESELECT')
    merged_mesh.select_set(True)
    bpy.context.view_layer.objects.active = merged_mesh
    export_path = os.path.join(save_path, "base_mesh.obj")
    bpy.ops.wm.obj_export(
        filepath=export_path,
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
    )
    print(f"✅ Exported base_mesh.obj/.mtl to {save_path}")

    _cleanup_scene()
    return True


# ---------------------------------------------------------------------------
# BVH-only extraction (for animation FBXs and for the base rest pose)
# ---------------------------------------------------------------------------
def extract_bvh_only(fbx_file_path, save_path, bvh_filename=None):
    """Import an FBX, export ONLY its skeletal animation as a BVH, then clean up.
    Returns the full path to the written .bvh on success, or None on failure.
    """
    os.makedirs(save_path, exist_ok=True)
    bpy.ops.import_scene.fbx(filepath=fbx_file_path)

    armature_obj = next(
        (o for o in bpy.context.selected_objects if o.type == 'ARMATURE'),
        None,
    )

    default_cube = bpy.data.objects.get('Cube')
    if default_cube:
        bpy.data.objects.remove(default_cube, do_unlink=True)

    if armature_obj is None:
        print(f"⚠️ No armature in {fbx_file_path}, skipping BVH export")
        _cleanup_scene()
        return None

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    if armature_obj.animation_data and armature_obj.animation_data.action:
        start, end = armature_obj.animation_data.action.frame_range
        bpy.context.scene.frame_start = int(start)
        bpy.context.scene.frame_end = int(end)
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)

    if bvh_filename is None:
        bvh_filename = os.path.basename(fbx_file_path).replace('.fbx', '.bvh')
    bvh_path = os.path.join(save_path, bvh_filename)
    bpy.ops.export_anim.bvh(filepath=bvh_path, root_transform_only=True)
    print(f"✅ Exported BVH to {bvh_path}")

    _cleanup_scene()
    return bvh_path


# ---------------------------------------------------------------------------
# Pick the rest-pose / base FBX for a character
# ---------------------------------------------------------------------------
def find_base_fbx(fbx_files, character_name):
    """Heuristic: prefer `{Character}.fbx` or `{Character}ALL.fbx`,
    else any FBX without an animation-suffix dash (and not a TPOSE variant),
    else fall back to the shortest filename."""
    stems = {f: os.path.splitext(os.path.basename(f))[0] for f in fbx_files}

    for fbx, name in stems.items():
        if name == character_name or name == f"{character_name}ALL":
            return fbx
    for fbx, name in stems.items():
        if '-' not in name and 'TPOSE' not in name.upper():
            return fbx
    return min(fbx_files, key=lambda x: len(os.path.basename(x))) if fbx_files else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import utils.bvh as BVH  # kept for the optional face-forward step below

    data_root = 'Truebone_Z-OO'
    target_folder = 'zoo'
    os.makedirs(target_folder, exist_ok=True)

    character_folder = os.path.join(target_folder, 'characters_fix_facezplus')
    motions_folder = os.path.join(target_folder, 'motions')
    os.makedirs(character_folder, exist_ok=True)
    os.makedirs(motions_folder, exist_ok=True)

    skip_tpose = True

    character_dirs = sorted(
        d for d in glob(os.path.join(data_root, "*")) if os.path.isdir(d)
    )

    for char_dir in character_dirs:
        character_name = os.path.basename(char_dir)
        fbx_files = sorted(glob(os.path.join(char_dir, "*.fbx")))
        if not fbx_files:
            print(f"⚠️ No FBX files in {character_name}, skipping.")
            continue

        base_fbx = find_base_fbx(fbx_files, character_name)
        if base_fbx is None:
            print(f"⚠️ Could not identify base FBX for {character_name}, skipping.")
            continue
        animation_fbxs = [f for f in fbx_files if f != base_fbx]

        print(f"\n{'='*60}")
        print(f"Character : {character_name}")
        print(f"  Base FBX  : {os.path.basename(base_fbx)}")
        print(f"  Animations: {len(animation_fbxs)}")
        print(f"{'='*60}")

        char_save_dir = os.path.join(character_folder, character_name)
        os.makedirs(char_save_dir, exist_ok=True)

        # 1) Export the rest BVH from the base FBX → learn the BVH joint order.
        #    Blender adds a synthetic root joint when the armature has multiple
        #    top-level bones, so this list can be longer than the armature's
        #    own bone list. We use this to size/order the skinning matrix.
        rest_bvh_path = extract_bvh_only(
            base_fbx, char_save_dir, bvh_filename='rest.bvh'
        )
        bvh_joint_names = _read_bvh_joint_names(rest_bvh_path) if rest_bvh_path else None

        # 2) Mesh/weights/textures from the base FBX, weights aligned to BVH joints
        extract_character_data(base_fbx, char_save_dir, bvh_joint_names=bvh_joint_names)

        # 3) One BVH per animation FBX → zoo/motions/{name}#{anim_stem}.bvh
        for anim_fbx in animation_fbxs:
            try:
                anim_name = os.path.splitext(os.path.basename(anim_fbx))[0]
                if skip_tpose and 'TPOSE' in anim_name.upper():
                    print(f"⏭️ Skipping TPOSE: {anim_name}")
                    continue

                bvh_filename = f'{character_name}#{anim_name}.bvh'
                bvh_pth = extract_bvh_only(anim_fbx, motions_folder, bvh_filename=bvh_filename)
                if bvh_pth is None:
                    continue

                # --- Optional: face-forward + scale pass on the freshly exported BVH ---
                anim, name, ft = BVH.load(bvh_pth)
                print(anim.rotations.shape)
                face_forward_bvh_with_scale(bvh_pth, anim, name, ft, scale=0.01)
            except Exception as e:
                print(e)
        print(f"✅ Done with {character_name}")

    # Skip bpy's broken atexit cleanup so the pipeline sees a clean exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

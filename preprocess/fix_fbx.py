import argparse
import datetime
import json
import os
import re
import sys

import bpy
from mathutils import Vector


class SimpleLogger:
    def __init__(self, filepath, mode="a"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, mode, buffering=1)

    def log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"[{timestamp}] {message}\n")

    def error(self, message: str):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"[{timestamp}] ERROR: {message}\n")

    def close(self):
        self.file.close()


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inpath", required=True, help="input FBX path")
    parser.add_argument("--out", dest="outpath", required=True, help="output FBX path")
    parser.add_argument("--prefer", type=str, default="", help="comma-separated translation bones")
    parser.add_argument("--fixed-len", type=float, default=0.01, help="length for shortened leaf parents")
    parser.add_argument("--flagpole-threshold", type=float, default=10.0, help="max allowed scaled bone length")
    parser.add_argument("--log", type=str, default="", help="optional log file")
    parser.add_argument("--meta", type=str, default="", help="optional metadata JSON path")
    parser.add_argument("--expect-clean-signature", type=str, default="",
                        help="expected 'shortened,deleted' cleanup totals")
    parser.add_argument("--analysis-only", action="store_true", help="analyze but do not export")
    parser.add_argument("--skip-clean", action="store_true", help="skip no-weight leaf cleanup")
    parser.add_argument("--skip-root-push", action="store_true", help="skip root-motion push")
    return parser.parse_args(argv)


def safe_log_path(inpath):
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(inpath))[0])
    return os.path.join("logs", "fix_fbx", f"{stem}.log")


def cleanup_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                       bpy.data.armatures, bpy.data.actions):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def ensure_blender_fbx_import_compat():
    """Patch Blender 5.x FBX importer expectations for older FBX light fields."""
    import io_scene_fbx.import_fbx as import_fbx

    if getattr(import_fbx.blen_read_light, "_mocapanything_cast_shadow_patch", False):
        return

    globals_ = import_fbx.blen_read_light.__globals__

    def blen_read_light_compat(fbx_tmpl, fbx_obj, settings):
        import math

        elem_name_utf8 = globals_["elem_name_ensure_class"](fbx_obj, b"NodeAttribute")
        fbx_props = (
            globals_["elem_find_first"](fbx_obj, b"Properties70"),
            globals_["elem_find_first"](fbx_tmpl, b"Properties70", globals_["fbx_elem_nil"]),
        )
        light_type = {
            0: "POINT",
            1: "SUN",
            2: "SPOT",
        }.get(globals_["elem_props_get_enum"](fbx_props, b"LightType", 0), "POINT")

        lamp = bpy.data.lights.new(name=elem_name_utf8, type=light_type)

        if light_type == "SPOT":
            spot_size = globals_["elem_props_get_number"](fbx_props, b"OuterAngle", None)
            if spot_size is None:
                spot_size = globals_["elem_props_get_number"](fbx_props, b"Cone angle", 45.0)
            lamp.spot_size = math.radians(spot_size)

            spot_blend = globals_["elem_props_get_number"](fbx_props, b"InnerAngle", None)
            if spot_blend is None:
                spot_blend = globals_["elem_props_get_number"](fbx_props, b"HotSpot", 45.0)
            lamp.spot_blend = 1.0 - (spot_blend / spot_size)

        lamp.color = globals_["elem_props_get_color_rgb"](fbx_props, b"Color", (1.0, 1.0, 1.0))
        lamp.energy = globals_["elem_props_get_number"](fbx_props, b"Intensity", 100.0) / 100.0
        lamp.exposure = globals_["elem_props_get_number"](fbx_props, b"Exposure", 0.0)
        lamp.use_shadow = globals_["elem_props_get_bool"](fbx_props, b"CastShadow", True)
        if hasattr(lamp, "cycles") and hasattr(lamp.cycles, "cast_shadow"):
            lamp.cycles.cast_shadow = lamp.use_shadow

        if settings.use_custom_props:
            globals_["blen_read_custom_properties"](fbx_obj, lamp, settings)

        return lamp

    blen_read_light_compat._mocapanything_cast_shadow_patch = True
    import_fbx.blen_read_light = blen_read_light_compat


def get_skinned_meshes(arm_obj):
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for modifier in obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == arm_obj:
                meshes.append(obj)
                break
    return meshes


def count_skinned_meshes(arm_obj):
    return len(get_skinned_meshes(arm_obj))


def import_main_armature(inpath):
    ensure_blender_fbx_import_compat()
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=inpath, automatic_bone_orientation=False)
    after = [obj for obj in bpy.data.objects if obj not in before]
    candidates = [obj for obj in after if obj.type == "ARMATURE"] or [
        obj for obj in bpy.data.objects if obj.type == "ARMATURE"
    ]
    if not candidates:
        raise RuntimeError("No armature found")
    arm = max(candidates, key=count_skinned_meshes)
    bpy.context.view_layer.objects.active = arm
    return arm


def has_weight(meshes, bone_name, min_weight=1e-6):
    for mesh in meshes:
        group = mesh.vertex_groups.get(bone_name)
        if not group:
            continue
        group_idx = group.index
        for vertex in mesh.data.vertices:
            for weight in vertex.groups:
                if weight.group == group_idx and weight.weight > min_weight:
                    return True
    return False


def clean_and_shorten_by_weights(arm, logger, fixed_len=0.01):
    """Shorten parents of no-weight leaf bones, then delete those no-weight leaves."""
    meshes = get_skinned_meshes(arm)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")

    leaf_names = []
    parent_to_leaf = {}
    for bone in arm.data.edit_bones:
        if len(bone.children) == 0:
            leaf_names.append(bone.name)
            if bone.parent and len(bone.parent.children) == 1:
                parent_to_leaf[bone.parent.name] = bone.name

    bpy.ops.object.mode_set(mode="OBJECT")
    leaf_no_weight = {name for name in leaf_names if not has_weight(meshes, name)}
    parents_to_shorten = [
        parent for parent, leaf in parent_to_leaf.items() if leaf in leaf_no_weight
    ]

    bpy.ops.object.mode_set(mode="EDIT")
    shortened = 0
    for parent_name in parents_to_shorten:
        parent = arm.data.edit_bones.get(parent_name)
        if not parent:
            continue
        if len(parent.children) == 1 and len(parent.children[0].children) == 0:
            direction = parent.y_axis.normalized() if parent.y_axis.length > 0 else Vector((0, 1, 0))
            parent.tail = parent.head + direction * float(fixed_len)
            shortened += 1

    bpy.ops.object.mode_set(mode="OBJECT")
    for mesh in meshes:
        for bone_name in leaf_no_weight:
            group = mesh.vertex_groups.get(bone_name)
            if group:
                mesh.vertex_groups.remove(group)

    bpy.ops.object.mode_set(mode="EDIT")
    deleted = 0
    for bone_name in list(leaf_no_weight):
        bone = arm.data.edit_bones.get(bone_name)
        if bone:
            arm.data.edit_bones.remove(bone)
            deleted += 1

    bpy.ops.object.mode_set(mode="OBJECT")
    logger.log(f"Shortened {shortened} leaf parents; deleted {deleted} no-weight leaves")
    return shortened, deleted


def clean_leaf_bones_until_stable(arm, logger, fixed_len=0.01):
    total_shortened = 0
    total_deleted = 0
    while True:
        shortened, deleted = clean_and_shorten_by_weights(arm, logger, fixed_len=fixed_len)
        total_shortened += shortened
        total_deleted += deleted
        if deleted == 0:
            break
    logger.log(f"Cleanup totals: shortened={total_shortened}, deleted={total_deleted}")
    return total_shortened, total_deleted


def has_flagpole_bone(arm, len_threshold=10.0, topk_log=3, logger=None):
    scale = max(abs(arm.scale.x), abs(arm.scale.y), abs(arm.scale.z))
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")

    lengths = []
    names = []
    for bone in arm.data.edit_bones:
        lengths.append((bone.tail - bone.head).length * scale)
        names.append(bone.name)

    bpy.ops.object.mode_set(mode="OBJECT")
    if not lengths:
        return False, ("", 0.0)

    max_idx = max(range(len(lengths)), key=lambda i: lengths[i])
    if logger and topk_log:
        for idx in sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)[:topk_log]:
            logger.log(f"[DBG] bone_len rank {idx}: {names[idx]} len={lengths[idx]:.3f}")
    return lengths[max_idx] > len_threshold, (names[max_idx], lengths[max_idx])


def has_keyframe(action, bone_name, data_path, frame):
    target_path = f'pose.bones["{bone_name}"].{data_path}'
    for curve in action.fcurves:
        if curve.data_path != target_path:
            continue
        for keyframe in curve.keyframe_points:
            if int(round(keyframe.co.x)) == frame:
                return True
    return False


def measure_bone_loc_motion(arm, logger, start=None, end=None):
    action = arm.animation_data.action if (arm.animation_data and arm.animation_data.action) else None
    if not action:
        raise RuntimeError("Armature has no available action")
    frame_start, frame_end = map(int, action.frame_range)
    start = frame_start if start is None else start
    end = frame_end if end is None else end
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = start, end
    result = {}
    bpy.context.scene.frame_set(start)
    base = {pb.name: pb.location.copy() for pb in arm.pose.bones}
    for frame in range(start, end + 1):
        bpy.context.scene.frame_set(frame)
        for pose_bone in arm.pose.bones:
            delta = (pose_bone.location - base[pose_bone.name]).length
            result[pose_bone.name] = max(result.get(pose_bone.name, 0.0), delta)
    logger.log(f"Measured location motion on frames {start}-{end}")
    return start, end, result


def bones_with_skin_weights(arm, min_weight=1e-6):
    weighted = set()
    meshes = get_skinned_meshes(arm)
    deform_bone_names = {bone.name for bone in arm.data.bones if bone.use_deform}
    for mesh in meshes:
        idx_to_name = {group.index: group.name for group in mesh.vertex_groups}
        for vertex in mesh.data.vertices:
            for group in vertex.groups:
                if group.weight > min_weight:
                    name = idx_to_name.get(group.group)
                    if name in deform_bone_names:
                        weighted.add(name)
        if len(weighted) == len(deform_bone_names):
            break
    return weighted


def bone_depth(arm, bone_name):
    bone = arm.data.bones.get(bone_name)
    if bone is None:
        return 10**9
    depth = 0
    while bone.parent is not None:
        depth += 1
        bone = bone.parent
    return depth


def build_parent_children_maps(arm):
    parent = {}
    children = {}
    for bone in arm.data.bones:
        parent[bone.name] = bone.parent.name if bone.parent else None
        children.setdefault(bone.name, [])
    for bone in arm.data.bones:
        if bone.parent:
            children[bone.parent.name].append(bone.name)
    return parent, children


def path_root_to(parent_map, bone_name):
    chain = []
    current = bone_name
    while current is not None:
        chain.append(current)
        current = parent_map.get(current)
    chain.reverse()
    return chain


def subtree_has_weighted(start_name, children, weighted_set):
    stack = [start_name]
    while stack:
        node = stack.pop()
        if node in weighted_set:
            return True
        stack.extend(children.get(node, []))
    return False


def is_trunk_bone_allow_weight(bone_name, parent_map, children, weighted_set):
    path = path_root_to(parent_map, bone_name)
    path_set = set(path)
    for idx in range(len(path) - 1):
        ancestor = path[idx]
        next_on_path = path[idx + 1]
        for child in children.get(ancestor, []):
            if child == next_on_path or child in path_set:
                continue
            if subtree_has_weighted(child, children, weighted_set):
                return False
    return True


def pick_translation_bones(metrics, arm, logger, rel_ratio=0.02, abs_eps=0.05,
                           depth_limit=2, exclude_weighted=True, min_weight=1e-6):
    if not metrics:
        return None, set()
    max_all = max(max(metrics.values()), 1e-8)
    base_keep = {
        name for name, value in metrics.items()
        if value >= max_all * rel_ratio and value >= abs_eps
    } or set(metrics.keys())

    eligible = {name for name in base_keep if bone_depth(arm, name) <= depth_limit}
    if not eligible:
        logger.log("No eligible translation bones after depth filter")
        return "NA", set()

    if exclude_weighted:
        weighted = bones_with_skin_weights(arm, min_weight=min_weight)
        parent_map, children = build_parent_children_maps(arm)
        eligible = {
            name for name in eligible
            if name not in weighted or is_trunk_bone_allow_weight(name, parent_map, children, weighted)
        }
        logger.log(f"Eligible translation bones after skin/trunk filter: {sorted(eligible)}")

    if not eligible:
        logger.log("No eligible translation bones after skin weights filter")
        return "NA", set()

    main = max(eligible, key=lambda name: metrics.get(name, 0.0))
    keep = {name for name in eligible if name in base_keep}
    keep.add(main)
    return main, keep


def ensure_single_root(arm, root_name="Root", align_to_bone=None):
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = arm.data.edit_bones
    roots = [bone for bone in edit_bones if bone.parent is None]
    if len(roots) <= 1:
        out = roots[0].name if roots else None
        bpy.ops.object.mode_set(mode="POSE")
        return out

    root = edit_bones.new(root_name)
    if align_to_bone is not None and align_to_bone in edit_bones:
        bone = edit_bones[align_to_bone]
        root.head = bone.head.copy()
        root.tail = (bone.head + Vector((0, 1, 0))).copy()
    else:
        root.head = Vector((0, 0, 0))
        root.tail = Vector((0, 1, 0))

    for bone in roots:
        if bone != root:
            bone.parent = root
    bpy.ops.object.mode_set(mode="POSE")
    arm.data.bones[root_name].use_deform = False
    return root_name


def head_in_armature_space(pose_bone):
    return (pose_bone.matrix @ Vector((0, 0, 0, 1))).to_3d()


def select_translation_bones(arm, logger, prefer_bones=None):
    frame_start, frame_end, metrics = measure_bone_loc_motion(arm, logger)
    main_bone, keep_set = pick_translation_bones(metrics, arm, logger)
    if prefer_bones:
        main_bone = prefer_bones
    if isinstance(main_bone, str):
        main_bone = [main_bone]
    return frame_start, frame_end, main_bone, keep_set


def push_translation_to_root(arm, logger, prefer_bones=None):
    frame_start, frame_end, main_bone, keep_set = select_translation_bones(
        arm, logger, prefer_bones=prefer_bones
    )
    if not main_bone or "NA" in main_bone:
        logger.log("No translation bone selected; skip root push")
        return main_bone

    logger.log(f"locator = {main_bone}, frames: {frame_start}-{frame_end}")
    root_name = ensure_single_root(arm, "Root", align_to_bone=main_bone[0])
    if root_name is None:
        pose_bone = arm.pose.bones[main_bone[0]]
        while pose_bone.parent:
            pose_bone = pose_bone.parent
        root_name = pose_bone.name

    if prefer_bones:
        push_flag = not (root_name == main_bone[0] and len(main_bone) == 1)
    else:
        push_flag = root_name not in keep_set
    if not push_flag:
        logger.log(f"Root {root_name} already has translation or selected locator; skip root push")
        return main_bone

    logger.log(f"Current root is {root_name}; pushing {main_bone} translation to root")
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    root_edit_bone = arm.data.edit_bones[root_name]
    root_edit_bone.tail = (root_edit_bone.head + Vector((0.0, 1.0, 0.0))).copy()
    main_edit_bone = arm.data.edit_bones[main_bone[0]]
    delta_static = (root_edit_bone.head - main_edit_bone.head).copy()
    bpy.ops.object.mode_set(mode="POSE")

    original_matrix_world = arm.matrix_world.copy()
    arm.location = (0, 0, 0)
    arm.rotation_euler = (0, 0, 0)
    arm.scale = (1, 1, 1)

    root_pose_bone = arm.pose.bones[root_name]
    locator = arm.pose.bones[main_bone[0]]
    loc_record = []
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        loc_record.append(head_in_armature_space(locator) - head_in_armature_space(root_pose_bone))

    if len(main_bone) > 1:
        for loc_name in main_bone[1:]:
            sub_locator = arm.pose.bones[loc_name]
            bpy.ops.object.mode_set(mode="EDIT")
            loc_edit_bone = arm.data.edit_bones[loc_name]
            loc_edit_head = loc_edit_bone.head.copy()
            bpy.ops.object.mode_set(mode="POSE")
            for frame in range(frame_start, frame_end + 1):
                bpy.context.scene.frame_set(frame)
                if sub_locator.location == Vector((0, 0, 0)) and not has_keyframe(
                    arm.animation_data.action, loc_name, "location", frame
                ):
                    bpy.context.scene.frame_set(frame - 1)
                    prev_location = sub_locator.location.copy()
                    bpy.context.scene.frame_set(frame)
                    sub_locator.location = prev_location
                    sub_locator.keyframe_insert(data_path="location", frame=frame)
                loc_record[frame - frame_start] += head_in_armature_space(sub_locator) - loc_edit_head

            for frame in range(frame_start, frame_end + 1):
                bpy.context.scene.frame_set(frame)
                sub_locator.location = Vector((0, 0, 0))
                sub_locator.keyframe_insert(data_path="location", frame=frame)

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        root_pose_bone.location = loc_record[frame - frame_start] + delta_static
        root_pose_bone.keyframe_insert(data_path="location", frame=frame)
        locator.location = Vector((0, 0, 0))
        locator.keyframe_insert(data_path="location", frame=frame)
    arm.matrix_world = original_matrix_world
    return main_bone


def export_fbx_selection(filepath, objects, logger, bake_anim=True):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        add_leaf_bones=False,
        bake_space_transform=True,
        apply_unit_scale=True,
        path_mode="COPY",
        embed_textures=True,
        bake_anim=bake_anim,
        bake_anim_use_all_bones=True,
        bake_anim_force_startend_keying=True,
        armature_nodetype="NULL",
        axis_forward="-Z",
        axis_up="Y",
    )
    logger.log(f"[OK] FBX exported -> {filepath}")


def main():
    args = parse_args()
    prefer_bones = [item for item in args.prefer.split(",") if item]
    logger = SimpleLogger(args.log or safe_log_path(args.inpath), mode="w")
    logger.log(f"Fixing {args.inpath} -> {args.outpath}")
    meta = {
        "input": args.inpath,
        "output": args.outpath,
        "success": False,
        "analysis_only": bool(args.analysis_only),
        "prefer": prefer_bones,
        "locator": [],
        "cleanup": {"shortened": 0, "deleted": 0},
        "error": "",
    }

    try:
        cleanup_scene()
        arm = import_main_armature(args.inpath)
        if not args.skip_clean:
            shortened, deleted = clean_leaf_bones_until_stable(arm, logger, fixed_len=args.fixed_len)
            meta["cleanup"] = {"shortened": shortened, "deleted": deleted}
        if args.expect_clean_signature:
            expected_shortened, expected_deleted = [
                int(x.strip()) for x in args.expect_clean_signature.split(",", 1)
            ]
            actual = meta["cleanup"]
            if (actual["shortened"], actual["deleted"]) != (expected_shortened, expected_deleted):
                raise RuntimeError(
                    "Cleanup signature mismatch: "
                    f"actual={actual['shortened']},{actual['deleted']} "
                    f"expected={expected_shortened},{expected_deleted}"
                )
        bad, info = has_flagpole_bone(
            arm,
            len_threshold=args.flagpole_threshold,
            logger=logger,
        )
        if bad:
            bone_name, bone_len = info
            raise RuntimeError(f"Flagpole bone detected: {bone_name}, len={bone_len:.3f}")
        if args.skip_root_push:
            _, _, locator, _ = select_translation_bones(arm, logger, prefer_bones=prefer_bones)
        else:
            locator = push_translation_to_root(arm, logger, prefer_bones=prefer_bones)
        meta["locator"] = [x for x in (locator or []) if x and x != "NA"]
        if not args.analysis_only:
            meshes = get_skinned_meshes(arm)
            export_fbx_selection(args.outpath, [arm] + meshes, logger)
        meta["success"] = True
    except Exception as exc:
        meta["error"] = str(exc)
        logger.error(str(exc))
        raise
    finally:
        if args.meta:
            os.makedirs(os.path.dirname(args.meta), exist_ok=True)
            with open(args.meta, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.close()


if __name__ == "__main__":
    main()

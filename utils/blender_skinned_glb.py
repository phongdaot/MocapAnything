### blender_skinned_glb.py ###
"""
在 Blender 内把「预测 BVH + 角色 mesh + skinning 权重」导出为真正的骨骼蒙皮 glTF(.glb),
供 Web 端(gr.Model3D / model-viewer)交互式播放 —— 替代 morph-target 方案
(three.js 对大量 morph target 会全部叠加导致 mesh 爆炸)。

用法(由 utils/glb_export.py 的 wrapper 调用):
  blender -b -P utils/blender_skinned_glb.py -- \
      --bvh pred.bvh --data pack.npz --out-mesh mesh.glb --out-skel skel.glb [--validate]

pack.npz 由 Python 侧预生成(绕开 obj 导入器轴向歧义):
  verts   (V,3)  rest mesh 顶点(BVH 同一坐标系,即 LBS 的 v_template)
  faces   (F,3)  三角面
  weights (V,J)  LBS 权重(列序 = BVH 关节序)
  names   (J,)   BVH 关节名(str)
  rest_joints (J,3) rest 骨架关节位置(compute_rest_joints,BVH 坐标系)
  lbs_check   (K,V,3) 可选:LBS 的若干帧顶点(验证用)
  lbs_check_frames (K,) 对应帧号
"""
import bpy
import sys
import numpy as np


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    out = {"validate": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--validate":
            out["validate"] = True; i += 1
        else:
            out[a.lstrip("-").replace("-", "_")] = argv[i + 1]; i += 2
    return out


def kabsch(P, Q):
    """求 R,t 使 R@P+t ≈ Q(P,Q: N×3)。返回 4×4 矩阵。"""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = Qc - R @ Pc
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def main():
    args = parse_args()
    pack = np.load(args["data"], allow_pickle=True)
    verts = pack["verts"].astype(np.float64)
    faces = pack["faces"].astype(np.int64)
    weights = pack["weights"].astype(np.float64)
    names = [str(n) for n in pack["names"]]
    rest_joints = pack["rest_joints"].astype(np.float64)

    # 清场
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 1) 导入 BVH → 带动画的 armature
    bpy.ops.import_anim.bvh(filepath=args["bvh"], frame_start=1,
                            use_fps_scale=False, update_scene_fps=True,
                            use_cyclic=False, rotate_mode="NATIVE")
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    Mw = np.array(arm.matrix_world)

    # 2) blender 里的 rest 关节位置(世界系),按 names 序取
    heads = []
    for nm in names:
        b = arm.data.bones.get(nm)
        if b is None:
            raise RuntimeError(f"bone {nm} not found in imported BVH armature")
        h = Mw @ np.array([*b.head_local, 1.0])
        heads.append(h[:3])
    heads = np.array(heads)

    # 3) Kabsch:把「LBS/BVH 坐标系」对齐到「blender armature 世界系」
    M_align = kabsch(rest_joints, heads)
    resid = np.abs((M_align[:3, :3] @ rest_joints.T).T + M_align[:3, 3] - heads).max()
    print(f"[skinned_glb] rest 对齐残差(max abs) = {resid:.6f}")

    # 4) 建 mesh(from_pydata,原始 BVH 系坐标)+ 对齐矩阵
    me = bpy.data.meshes.new("charmesh")
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()
    ob = bpy.data.objects.new("char", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.matrix_world = M_align.T.tolist() if False else [list(r) for r in M_align]

    # 平滑着色 + 简单材质(web 端好看些)
    for p in me.polygons:
        p.use_smooth = True
    mat = bpy.data.materials.new("charmat"); mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.78, 0.9, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    me.materials.append(mat)

    # 5) 顶点组(按关节名)+ armature modifier
    idx_by_joint = [np.nonzero(weights[:, j] > 1e-6)[0] for j in range(len(names))]
    for j, nm in enumerate(names):
        vg = ob.vertex_groups.new(name=nm)
        for vid in idx_by_joint[j]:
            vg.add([int(vid)], float(weights[vid, j]), "REPLACE")
    mod = ob.modifiers.new("arm", "ARMATURE")
    mod.object = arm
    ob.parent = arm

    # 场景帧范围 = 动画范围
    act = arm.animation_data.action
    f0, f1 = (int(act.frame_range[0]), int(act.frame_range[1])) if act else (1, 1)
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = f0, f1
    print(f"[skinned_glb] 动画帧范围 {f0}..{f1}")

    # 6) 数值验证:blender 蒙皮结果 vs LBS 顶点(旋转不变残差,Kabsch 后 RMS)
    if args["validate"] and "lbs_check" in pack:
        dg = bpy.context.evaluated_depsgraph_get()
        for k, fr in enumerate(pack["lbs_check_frames"]):
            bpy.context.scene.frame_set(int(fr) + f0)   # LBS 帧0 == blender 帧 f0
            dg.update()
            # 6a) 关节 FK 对比(定位:FK 错 vs 蒙皮错)
            if "lbs_joints_check" in pack:
                MwP = np.array(arm.matrix_world)
                Jb = np.array([ (MwP @ np.array([*arm.pose.bones[nm].head, 1.0]))[:3] for nm in names ])
                Jl = pack["lbs_joints_check"][k].astype(np.float64)
                Mj = kabsch(Jl, Jb)
                jrms = float(np.sqrt((((Mj[:3,:3] @ Jl.T).T + Mj[:3,3] - Jb) ** 2).sum(-1)).mean())
                print(f"[validate] frame {int(fr)}: 关节FK Kabsch-RMS={jrms:.6f}")
            ev = ob.evaluated_get(dg)
            evme = ev.to_mesh()
            Vb = np.array([ (np.array(ev.matrix_world) @ np.array([*v.co, 1.0]))[:3] for v in evme.vertices ])
            ev.to_mesh_clear()
            Vl = pack["lbs_check"][k].astype(np.float64)
            Mk = kabsch(Vl, Vb)
            rec = (Mk[:3, :3] @ Vl.T).T + Mk[:3, 3]
            rms = float(np.sqrt(((rec - Vb) ** 2).sum(-1)).mean())
            span = float(np.abs(Vl - Vl.mean(0)).max())
            print(f"[validate] frame {int(fr)}: Kabsch-RMS={rms:.6f} (span={span:.3f}, 比例={rms/max(span,1e-9):.4f})")

    # 7) 导出 mesh glb(armature + 蒙皮 mesh)
    for o in bpy.data.objects:
        o.select_set(False)
    arm.select_set(True); ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=args["out_mesh"], export_format="GLB",
                              use_selection=True, export_animations=True,
                              export_skins=True, export_yup=True)
    print(f"[skinned_glb] mesh glb -> {args['out_mesh']}")

    # 8) 骨架可视化:每根骨一个细长盒,骨骼父子(bone parent)→ 刚性跟随骨骼动画
    span = float(np.abs(rest_joints - rest_joints.mean(0)).max())
    r = max(span * 0.012, 1e-4)
    viz = []
    for nm in names:
        b = arm.data.bones.get(nm)
        L = float(b.length)
        if L <= 1e-8:
            continue
        me2 = bpy.data.meshes.new(f"bone_{nm}")
        # 盒:沿 -Y 方向(bone-parent 的原点在骨尾,骨沿 +Y,故 [-L,0])
        hx = r
        vs = [(-hx, -L, -hx), (hx, -L, -hx), (hx, -L, hx), (-hx, -L, hx),
              (-hx, 0, -hx), (hx, 0, -hx), (hx, 0, hx), (-hx, 0, hx)]
        fs = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        me2.from_pydata(vs, [], fs); me2.update()
        ob2 = bpy.data.objects.new(f"viz_{nm}", me2)
        bpy.context.scene.collection.objects.link(ob2)
        ob2.parent = arm; ob2.parent_type = "BONE"; ob2.parent_bone = nm
        viz.append(ob2)
    for o in bpy.data.objects:
        o.select_set(False)
    arm.select_set(True)
    for o in viz:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=args["out_skel"], export_format="GLB",
                              use_selection=True, export_animations=True, export_yup=True)
    print(f"[skinned_glb] skeleton glb -> {args['out_skel']}")
    print("[skinned_glb] DONE")


if __name__ == "__main__":
    main()

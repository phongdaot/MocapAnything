### glb_export.py ###
"""
把一段 mesh 顶点动画(逐帧 verts,拓扑固定)导出成带 morph-target 动画的 .glb,
供 Web 端交互式 3D 查看器(gr.Model3D / model-viewer)旋转 + 播放。

morph-target 方案:frame0 作 base;frame i(i>=1)作 morph target(delta = verts[i]-verts[0]);
动画在每个时间点把对应 target 的权重设 1、其余 0,model-viewer 在关键帧间插值 → 平滑动画。
"""
import os
import numpy as np
import pygltflib
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView, Accessor, Animation, AnimationChannel, AnimationSampler


def mesh_sequence_to_glb(verts, faces, out_path, fps=15, max_frames=120):
    """
    verts: (F, V, 3) float  逐帧顶点
    faces: (T, 3) int        三角面(固定拓扑)
    out_path: 输出 .glb
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    F, V, _ = verts.shape

    # 帧太多则均匀抽稀(morph target 数 = 帧数,控制体积)
    if F > max_frames:
        idx = np.linspace(0, F - 1, max_frames).round().astype(int)
        verts = verts[idx]
        F = verts.shape[0]

    base = verts[0]                          # (V,3)
    targets = verts[1:] - base[None]         # (F-1, V, 3) morph deltas
    n_targets = targets.shape[0]

    # 居中 + 缩放到单位左右,方便查看器取景
    center = base.mean(axis=0)
    base_c = base - center
    scale = float(np.abs(base_c).max()) or 1.0
    base_c = base_c / scale
    targets = targets / scale

    # ---- 拼 binary buffer ----
    blobs = []
    def add(arr, comp_type, acc_type, target=None, minmax=False):
        a = np.ascontiguousarray(arr)
        raw = a.tobytes()
        off = sum(len(b["padded"]) for b in blobs)   # 前面所有(含 pad)的总长 = 本 view 偏移
        pad = (-len(raw)) % 4                          # 4 字节对齐
        blobs.append({"padded": raw + b"\x00" * pad, "raw_len": len(raw),
                      "comp": comp_type, "acc_type": acc_type, "off": off,
                      "arr": a, "target": target, "minmax": minmax})
        return len(blobs) - 1

    # indices
    ii = add(faces.reshape(-1).astype(np.uint32), pygltflib.UNSIGNED_INT, "SCALAR", target=pygltflib.ELEMENT_ARRAY_BUFFER)
    # base positions
    ip = add(base_c.astype(np.float32), pygltflib.FLOAT, "VEC3", target=pygltflib.ARRAY_BUFFER, minmax=True)
    # morph targets (positions delta)
    target_pos_idx = [add(t.astype(np.float32), pygltflib.FLOAT, "VEC3", target=pygltflib.ARRAY_BUFFER) for t in targets]
    # animation input (times) + output (weights)
    times = (np.arange(F, dtype=np.float32) / float(fps))
    it = add(times, pygltflib.FLOAT, "SCALAR", minmax=True)
    # weights: F 帧 × n_targets;frame k → target(k-1)=1(k>=1),frame0 全 0
    W = np.zeros((F, n_targets), dtype=np.float32)
    for k in range(1, F):
        W[k, k - 1] = 1.0
    iw = add(W.reshape(-1), pygltflib.FLOAT, "SCALAR")

    # ---- 组装 gltf 对象 ----
    bin_blob = b"".join(b["padded"] for b in blobs)
    bufferViews, accessors = [], []
    for k, b in enumerate(blobs):
        a, acc_type = b["arr"], b["acc_type"]
        # byteLength = 未 pad 的真实数据长度(pad 只影响下一 view 的偏移)
        bufferViews.append(BufferView(buffer=0, byteOffset=b["off"], byteLength=b["raw_len"], target=b["target"]))
        count = a.shape[0] if (acc_type == "VEC3") else a.size
        acc = Accessor(bufferView=k, componentType=b["comp"], count=count, type=acc_type, byteOffset=0)
        if b["minmax"]:
            flat = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(-1, 1)
            acc.min = flat.min(axis=0).tolist()
            acc.max = flat.max(axis=0).tolist()
        accessors.append(acc)

    prim = Primitive(
        attributes=Attributes(POSITION=ip),
        indices=ii,
        targets=[{"POSITION": ti} for ti in target_pos_idx],
    )
    mesh = Mesh(primitives=[prim], weights=[0.0] * n_targets)
    node = Node(mesh=0)
    anim = Animation(
        samplers=[AnimationSampler(input=it, output=iw, interpolation="LINEAR")],
        channels=[AnimationChannel(sampler=0, target={"node": 0, "path": "weights"})],
    )

    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[node],
        meshes=[mesh],
        animations=[anim],
        buffers=[Buffer(byteLength=len(bin_blob))],
        bufferViews=bufferViews,
        accessors=accessors,
    )
    gltf.set_binary_blob(bin_blob)
    gltf.save(out_path)
    return out_path


# 单个盒子(8 角)的三角面(相对 8 顶点的索引);顶点顺序见 _bone_box
_BOX_FACES = np.array([
    [0, 1, 3], [0, 3, 2],   # end a
    [4, 6, 7], [4, 7, 5],   # end b
    [0, 2, 6], [0, 6, 4],   # side -v
    [1, 5, 7], [1, 7, 3],   # side +v
    [0, 4, 5], [0, 5, 1],   # side -u
    [2, 3, 7], [2, 7, 6],   # side +u
], dtype=np.uint32)


def _bone_box(a, b, r):
    """从 a 到 b 的细长方盒 8 顶点(截面半宽 r)。"""
    axis = b - a
    L = np.linalg.norm(axis)
    axis = axis / L if L > 1e-8 else np.array([0.0, 0.0, 1.0])
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(axis, ref); u = u / (np.linalg.norm(u) + 1e-9)
    v = np.cross(axis, u)
    out = []
    for end in (a, b):
        for su in (-1, 1):
            for sv in (-1, 1):
                out.append(end + su * r * u + sv * r * v)
    return np.array(out, dtype=np.float32)   # (8,3)


def skeleton_sequence_to_glb(joints, parents, out_path, fps=15, thickness=0.02, max_frames=120):
    """
    joints: (F, J, 3)  逐帧关节位置
    parents: (J,)      每关节父节点(-1=根)
    把骨架(每根骨=细盒)导成带 morph 动画的 glb,供交互式 3D 查看。
    """
    joints = np.asarray(joints, dtype=np.float32)
    parents = np.asarray(parents).astype(int)
    F, J, _ = joints.shape
    bones = [(int(p), j) for j, p in enumerate(parents) if p is not None and p >= 0]

    # 盒子半宽按骨架尺度定
    span = float(np.linalg.norm(joints[0].max(0) - joints[0].min(0))) or 1.0
    r = thickness * span

    # 每帧构建全骨架顶点(所有骨盒拼接);faces 固定
    seq_verts = []
    for f in range(F):
        vs = [ _bone_box(joints[f, p], joints[f, c], r) for (p, c) in bones ]
        seq_verts.append(np.concatenate(vs, axis=0) if vs else np.zeros((0, 3), np.float32))
    seq_verts = np.stack(seq_verts, 0)                      # (F, nbones*8, 3)
    faces = np.concatenate([_BOX_FACES + k * 8 for k in range(len(bones))], axis=0)  # (nbones*12, 3)

    return mesh_sequence_to_glb(seq_verts, faces, out_path, fps=fps, max_frames=max_frames)


# ------------------------------------------------------------------
# 骨骼蒙皮 glb(纯 python, pygltflib)—— Web 端首选方案。
# morph-target 方案在 three.js 会把大量 target 全叠加导致 mesh 爆炸;
# blender BVH 导入的骨 roll 约定会让末端骨蒙皮 twist 偏差。
# 这里直接按 glTF skin 规范写:关节层级/局部旋转/IBM 全取自我们自己的
# BVH 数据(与 python LBS 完全同一套数学),导出前数值验证 == LBS。
# ------------------------------------------------------------------
def _mk_gltf_builder():
    """返回 (add, finalize):add(arr, comp, acc_type, target, minmax) → accessor idx。"""
    blobs = []
    def add(arr, comp_type, acc_type, target=None, minmax=False):
        a = np.ascontiguousarray(arr)
        raw = a.tobytes()
        off = sum(len(b["padded"]) for b in blobs)
        pad = (-len(raw)) % 4
        blobs.append({"padded": raw + b"\x00" * pad, "raw_len": len(raw), "comp": comp_type,
                      "acc_type": acc_type, "off": off, "arr": a, "target": target, "minmax": minmax})
        return len(blobs) - 1
    def finalize():
        bin_blob = b"".join(b["padded"] for b in blobs)
        bvs, accs = [], []
        ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
        for k, b in enumerate(blobs):
            a, t = b["arr"], b["acc_type"]
            bvs.append(BufferView(buffer=0, byteOffset=b["off"], byteLength=b["raw_len"], target=b["target"]))
            count = a.size // ncomp[t]
            acc = Accessor(bufferView=k, componentType=b["comp"], count=count, type=t, byteOffset=0)
            if b["minmax"]:
                flat = a.reshape(count, -1)
                acc.min = flat.min(axis=0).tolist()
                acc.max = flat.max(axis=0).tolist()
            accs.append(acc)
        return bin_blob, bvs, accs
    return add, finalize


def export_skinned_glb(bvh_pth, char_dir, out_mesh, out_skel, fps=None, validate=True):
    """
    预测 BVH + 角色(base_mesh.obj + skinning_weights.npy)→ 两个标准 skinned/rigid glTF:
      out_mesh: 蒙皮 mesh 动画(model-viewer/three.js 原生正确播放)
      out_skel: 骨架盒可视化(每骨一个刚体盒,挂在关节节点下,同一套 FK 动画)
    validate: 用 glTF 规范公式(numpy)重算蒙皮顶点,对比 python LBS,残差大则抛异常。
    """
    from utils import bvh as BVH
    from utils.mesh import read_obj_mesh, compute_rest_joints

    anim, names, ft = BVH.load(bvh_pth)
    fps = fps or (1.0 / ft if ft else 15.0)
    offsets = np.asarray(anim.offsets, np.float32)            # (J,3) 局部 rest 偏移
    parents = np.asarray(anim.parents, np.int64)              # (J,)
    quats_wxyz = np.asarray(anim.rotations, np.float32)       # (F,J,4) 局部四元数 w,x,y,z
    F, J = quats_wxyz.shape[:2]
    rest_g = compute_rest_joints(anim.offsets, anim.parents).astype(np.float32)  # (J,3) 全局 rest
    # 根平移(与 extract_mesh_from_bvh 一致):pelvis_pos - root_to_pelvis
    trans = (np.asarray(anim.positions, np.float32)[:, 0] - offsets[0:1])        # (F,3)

    verts, faces, uvs, face_uvs = read_obj_mesh(os.path.join(char_dir, "base_mesh.obj"))
    verts = np.asarray(verts, np.float32); faces = np.asarray(faces, np.int64)
    W = np.load(os.path.join(char_dir, "skinning_weights.npy")).astype(np.float32)  # (V,J)
    V = verts.shape[0]
    assert W.shape == (V, J), f"weights {W.shape} != (V={V}, J={J})"

    # 每顶点 top-4 影响(实测本数据 ≤4,无截断)
    top4 = np.argsort(-W, axis=1)[:, :4]                       # (V,4)
    w4 = np.take_along_axis(W, top4, axis=1)                   # (V,4)
    w4 = w4 / np.clip(w4.sum(1, keepdims=True), 1e-8, None)
    joints4 = top4.astype(np.uint16)

    # 贴图:OBJ 的 uv 是分离索引(顶点/uv 各一套)→ 按面角点 unweld,得到每顶点唯一 uv,
    # 同时把 position/joints/weights 也按角点展开(蒙皮不变,只是复制顶点)。
    tex_path = os.path.join(char_dir, "texmap0.png")
    has_tex = (uvs is not None and face_uvs is not None and os.path.exists(tex_path))
    if has_tex:
        uvs = np.asarray(uvs, np.float32); face_uvs = np.asarray(face_uvs, np.int64)
        corner_v = faces.reshape(-1)                      # (Nf*3,) 原顶点索引
        corner_uv = face_uvs.reshape(-1)                  # (Nf*3,) uv 索引
        verts = verts[corner_v]                           # (Nc,3)
        uv_arr = uvs[corner_uv].copy()                    # (Nc,2)
        uv_arr[:, 1] = 1.0 - uv_arr[:, 1]                 # OBJ→glTF v 翻转
        joints4 = joints4[corner_v]; w4 = w4[corner_v]
        faces = np.arange(len(corner_v), dtype=np.int64).reshape(-1, 3)
    faces = faces.astype(np.uint32)

    # glTF 四元数 x,y,z,w + 归一化
    q_xyzw = np.concatenate([quats_wxyz[..., 1:4], quats_wxyz[..., 0:1]], axis=-1)
    q_xyzw = q_xyzw / np.clip(np.linalg.norm(q_xyzw, axis=-1, keepdims=True), 1e-8, None)
    # 四元数连续性对齐:相邻帧点积为负则翻符号,否则 model-viewer 的 LINEAR 插值会走反方向
    # → 关键帧精确但中间帧动作扭曲(q 与 -q 表示同一旋转,但插值路径不同)。逐关节沿时间对齐。
    for j in range(J):
        for f in range(1, F):
            if float(np.dot(q_xyzw[f, j], q_xyzw[f - 1, j])) < 0.0:
                q_xyzw[f, j] = -q_xyzw[f, j]
    q_xyzw = q_xyzw.astype(np.float32)

    # IBM_j = translate(-rest_g[j]),列主序
    ibms = np.zeros((J, 16), np.float32)
    for j in range(J):
        ibms[j] = np.array([1,0,0,0, 0,1,0,0, 0,0,1,0, -rest_g[j,0], -rest_g[j,1], -rest_g[j,2], 1], np.float32)

    times = (np.arange(F, dtype=np.float32) / float(fps))
    root_trans_anim = (offsets[0:1] + trans).astype(np.float32)   # (F,3) 根节点每帧局部平移

    # ---------- 数值验证:glTF 规范 FK+蒙皮 重算 == python LBS ----------
    if validate:
        from utils.mesh import extract_mesh_from_bvh
        lbs_verts, _, _, _ = extract_mesh_from_bvh(
            bvh_pth=bvh_pth, template_pth=os.path.join(char_dir, "base_mesh.obj"),
            save_root="/tmp/_unused", lbs_weights_pth=os.path.join(char_dir, "skinning_weights.npy"),
            return_arrays=True)
        def q2m(q):  # xyzw → 3x3
            x, y, z, w = q
            return np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]], np.float64)
        worst = 0.0
        for f in [0, F // 2, F - 1]:
            G = [None] * J
            for j in range(J):
                Rl = q2m(q_xyzw[f, j])
                Tl = root_trans_anim[f] if j == 0 else offsets[j]
                M = np.eye(4); M[:3, :3] = Rl; M[:3, 3] = Tl
                G[j] = M if parents[j] < 0 else G[parents[j]] @ M
            K = np.stack([G[j] @ ibms[j].reshape(4, 4).T for j in range(J)])   # (J,4,4) 行主序
            Kv = K[joints4]                                                    # (Nc,4,4,4)
            vh = np.concatenate([verts, np.ones((verts.shape[0], 1), np.float32)], 1)
            out = np.einsum("vk,vkab,vb->va", w4.astype(np.float64), Kv.astype(np.float64), vh.astype(np.float64))[:, :3]
            # unweld 后 verts 是角点序 → LBS 真值按 corner_v 展开对比
            ref = lbs_verts[f][corner_v] if has_tex else lbs_verts[f]
            err = np.abs(out - ref).max()
            worst = max(worst, err)
        span = float(np.abs(lbs_verts[0] - lbs_verts[0].mean(0)).max())
        print(f"[skinned_glb] glTF重算 vs LBS 最大误差 = {worst:.6f} (span={span:.3f})")
        if worst > span * 0.01:
            raise RuntimeError(f"skinned glTF 验证失败: err={worst}")

    # ---------- 写 mesh glb ----------
    add, finalize = _mk_gltf_builder()
    ii  = add(faces.reshape(-1).astype(np.uint32), pygltflib.UNSIGNED_INT, "SCALAR", target=pygltflib.ELEMENT_ARRAY_BUFFER)
    ip  = add(verts, pygltflib.FLOAT, "VEC3", target=pygltflib.ARRAY_BUFFER, minmax=True)
    ij  = add(joints4, pygltflib.UNSIGNED_SHORT, "VEC4", target=pygltflib.ARRAY_BUFFER)
    iw  = add(w4.astype(np.float32), pygltflib.FLOAT, "VEC4", target=pygltflib.ARRAY_BUFFER)
    ib  = add(ibms, pygltflib.FLOAT, "MAT4")
    it  = add(times, pygltflib.FLOAT, "SCALAR", minmax=True)
    irt = add(root_trans_anim, pygltflib.FLOAT, "VEC3")
    irq = [add(q_xyzw[:, j], pygltflib.FLOAT, "VEC4") for j in range(J)]

    nodes = []
    for j in range(J):
        # 默认 TRS 用第 0 帧的旋转 + 平移(而非 identity):即使查看器不自动播放动画,
        # 静态显示的也是「预测第 0 帧姿态」而非 bind/rest 姿态(否则看着像动作错了)。
        nodes.append(Node(name=str(names[j]) if j < len(names) else f"j{j}",
                          translation=[float(x) for x in (root_trans_anim[0] if j == 0 else offsets[j])],
                          rotation=[float(x) for x in q_xyzw[0, j]],
                          children=[int(c) for c in np.nonzero(parents == j)[0]] or None))
    _attr = Attributes(POSITION=ip, JOINTS_0=ij, WEIGHTS_0=iw)
    if has_tex:
        _attr.TEXCOORD_0 = add(uv_arr.astype(np.float32), pygltflib.FLOAT, "VEC2", target=pygltflib.ARRAY_BUFFER)
    mesh_node_idx = J
    prim = Primitive(attributes=_attr, indices=ii, material=0)
    nodes.append(Node(name="charmesh", mesh=0, skin=0))

    samplers, channels = [], []
    samplers.append(AnimationSampler(input=it, output=irt, interpolation="LINEAR"))
    channels.append(AnimationChannel(sampler=0, target={"node": 0, "path": "translation"}))
    for j in range(J):
        samplers.append(AnimationSampler(input=it, output=irq[j], interpolation="LINEAR"))
        channels.append(AnimationChannel(sampler=len(samplers) - 1, target={"node": j, "path": "rotation"}))

    bin_blob, bvs, accs = finalize()
    images = images_list = textures = samplers_tex = None
    if has_tex:
        with open(tex_path, "rb") as _tf:
            png = _tf.read()
        pad = (-len(bin_blob)) % 4
        img_off = len(bin_blob) + pad
        bin_blob = bin_blob + b"\x00" * pad + png + b"\x00" * ((-len(png)) % 4)
        bvs.append(BufferView(buffer=0, byteOffset=img_off, byteLength=len(png)))
        img_bv = len(bvs) - 1
        images_list = [pygltflib.Image(bufferView=img_bv, mimeType="image/png")]
        samplers_tex = [pygltflib.Sampler(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]
        textures = [pygltflib.Texture(source=0, sampler=0)]
        material = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "roughnessFactor": 0.7, "metallicFactor": 0.0}, "doubleSided": True}
    else:
        material = {"pbrMetallicRoughness": {"baseColorFactor": [0.72, 0.78, 0.9, 1.0], "roughnessFactor": 0.6, "metallicFactor": 0.0}, "doubleSided": True}
    gltf = GLTF2(
        scene=0, scenes=[Scene(nodes=[0, mesh_node_idx])], nodes=nodes,
        meshes=[Mesh(primitives=[prim])],
        skins=[{"joints": list(range(J)), "inverseBindMatrices": ib, "skeleton": 0}],
        materials=[material],
        images=images_list, textures=textures, samplers=samplers_tex,
        animations=[Animation(samplers=samplers, channels=channels)],
        buffers=[Buffer(byteLength=len(bin_blob))], bufferViews=bvs, accessors=accs)
    gltf.set_binary_blob(bin_blob)
    gltf.save(out_mesh)

    # ---------- 写 skeleton glb(同一 FK,骨盒刚体挂关节节点)----------
    add2, finalize2 = _mk_gltf_builder()
    it2  = add2(times, pygltflib.FLOAT, "SCALAR", minmax=True)
    irt2 = add2(root_trans_anim, pygltflib.FLOAT, "VEC3")
    irq2 = [add2(q_xyzw[:, j], pygltflib.FLOAT, "VEC4") for j in range(J)]
    span = float(np.abs(rest_g - rest_g.mean(0)).max()); r = max(span * 0.012, 1e-4)
    nodes2, meshes2 = [], []
    for j in range(J):
        nodes2.append(Node(name=f"j{j}", translation=[float(x) for x in (offsets[j] if j > 0 else offsets[0])],
                           children=[int(c) for c in np.nonzero(parents == j)[0]] or None))
    # 每骨(p→c)一个盒:几何在 p 局部系(0 → offsets[c]),挂成 p 的额外子节点
    extra = []
    for c in range(J):
        p = parents[c]
        if p < 0:
            continue
        box = _bone_box(np.zeros(3, np.float32), offsets[c].astype(np.float64), r).astype(np.float32)
        iiB = add2(_BOX_FACES.reshape(-1).astype(np.uint32), pygltflib.UNSIGNED_INT, "SCALAR", target=pygltflib.ELEMENT_ARRAY_BUFFER)
        ipB = add2(box, pygltflib.FLOAT, "VEC3", target=pygltflib.ARRAY_BUFFER, minmax=True)
        meshes2.append(Mesh(primitives=[Primitive(attributes=Attributes(POSITION=ipB), indices=iiB, material=0)]))
        node_idx = J + len(extra)
        nodes2.append(Node(name=f"bone_{c}", mesh=len(meshes2) - 1))
        extra.append((int(p), node_idx))
    for p, ni in extra:
        ch = nodes2[p].children or []
        nodes2[p].children = ch + [ni]
    samplers2, channels2 = [], []
    samplers2.append(AnimationSampler(input=it2, output=irt2, interpolation="LINEAR"))
    channels2.append(AnimationChannel(sampler=0, target={"node": 0, "path": "translation"}))
    for j in range(J):
        samplers2.append(AnimationSampler(input=it2, output=irq2[j], interpolation="LINEAR"))
        channels2.append(AnimationChannel(sampler=len(samplers2) - 1, target={"node": j, "path": "rotation"}))
    bin2, bvs2, accs2 = finalize2()
    g2 = GLTF2(scene=0, scenes=[Scene(nodes=[0])], nodes=nodes2, meshes=meshes2,
               materials=[{"pbrMetallicRoughness": {"baseColorFactor": [0.32, 0.85, 1.0, 1.0], "roughnessFactor": 0.4, "metallicFactor": 0.0}, "doubleSided": True}],
               animations=[Animation(samplers=samplers2, channels=channels2)],
               buffers=[Buffer(byteLength=len(bin2))], bufferViews=bvs2, accessors=accs2)
    g2.set_binary_blob(bin2)
    g2.save(out_skel)
    return out_mesh, out_skel


# ------------------------------------------------------------------
# (旧)经 Blender 的 skinned 导出 —— 留作参考;blender BVH 导入的骨 roll
# 会造成末端骨 twist 偏差(验证 35% 残差),已被上面的纯 python 方案替代。
# ------------------------------------------------------------------
def export_skinned_glb_via_blender(bvh_pth, char_dir, out_mesh, out_skel,
                                   blender_sh=None, validate=True, timeout=300):
    """
    bvh_pth:  预测 BVH(convert_npy_to_bvh 产物,骨架序 = skinning 权重列序)
    char_dir: 角色目录(base_mesh.obj + skinning_weights.npy)
    返回 (out_mesh, out_skel);失败抛异常(调用方可回退 morph 方案)。
    """
    import subprocess, tempfile
    from utils import bvh as BVH
    from utils.mesh import read_obj_mesh, compute_rest_joints, extract_mesh_from_bvh

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blender_sh = blender_sh or os.path.join(repo, "blender_mocapanything.sh")
    script = os.path.join(repo, "utils", "blender_skinned_glb.py")

    anim, names, _ = BVH.load(bvh_pth)
    rest_joints = compute_rest_joints(anim.offsets, anim.parents)
    tmpl = os.path.join(char_dir, "base_mesh.obj")
    verts_rest, faces, _, _ = read_obj_mesh(tmpl)
    weights = np.load(os.path.join(char_dir, "skinning_weights.npy"))

    pack = {
        "verts": np.asarray(verts_rest, np.float32),
        "faces": np.asarray(faces, np.int64),
        "weights": np.asarray(weights, np.float32),
        "names": np.array([str(n) for n in names]),
        "rest_joints": np.asarray(rest_joints, np.float32),
    }
    if validate:
        lbs_verts, _, _, _ = extract_mesh_from_bvh(
            bvh_pth=bvh_pth, template_pth=tmpl, save_root="/tmp/_unused",
            lbs_weights_pth=os.path.join(char_dir, "skinning_weights.npy"),
            return_arrays=True)
        F = lbs_verts.shape[0]
        chk = [0, F // 2, F - 1]
        pack["lbs_check"] = lbs_verts[chk].astype(np.float32)
        pack["lbs_check_frames"] = np.array(chk, np.int64)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        np.savez_compressed(tf.name, **pack)
        pack_path = tf.name
    try:
        cmd = [blender_sh, "-b", "-P", script, "--",
               "--bvh", bvh_pth, "--data", pack_path,
               "--out-mesh", out_mesh, "--out-skel", out_skel]
        if validate:
            cmd.append("--validate")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        log = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0 or not (os.path.exists(out_mesh) and os.path.exists(out_skel)):
            raise RuntimeError(f"blender skinned glb 失败(rc={r.returncode}): {log[-600:]}")
        # 验证残差过大 → 判失败(轴向/权重错)
        if validate:
            import re
            ratios = [float(m) for m in re.findall(r"比例=([0-9.]+)", log)]
            if ratios and max(ratios) > 0.02:
                raise RuntimeError(f"skinned glb 验证残差过大 ratios={ratios}: {log[-400:]}")
            print(f"[skinned_glb] 验证通过 ratios={ratios}")
        return out_mesh, out_skel
    finally:
        os.unlink(pack_path)

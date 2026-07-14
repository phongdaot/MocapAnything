"""Server-side render for the shareable composite clip: input | skeleton | mesh.

No Blender, no GL/EGL (unreliable/software-slow on Spaces). The skeleton (FK from the
predicted BVH) and the skinned mesh (same LBS as glb_export) are rendered in the SAME
world space with ONE shared camera, so the two panels show the same character from the
same angle — a small vectorized numpy rasterizer for the mesh, PIL lines/dots for the
skeleton. hstacks with the pipeline's input panel. Returns None on any failure so the
Space keeps working without it.
"""
import os
import numpy as np


# ---------- geometry ----------
def _quat_wxyz_to_mat(q):
    """(...,4) w,x,y,z → (...,3,3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-9
    w, x, y, z = w / n, x / n, y / n, z / n
    m = np.empty(q.shape[:-1] + (3, 3), np.float32)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - z * w); m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w); m[..., 2, 1] = 2 * (y * z + x * w); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _fk_joints(offsets, parents, quats_wxyz, root_pos):
    """Forward kinematics → global joint positions (F,J,3), world space (matches LBS)."""
    F, J = quats_wxyz.shape[:2]
    lm = _quat_wxyz_to_mat(quats_wxyz)                    # (F,J,3,3)
    gm = np.zeros((F, J, 3, 3), np.float32)
    gp = np.zeros((F, J, 3), np.float32)
    gm[:, 0] = lm[:, 0]
    gp[:, 0] = root_pos
    for j in range(1, J):
        p = parents[j]
        gm[:, j] = gm[:, p] @ lm[:, j]
        gp[:, j] = gp[:, p] + np.einsum("fab,b->fa", gm[:, p], offsets[j])
    return gp


# ---------- shared projection ----------
def _projector(all_pts, size, yaw_deg=0.0, margin=0.86):
    # yaw=0 ≡ pipeline 的 camera 视角(与 *_pose_compare_camera 同向,实测对齐);
    # 传非 0 可得 3/4 视角,但会偏离输入相机朝向。
    a = np.radians(yaw_deg); ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], np.float32)
    P = all_pts @ R.T
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2.0
    scale = margin * size / max((hi - lo)[:2].max(), 1e-6)

    def proj(pts):
        p = pts @ R.T
        sx = (p[:, 0] - c[0]) * scale + size / 2.0
        sy = (-(p[:, 1] - c[1])) * scale + size / 2.0
        return sx, sy, p[:, 2]
    return proj, R, c, scale


# ---------- mesh rasterizer ----------
def _rasterize_mesh(sx, sy, sz, faces, normals, size, base_rgb, bg_rgb, light=(0.35, 0.55, 1.0)):
    front = normals[:, 2] > 0
    tri = faces[front]; n = normals[front]
    L = np.asarray(light, np.float32); L = L / np.linalg.norm(L)
    shade = np.clip(n @ L, 0, 1) * 0.7 + 0.4
    p = np.stack([sx, sy], 1)
    p0, p1, p2 = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    z0, z1, z2 = sz[tri[:, 0]], sz[tri[:, 1]], sz[tri[:, 2]]
    img = np.tile(np.asarray(bg_rgb, np.float32), (size, size, 1))
    zbuf = np.full((size, size), 1e9, np.float32)
    base = np.asarray(base_rgb, np.float32)
    minx = np.floor(np.minimum.reduce([p0[:, 0], p1[:, 0], p2[:, 0]])).astype(int)
    maxx = np.ceil(np.maximum.reduce([p0[:, 0], p1[:, 0], p2[:, 0]])).astype(int)
    miny = np.floor(np.minimum.reduce([p0[:, 1], p1[:, 1], p2[:, 1]])).astype(int)
    maxy = np.ceil(np.maximum.reduce([p0[:, 1], p1[:, 1], p2[:, 1]])).astype(int)
    for i in range(len(tri)):
        x0, x1, y0, y1 = max(minx[i], 0), min(maxx[i], size), max(miny[i], 0), min(maxy[i], size)
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        ax, ay = p0[i]; bx, by = p1[i]; cx, cy = p2[i]
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-9:
            continue
        wa = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / d
        wb = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / d
        wc = 1 - wa - wb
        m = (wa >= 0) & (wb >= 0) & (wc >= 0)
        if not m.any():
            continue
        zz = wa * z0[i] + wb * z1[i] + wc * z2[i]
        sub = zbuf[y0:y1, x0:x1]
        upd = m & (zz < sub)
        if not upd.any():
            continue
        sub[upd] = zz[upd]
        img[y0:y1, x0:x1][upd] = base * shade[i]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render_composite(bvh_pth, char_dir, input_panel_mp4, out_mp4, fps=15, size=360,
                     max_frames=110):
    """Render input | skeleton | mesh (skeleton+mesh share one camera). Returns
    (out_mp4, out_fps) or None."""
    try:
        import imageio
        from PIL import Image, ImageDraw
        from utils import bvh as BVH
        from utils.mesh import extract_mesh_from_bvh, read_obj_mesh
    except Exception as e:
        print(f"[composite] deps unavailable: {e}")
        return None
    try:
        anim, names, ft = BVH.load(bvh_pth)
        offsets = np.asarray(anim.offsets, np.float32)
        parents = np.asarray(anim.parents, np.int64)
        quats = np.asarray(anim.rotations, np.float32)          # (F,J,4) wxyz
        root_pos = np.asarray(anim.positions, np.float32)[:, 0]  # (F,3)
        joints = _fk_joints(offsets, parents, quats, root_pos)   # (F,J,3)

        _, faces_t, _, _ = read_obj_mesh(os.path.join(char_dir, "base_mesh.obj"))
        faces = np.asarray(faces_t, np.int64)
        lbs, _, _, _ = extract_mesh_from_bvh(
            bvh_pth=bvh_pth, template_pth=os.path.join(char_dir, "base_mesh.obj"),
            save_root="/tmp/_unused", lbs_weights_pth=os.path.join(char_dir, "skinning_weights.npy"),
            return_arrays=True)
        lbs = np.asarray(lbs, np.float32)                        # (F,V,3)
        F = min(len(lbs), len(joints))
        lbs, joints = lbs[:F], joints[:F]
        out_fps = fps
        if F > max_frames:
            idx = np.linspace(0, F - 1, max_frames).round().astype(int)
            out_fps = fps * max_frames / F
            lbs, joints = lbs[idx], joints[idx]; F = len(idx)

        proj, R, c, scale = _projector(lbs.reshape(-1, 3), size)  # shared camera from mesh
        BG = (0.09, 0.09, 0.11)
        bones = [(j, int(parents[j])) for j in range(len(parents)) if parents[j] >= 0]

        mesh_frames, skel_frames = [], []
        for f in range(F):
            # mesh
            sx, sy, sz = proj(lbs[f])
            e1 = lbs[f][faces[:, 1]] - lbs[f][faces[:, 0]]
            e2 = lbs[f][faces[:, 2]] - lbs[f][faces[:, 0]]
            nn = np.cross(e1, e2)
            nn = (nn @ R.T)
            nn /= (np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9)
            mesh_frames.append(_rasterize_mesh(sx, sy, sz, faces, nn, size, (0.56, 0.73, 0.98), BG))
            # skeleton (PIL lines + dots, depth-sorted color)
            jx, jy, jz = proj(joints[f])
            im = Image.new("RGB", (size, size), tuple(int(v * 255) for v in BG))
            dr = ImageDraw.Draw(im)
            zr = (jz - jz.min()) / (float(jz.max() - jz.min()) + 1e-6)  # np2: ndarray.ptp removed
            for a, b in bones:
                sh = 0.5 + 0.5 * (1 - (zr[a] + zr[b]) / 2)
                col = (int(90 * sh), int(200 * sh), int(255 * sh))
                dr.line([(jx[a], jy[a]), (jx[b], jy[b])], fill=col, width=3)
            for j in range(len(jx)):
                r = 3.2
                dr.ellipse([jx[j] - r, jy[j] - r, jx[j] + r, jy[j] + r], fill=(240, 240, 255))
            skel_frames.append(np.asarray(im))

        work = os.path.dirname(out_mp4)
        skel_mp4 = os.path.join(work, "_skel.mp4"); mesh_mp4 = os.path.join(work, "_mesh.mp4")
        imageio.mimsave(skel_mp4, skel_frames, fps=out_fps, macro_block_size=1)
        imageio.mimsave(mesh_mp4, mesh_frames, fps=out_fps, macro_block_size=1)
        comp = hstack_panels([input_panel_mp4, skel_mp4, mesh_mp4], out_mp4, height=size)
        return (comp, out_fps) if comp else None
    except Exception as e:
        import traceback
        print(f"[composite] render failed: {e}")
        traceback.print_exc()
        return None


def hstack_panels(panels, out_mp4, ffmpeg_exe=None, height=360, common_fps=15):
    """hstack mp4 panels (possibly different fps) into out_mp4, normalizing fps+height."""
    panels = [p for p in panels if p and os.path.exists(p)]
    if len(panels) < 2:
        return None
    if ffmpeg_exe is None:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    cmd = [ffmpeg_exe, "-y"]
    for p in panels:
        cmd += ["-i", p]
    n = len(panels)
    pre = "".join(f"[{i}:v]fps={common_fps},scale=-2:{height}[v{i}];" for i in range(n))
    ins = "".join(f"[v{i}]" for i in range(n))
    cmd += ["-filter_complex", f"{pre}{ins}hstack=inputs={n}[v]", "-map", "[v]",
            "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p", out_mp4]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out_mp4 if r.returncode == 0 and os.path.exists(out_mp4) else None

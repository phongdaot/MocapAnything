
### render_video2mesh.py ###
import os
import trimesh
import pyrender
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# EGL headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# ==== 坐标定义: Y向上，Z朝向你 ====
VIEW_POSITIONS = {
    "front":             [0.1, 0.1, 3],
    "back":              [0.1, 0.1, -3],
    "left":              [-3, 0.1, 0.1],
    "right":             [3, 0.1, 0.1],
    "top":               [0.1, 3, 0.1],
    "bottom":            [0.1, -3, 0.1],
    "top_left_back":     [-3, 3, -3],
    "top_right_front":   [3, 3, 3],
}

def look_at(eye, target, up=[0, 1, 0]):
    """创建 look-at 摄像机姿态矩阵，并确保所有数组为浮点类型。"""
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)
    
    z_axis = (eye - target)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    mat = np.eye(4)
    mat[:3, 0] = x_axis
    mat[:3, 1] = y_axis
    mat[:3, 2] = z_axis
    mat[:3, 3] = eye
    return mat

def vec2str(v):
    return ', '.join([f'{x:.1f}' for x in v])

def is_black_frame(frame):
    # 判断全黑帧（容忍轻微波动，主要为 all<10 即可）
    return np.mean(frame) < 10

def render_one_frame(args):
    """
    负责渲染一个单独的帧。每个进程都会调用此函数。
    为保证进程安全，每个进程在此函数内部创建和销毁自己的 pyrender.OffscreenRenderer 实例。
    """
    frame_idx, glb_file, raw_img_file, glb_dir, img_dir, img_w, img_h, cols, font, selected_view_tuples = args

    renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h)

    try:
        raw_img = Image.open(os.path.join(img_dir, raw_img_file)).convert('RGB').resize((img_w, img_h))
        mesh_or_scene = trimesh.load(os.path.join(glb_dir, glb_file))
        mesh = trimesh.util.concatenate(list(mesh_or_scene.geometry.values())) if isinstance(mesh_or_scene, trimesh.Scene) else mesh_or_scene
    except Exception as e:
        print(f"Error loading file for frame {frame_idx} in dir {glb_dir}: {e}")
        total_views = len(selected_view_tuples) + 1
        rows = (total_views + cols - 1) // cols
        return np.zeros((img_h * rows, img_w * cols, 3), dtype=np.uint8)

    view_imgs = [raw_img]
    for view_name, eye in selected_view_tuples:
        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        camera = pyrender.PerspectiveCamera(yfov=3.14 / 3.0)
        camera_pose = look_at(eye, [0, 0, 0])
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=camera_pose)
        scene.add(pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0), pose=look_at([3, 3, 3], [0, 0, 0]))

        color, _ = renderer.render(scene)
        pil_img = Image.fromarray(color)
        draw = ImageDraw.Draw(pil_img)
        text = f"{view_name}: [{vec2str(eye)}]"
        x = (img_w - draw.textlength(text, font=font)) // 2
        
        # 只保留一次 draw.text 调用，并使用黑色填充
        draw.text((x, 10), text, font=font, fill=(0, 0, 0, 255))
        view_imgs.append(pil_img)

    renderer.delete()

    rows = (len(view_imgs) + cols - 1) // cols
    big_img = Image.new('RGB', (img_w * cols, img_h * rows), (30, 30, 30))
    for idx, im in enumerate(view_imgs):
        big_img.paste(im, ((idx % cols) * img_w, (idx // cols) * img_h))
    return np.array(big_img)

def render_folder(glb_dir, img_dir, output_video_path, selected_views, num_threads, img_w, img_h, cols, font_size):
    """将一系列 .glb 和图像文件渲染成视频。"""
    shared_view_tuples = [(name, VIEW_POSITIONS[name]) for name in selected_views if name in VIEW_POSITIONS]

    # 使用传入的 font_size 参数
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()

    glb_files = sorted([f for f in os.listdir(glb_dir) if f.endswith('.glb')],
                       key=lambda x: int(os.path.splitext(x)[0]))
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')],
                       key=lambda x: int(os.path.splitext(x)[0]))
    n_frames = min(len(glb_files), len(img_files))
    if n_frames == 0:
        print(f"{glb_dir}: 没有帧，跳过")
        return

    print(f"{glb_dir}: 开始处理，共 {n_frames} 帧")
    args_list = [
        (idx, glb_files[idx], img_files[idx], glb_dir, img_dir, img_w, img_h, cols, font, shared_view_tuples)
        for idx in range(n_frames)
    ]
    
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        frames = list(tqdm(executor.map(render_one_frame, args_list), total=n_frames, desc=f"Rendering {os.path.basename(glb_dir)}"))

    # ====== 新增全黑帧检查 ======
    has_black = any(is_black_frame(frame) for frame in frames)
    if has_black:
        print(f"[Warning] Skip saving video {output_video_path} (contains black frame(s), likely permission error)")
        return
    # ====== ============== ======

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    imageio.mimsave(output_video_path, frames, fps=10)
    print(f"{glb_dir}: 视频已保存到 {output_video_path}")

def main(input_root, candidate_roots, selected_views, img_w, img_h, cols, font_size=32, num_threads=4):
    """遍历目录，找到 .glb 文件，并渲染成视频，同时保留目录结构。"""
    renders_dir = 'outputs/video2mesh'

    for dirpath, dirnames, filenames in os.walk(input_root):
        glb_files = [f for f in filenames if f.endswith('.glb')]
        if not glb_files:
            continue

        glb_dir = dirpath
        
        relative_path = os.path.relpath(glb_dir, input_root)
        
        prefix = relative_path.split(os.sep)[0]
        
        img_dir = None
        for root in candidate_roots:
            candidate_path = os.path.join(root, prefix)
            # candidate_path = os.path.join(root, prefix.replace('_withglobal', '').replace('_noglobal', ''))  # 0828特殊情况.
            print(f'Looking for image directory: {candidate_path} for prefix: {prefix}')
            if os.path.exists(candidate_path):
                img_dir = candidate_path
                break
        
        if img_dir is None:
            print(f"Could not find corresponding image directory for: {prefix}. Skipping {glb_dir}.")
            continue

        out_video_dir = os.path.join(renders_dir, input_root.split('/')[1], os.path.dirname(relative_path))
        out_video_name = os.path.basename(glb_dir) + '.mp4'
        out_video_path = os.path.join(out_video_dir, out_video_name)

        # ✅ 若视频已存在则跳过
        if os.path.exists(out_video_path):
            print(f"[Skip] Video already exists: {out_video_path}")
            continue

        # 将 font_size 参数传递给 render_folder
        render_folder(glb_dir, img_dir, out_video_path, selected_views, num_threads, img_w, img_h, cols, font_size)

# if __name__ == "__main__":
    
#     glb_input_root='output_video2mesh/selected_nbg_sample/'
#     image_candidate_roots=['../data-test/selected_nbg_sample']

#     main(
#         input_root=glb_input_root,
#         candidate_roots=image_candidate_roots,
#         # other setting
#         selected_views=['front', 'bottom', 'left'],
#         img_w=512,
#         img_h=512,
#         cols=2,
#         font_size=18,  
#         num_threads=64
#     )

import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_folder", type=str, required=True)
    args = parser.parse_args()

    glb_input_root = f"output_video2mesh/{args.test_folder}/"
    image_candidate_roots = [f"./data-test/{args.test_folder}"]

    main(
        input_root=glb_input_root,
        candidate_roots=image_candidate_roots,
        selected_views=['front', 'bottom', 'left'],
        img_w=512,
        img_h=512,
        cols=2,
        font_size=18,
        num_threads=64
    )


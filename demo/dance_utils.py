### dance_utils.py ###
"""Dance Anything 音视频工具:标准化 30fps、抽音轨、并排合成、配乐回填。
全部走 imageio_ffmpeg 自带的 ffmpeg 二进制(不依赖系统 PATH)。"""
import os
import subprocess
import imageio_ffmpeg


def _ff():
    return imageio_ffmpeg.get_ffmpeg_exe()


def has_audio(video_path):
    """视频是否含音轨。"""
    try:
        r = subprocess.run([_ff(), "-i", video_path, "-hide_banner"],
                           capture_output=True, text=True)
        return "Audio:" in (r.stderr or "")
    except Exception:
        return False


def extract_audio(video_path, out_audio):
    """抽原始音轨(aac/m4a)。无音轨返回 None。"""
    if not has_audio(video_path):
        return None
    r = subprocess.run([_ff(), "-y", "-loglevel", "error", "-i", video_path,
                        "-vn", "-acodec", "aac", "-b:a", "192k", out_audio],
                       capture_output=True, text=True)
    return out_audio if os.path.exists(out_audio) and os.path.getsize(out_audio) > 0 else None


def standardize_30fps(video_path, out_video, max_seconds=None, fps=30):
    """重编码到 30fps(去音轨);max_seconds 截断时长。返回 out_video。"""
    cmd = [_ff(), "-y", "-loglevel", "error", "-i", video_path]
    if max_seconds and max_seconds > 0:
        cmd += ["-t", str(float(max_seconds))]
    cmd += ["-an", "-r", str(fps), "-vsync", "cfr",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", "-preset", "fast", out_video]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_video


def frames_from_video(video_path, out_dir, ext="png"):
    """把(已 30fps 的)视频抽成逐帧图,00000.png 起。返回帧数。"""
    os.makedirs(out_dir, exist_ok=True)
    pat = os.path.join(out_dir, f"%05d.{ext}")
    subprocess.run([_ff(), "-y", "-loglevel", "error", "-i", video_path,
                    "-start_number", "0", pat], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return len([f for f in os.listdir(out_dir) if f.endswith(ext)])


def compose_side_by_side(video_paths, out_path, height=480, fps=30):
    """多个视频统一高度后横向并排(hstack)。返回 out_path。"""
    vids = [v for v in video_paths if v and os.path.exists(v)]
    if len(vids) < 2:
        # 只有一个就直接标准化输出
        if vids:
            subprocess.run([_ff(), "-y", "-loglevel", "error", "-i", vids[0],
                            "-vf", f"scale=-2:{height}", "-r", str(fps), out_path],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_path
        return None
    cmd = [_ff(), "-y", "-loglevel", "error"]
    for v in vids:
        cmd += ["-i", v]
    fc = ";".join(f"[{i}:v]scale=-2:{height},fps={fps}[v{i}]" for i in range(len(vids))) + ";"
    fc += "".join(f"[v{i}]" for i in range(len(vids))) + f"hstack=inputs={len(vids)}[v]"
    cmd += ["-filter_complex", fc, "-map", "[v]",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", "-preset", "fast", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def mux_audio(video_path, audio_path, out_path):
    """把音轨混回视频(-shortest 对齐时长)。无音轨则直接拷贝视频。"""
    if not audio_path or not os.path.exists(audio_path):
        if video_path != out_path:
            subprocess.run([_ff(), "-y", "-loglevel", "error", "-i", video_path,
                            "-c", "copy", out_path], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path if os.path.exists(out_path) else video_path
    subprocess.run([_ff(), "-y", "-loglevel", "error", "-i", video_path, "-i", audio_path,
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    "-shortest", out_path], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path if os.path.exists(out_path) else video_path

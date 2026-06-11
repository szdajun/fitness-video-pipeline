"""
make_video_lib.py
共享工具库：跟拍裁切 + 拼 final + 抽帧验证 + 抽音轨
复用现有 main.py 内部逻辑，避免重新发明轮子。
"""
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"

# ============================================================
#  基础工具
# ============================================================

def run(cmd, check=True, capture=False):
    """跑命令，check=True 时失败抛错"""
    if isinstance(cmd, list):
        cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    else:
        cmd_str = cmd
    print(f"[RUN] {cmd_str[:200]}", flush=True)
    return subprocess.run(
        cmd, check=check, capture_output=capture,
        text=True, encoding="utf-8", errors="ignore"
    )


def get_video_info(path):
    """ffprobe 读视频信息.
    关键: 必须 -select_streams v:0 只读 video stream;
    否则音频 stream 的 r_frame_rate=0/0 会覆盖 fps, 算成 30 fallback → 慢动作 bug.
    """
    result = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "format=duration:stream=width,height,r_frame_rate,nb_frames,codec_name",
        "-of", "default", str(path)
    ], capture=True, check=False)
    info = {"path": path, "duration": 0, "w": 0, "h": 0, "fps": 0, "frames": 0}
    for line in result.stdout.splitlines():
        if line.startswith("duration="):
            info["duration"] = float(line.split("=", 1)[1])
        elif line.startswith("width="):
            info["w"] = int(line.split("=", 1)[1])
        elif line.startswith("height="):
            info["h"] = int(line.split("=", 1)[1])
        elif line.startswith("r_frame_rate="):
            fps_str = line.split("=", 1)[1]
            if "/" in fps_str:
                num, den = fps_str.split("/")
                info["fps"] = float(num) / float(den) if float(den) > 0 else 30
            else:
                info["fps"] = float(fps_str)
        elif line.startswith("nb_frames="):
            info["frames"] = int(line.split("=", 1)[1])
    return info


def check_disk(required_gb=30, path="F:/"):
    """启动前检查磁盘"""
    import shutil
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    print(f"[DISK] {path} 盘可用 {free_gb:.1f} GB (需 >= {required_gb} GB)")
    if free_gb < required_gb:
        print(f"!!! 磁盘空间不足! 需 {required_gb} GB, 只有 {free_gb:.1f} GB")
        return False
    return True


def clean_temp_dirs():
    """清空所有临时目录 (用 rmtree 整体清, 比逐个 unlink 快 100x)"""
    import shutil
    project_root = Path(__file__).parent
    for dirname in ["_temp", "_temp_track3x4", "_temp_track9x16"]:
        d = project_root / dirname
        if d.exists():
            try:
                shutil.rmtree(d)
            except OSError:
                pass
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
#  跟拍裁切（OpenCV 逐帧）
# ============================================================

def track_crop(video_in, keypoints_json, out_dir, out_w, out_h, crop_aspect):
    """OpenCV 跟拍裁切 (用 lead_person 的 nose x 坐标动态裁剪)
    out_dir: 输出 PNG 序列目录
    out_w, out_h: 最终输出尺寸
    crop_aspect: 9/16 或 3/4
    """
    in_w, in_h = 2560, 1080
    # 实际源尺寸以 keypoints 推断 (在 v2 链中是 2560x1080, 留个 param 余地)
    cap = cv2.VideoCapture(str(video_in))
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_w, in_h = real_w, real_h
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[TRACK] 源 {in_w}x{in_h} @ {fps:.2f}fps, {n} 帧 → 跟拍 {out_w}x{out_h}")

    # 算裁切宽 (in_h * crop_aspect)
    cw = int(in_h * crop_aspect)
    cw = cw if cw % 2 == 0 else cw - 1
    ch = in_h
    if cw > in_w:
        print(f"  [WARN] cw {cw} > in_w {in_w}, 限制为 in_w")
        cw = in_w

    # 解析 keypoints
    with open(keypoints_json, encoding="utf-8") as f:
        data = json.load(f)
    frames = data["keypoints"]

    cx_list = []
    for i in range(n):
        persons = frames[str(i)]
        best, best_score = None, -1
        for p in persons:
            score = sum(kp[2] for kp in p if kp[2] > 0)
            if score > best_score:
                best_score = score
                best = p
        if best and best[0][2] > 0:
            cx_list.append(best[0][0])
        else:
            cx_list.append(0.5)

    # 滑动均值平滑
    W = 5
    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - W), min(n, i + W + 1)
        smoothed.append(statistics.mean(cx_list[lo:hi]))
    smoothed = [max(0.25, min(0.75, c)) for c in smoothed]

    # 写 PNG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        c = smoothed[frame_idx] if frame_idx < len(smoothed) else 0.5
        x = int(c * in_w - cw / 2)
        x = max(0, min(x, in_w - cw))
        x = x if x % 2 == 0 else x - 1
        cropped = frame[0:ch, x:x+cw]
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(out_dir / f"f_{frame_idx:06d}.png"), resized)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{n}")
    cap.release()
    print(f"  [TRACK] {frame_idx} 帧 → {out_dir}")
    return frame_idx


# ============================================================
#  拼 final 视频 (intro+body+outro+音轨)
# ============================================================

def extract_full_audio(source_video, output_aac, intro_dur=4.0, outro_dur=2.5, fade_in=0.5, fade_out_sec=2.4):
    """构造 38.4s 完整音轨 (intro 淡入 + 主体 + outro 淡出)
    source_video: 源视频 (只有 80.3s 主体音轨)
    output_aac: 输出 m4a 路径
    """
    output_aac = Path(output_aac)
    # 抽源音轨
    raw = output_aac.with_suffix(".raw.m4a")
    run([
        "ffmpeg", "-y", "-i", str(source_video), "-vn", "-acodec", "copy", str(raw)
    ], check=True)

    # 拼 3 段: intro 淡入 (0~4s) + 主体 (0~34.4s) + outro 淡出 (31.9~34.4s)
    body_dur = get_video_info(raw)["duration"]
    run([
        "ffmpeg", "-y", "-i", str(raw),
        "-filter_complex",
        f"[0:a]asetpts=PTS-STARTPTS,atrim=0:{intro_dur},afade=t=in:st=0:d={fade_in},volume=0.8[intro];"
        f"[0:a]asetpts=PTS-STARTPTS,atrim=0:{body_dur:.2f}[body];"
        f"[0:a]asetpts=PTS-STARTPTS,atrim={body_dur - fade_out_sec:.2f}:{body_dur:.2f},"
        f"afade=t=out:st=0:d={fade_out_sec},volume=0.7[outro];"
        f"[intro][body][outro]concat=n=3:v=0:a=1[outa]",
        "-map", "[outa]",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-t", f"{intro_dur + body_dur + 0.5:.2f}",
        str(output_aac),
    ], check=True)
    raw.unlink(missing_ok=True)
    print(f"  [AUDIO] {output_aac}")


def build_final_from_png(png_dir, audio_aac, output_mp4, out_w, out_h, fps=30, timescale=None):
    """把 PNG 序列 + 完整音轨拼成主体视频.
    fps: 关键参数, 必须等于源视频的 fps (否则慢动作/快进).
    timescale: 容器时基, 默认 = round(fps) 整数; 60fps 源用 60, 30fps 用 30.
    """
    if timescale is None:
        timescale = int(round(fps))
    png_dir = Path(png_dir)
    audio_aac = Path(audio_aac)
    output_mp4 = Path(output_mp4)
    print(f"  [BUILD] fps={fps:.2f} timescale={timescale}")
    run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(png_dir / "f_%06d.png"),
        "-i", str(audio_aac),
        "-filter_complex", f"[0:v]scale={out_w}:{out_h}:flags=lanczos[v]",
        "-map", "[v]", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        "-video_track_timescale", str(timescale),
        str(output_mp4),
    ], check=True)
    print(f"  [BUILD] {output_mp4}")


def build_full_video(intro_path, body_path, outro_path, audio_aac, output_mp4,
                     out_w, out_h, fps=30, timescale=None):
    """拼 intro+body+outro+音轨 的完整 final.
    fps: 必须等于源视频 fps (否则慢动作/快进).
    intro/outro 会被 fps filter 重采样到目标 fps, 确保 timeline 对齐.
    """
    if timescale is None:
        timescale = int(round(fps))
    print(f"  [BUILD] fps={fps:.2f} timescale={timescale}")
    run([
        "ffmpeg", "-y",
        "-i", str(intro_path),
        "-i", str(body_path),
        "-i", str(outro_path),
        "-i", str(audio_aac),
        "-filter_complex",
        # 关键: 每段都先 fps={fps} 强制重采样, 否则不同 fps 段拼接会乱
        f"[0:v]fps={fps},scale={out_w}:{out_h}:flags=lanczos,setsar=1[v0];"
        f"[1:v]fps={fps},scale={out_w}:{out_h}:flags=lanczos,setsar=1[v1];"
        f"[2:v]fps={fps},scale={out_w}:{out_h}:flags=lanczos,setsar=1[v2];"
        f"[v0][v1][v2]concat=n=3:v=1:a=0[outv]",
        "-map", "[outv]", "-map", "3:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        "-video_track_timescale", str(timescale),
        str(output_mp4),
    ], check=True)
    print(f"  [BUILD] {output_mp4}")


# ============================================================
#  抽帧验证
# ============================================================

def verify_video(video_path, out_dir, timestamps=(0.5, 15, 30, 38), scale=480):
    """抽多帧验证"""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for t in timestamps:
        out = out_dir / f"{Path(video_path).stem}_t{t}.jpg"
        run([
            "ffmpeg", "-y", "-ss", str(t),
            "-i", str(video_path), "-vframes", "1",
            "-vf", f"scale={scale}:-1",
            str(out),
        ], check=False, capture=True)
    print(f"  [VERIFY] {video_path.name} 抽帧 {len(timestamps)} 张")


# ============================================================
#  关键修复：ffmpeg 失败检测
# ============================================================

def run_ffmpeg_check(cmd, description="ffmpeg"):
    """跑 ffmpeg 必检查 returncode, 失败必 throw"""
    print(f"[FFMPEG] {description}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        err_tail = (result.stderr or "")[-500:]
        raise RuntimeError(f"{description} 失败 (rc={result.returncode}): {err_tail}")
    return result


# ============================================================
#  换背景 (SAM2 集成) - 保留 main.py _run_bgswap
# ============================================================

def run_bgswap(video_path, coach_name=None, bg_image=None):
    """SAM2 背景替换 + 换脸 (复用 main.py _run_bgswap 逻辑)"""
    import subprocess as sp
    project_root = Path(__file__).parent

    comfy_py = "F:/wkspace/ComfyUI/venv/Scripts/python.exe"
    if not Path(comfy_py).exists():
        print(f"[bgswap] 跳过: ComfyUI Python 不存在 {comfy_py}")
        return None

    stable_script = project_root / "tools" / "bgswap_stable.py"
    if stable_script.exists():
        script = str(stable_script)
        mode = "运镜匹配"
    else:
        script = str(project_root / "tools" / "sam2_bg_swap.py")
        mode = "SAM2"

    cmd = [
        comfy_py, script,
        "--video", str(video_path),
        "--coach", coach_name or "",
    ]
    if bg_image:
        cmd.extend(["--bg-image", str(bg_image)])

    print(f"[bgswap] 跑 {mode}: {' '.join(cmd)}")
    result = sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        print(f"[bgswap] 失败: {result.stderr[-500:]}")
        return None
    return result

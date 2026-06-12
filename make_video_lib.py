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

    fps 优先级: avg_frame_rate > nb_frames/duration > r_frame_rate > 30
    - r_frame_rate 不可信: H.264 MP4 通常 90000/1 (time_scale), 跟实际 fps 无关
    - avg_frame_rate 是 ffmpeg 按真实播放速度算的, 准
    - 极端情况下 avg 是 0/0, 用 nb_frames/duration 兜底
    """
    result = run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "format=duration:stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name",
        "-of", "default", str(path)
    ], capture=True, check=False)
    info = {"path": path, "duration": 0, "w": 0, "h": 0, "fps": 0, "frames": 0}
    avg_fps = 0
    raw_fps = 0
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
                raw_fps = float(num) / float(den) if float(den) > 0 else 0
            else:
                raw_fps = float(fps_str)
        elif line.startswith("avg_frame_rate="):
            fps_str = line.split("=", 1)[1]
            if "/" in fps_str:
                num, den = fps_str.split("/")
                avg_fps = float(num) / float(den) if float(den) > 0 else 0
            else:
                avg_fps = float(fps_str)
        elif line.startswith("nb_frames="):
            info["frames"] = int(line.split("=", 1)[1])

    # 选最可信的 fps
    if 0 < avg_fps < 1000:           # 99% 情况
        info["fps"] = avg_fps
    elif info["frames"] > 0 and info["duration"] > 0:
        info["fps"] = info["frames"] / info["duration"]  # 兜底
    elif 0 < raw_fps < 1000:
        info["fps"] = raw_fps
    else:
        info["fps"] = 30  # 实在读不出来
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

def track_crop(video_in, keypoints_json, out_dir, out_w, out_h, crop_aspect,
               smooth_window: int = 60, max_step_ratio: float = 0.005,
               dead_zone_ratio: float = 0.12):
    """OpenCV 跟拍裁切 (稳版, 修左右扫动)

    关键修复 (全面):
    - 滑动窗口 5 → 30 帧 (~0.5s, 更稳)
    - 选躯干中心 (肩+胯平均) 代替 nose, 教练转身低头 nose 跳变/丢失时仍稳
    - nose 不可见时**用上一帧的 cx** (而不是跳回 0.5 中央, 那是突然大扫)
    - 限速: 相邻帧 crop 中心变化不超过 max_step_ratio * in_w (避免突变)
    - 中央死区: 教练位置在画面中央 ±dead_zone_ratio 范围时, crop 不动 (减少无谓扫动)
    - 滑窗后再做一次中位数滤波去极值 (单帧 keypoint 误检)

    out_dir: 输出 PNG 序列目录
    out_w, out_h: 最终输出尺寸
    crop_aspect: 9/16 或 3/4
    """
    in_w, in_h = 2560, 1080
    cap = cv2.VideoCapture(str(video_in))
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_w, in_h = real_w, real_h
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[TRACK] 源 {in_w}x{in_h} @ {fps:.2f}fps, {n} 帧 → 跟拍 {out_w}x{out_h}, 平滑窗口={smooth_window}")

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

    # YOLO 17 keypoints 顺序 (COCO):
    # 0=nose 5=left_shoulder 6=right_shoulder 11=left_hip 12=right_hip
    # 我们用躯干中心 = (左肩+右肩+左胯+右胯)/4, 比 nose 稳
    cx_list = []
    for i in range(n):
        persons = frames.get(str(i), [])
        best, best_score = None, -1
        for p in persons:
            score = sum(kp[2] for kp in p if kp[2] > 0)
            if score > best_score:
                best_score = score
                best = p
        if best:
            # 优先用躯干 4 点, 至少 2 个置信度 > 0.3 才用
            torso_pts = []
            for idx in (5, 6, 11, 12):
                if idx < len(best) and best[idx][2] > 0.3:
                    torso_pts.append(best[idx][0])
            if len(torso_pts) >= 2:
                cx_list.append(sum(torso_pts) / len(torso_pts))
            elif best[0][2] > 0.3:
                cx_list.append(best[0][0])  # 退到 nose
            else:
                cx_list.append(None)  # 标记丢失, 后面用上一帧填补
        else:
            cx_list.append(None)

    # 1) 填补 None: 用最近一个有值的帧
    last_cx = 0.5
    for i in range(n):
        if cx_list[i] is None:
            cx_list[i] = last_cx
        else:
            last_cx = cx_list[i]

    # 2) 中位数滤波去单帧极值 (3 帧邻域, 防误检)
    cx_med = []
    for i in range(n):
        lo, hi = max(0, i - 1), min(n, i + 2)
        cx_med.append(statistics.median(cx_list[lo:hi]))

    # 3) 滑动均值平滑 (smooth_window 帧, 0.5s 量级)
    W = smooth_window
    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - W), min(n, i + W + 1)
        smoothed.append(statistics.mean(cx_med[lo:hi]))

    # 4) 起点: 用全段 cx 中位数, 避免首帧 YOLO 误检把死区定到最左/最右
    # 钳制到 [0.35, 0.65] 范围, 避免极端中位数让画面偏太远
    start_cx = max(0.35, min(0.65, statistics.median(smoothed)))

    # 5) 中央死区 (画面稳心) - 这是关键!
    # 思路: 观众要看动作, 教练在中央 ±dead_zone 范围时, crop 锁 start_cx 完全不动.
    #      教练走出死区时, crop **慢慢**跟随 (inner_max_step, 比 max_step 慢很多),
    #      直到教练在画面里重新到死区边缘, 然后 crop 锁死区边缘.
    #      这样画面"基本不扫", 教练可以小幅移动不影响阅读.
    final = []
    cur_cx = start_cx
    half_dead = dead_zone_ratio
    # 死区外最大速度: 0.003 * in_w/帧 = 0.17%/帧. 30 帧=1s 最多走 5% in_w.
    inner_max_step = max_step_ratio * in_w * 0.6
    for i in range(n):
        desired = smoothed[i]
        delta = desired - cur_cx
        # 死区外: 慢慢向 desired 走
        if abs(delta) > inner_max_step:
            cur_cx = cur_cx + inner_max_step * (1 if delta > 0 else -1)
        else:
            cur_cx = desired
        # 钳制不让 crop 越过死区外缘 (在画面里始终有 dead_zone 的余量)
        if cur_cx > 0.5 + half_dead:
            cur_cx = 0.5 + half_dead
        if cur_cx < 0.5 - half_dead:
            cur_cx = 0.5 - half_dead
        final.append(cur_cx)

    # 钳制到 [0.25, 0.75] 避免裁出画
    final = [max(0.25, min(0.75, c)) for c in final]

    print(f"  [TRACK] cx 范围: {min(final):.3f} ~ {max(final):.3f}, 变动: {max(final)-min(final):.3f}")

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
        c = final[frame_idx] if frame_idx < len(final) else 0.5
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


def build_final_from_png(png_dir, audio_aac, output_mp4, out_w, out_h, fps=30, timescale=None, pad_color="black"):
    """把 PNG 序列 + 完整音轨拼成主体视频.
    fps: 必须等于源视频 fps (否则慢动作/快进).
    timescale: 容器时基, 默认 = round(fps).
    用 scale+pad 居中 letterbox, 防止源 aspect != 目标 aspect 时被强拉变形.
    """
    if timescale is None:
        timescale = int(round(fps))
    png_dir = Path(png_dir)
    audio_aac = Path(audio_aac)
    output_mp4 = Path(output_mp4)
    print(f"  [BUILD] fps={fps:.2f} timescale={timescale} target={out_w}x{out_h}")
    fit = (f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
           f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:{pad_color}")
    run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(png_dir / "f_%06d.png"),
        "-i", str(audio_aac),
        "-filter_complex", f"[0:v]{fit},setsar=1[v]",
        "-map", "[v]", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        "-video_track_timescale", str(timescale),
        str(output_mp4),
    ], check=True)
    print(f"  [BUILD] {output_mp4}")


def build_full_video(intro_path, body_path, outro_path, audio_aac, output_mp4,
                     out_w, out_h, fps=30, timescale=None, pad_color="black"):
    """拼 intro+body+outro+音轨 的完整 final.
    fps: 必须等于源视频 fps (否则慢动作/快进).

    关键: 每段都先按目标 aspect scale + pad 居中 (letterbox, 不拉伸).
    intro/outro 是 16:9 (2560x1080), body 跟拍后是 9:16 或 3:4.
    如果用 scale=W:H:flags=lanczos 强拉, 16:9 intro 到 9:16 会被上下拉宽, 字体人物全变形.
    修法: scale 按 fit-in 缩放, pad 黑边居中, 保留原 aspect.
    """
    if timescale is None:
        timescale = int(round(fps))
    print(f"  [BUILD] fps={fps:.2f} timescale={timescale} target={out_w}x{out_h}")
    # scale+pad: 按目标 W:H 比例缩放 + 黑边居中 (letterbox)
    fit = (f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
           f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:{pad_color}")
    run([
        "ffmpeg", "-y",
        "-i", str(intro_path),
        "-i", str(body_path),
        "-i", str(outro_path),
        "-i", str(audio_aac),
        "-filter_complex",
        f"[0:v]fps={fps},{fit},setsar=1[v0];"
        f"[1:v]fps={fps},{fit},setsar=1[v1];"
        f"[2:v]fps={fps},{fit},setsar=1[v2];"
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

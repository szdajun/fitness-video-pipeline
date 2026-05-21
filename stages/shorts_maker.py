"""智能 Shorts 生成 — 选高能段 + 教练居中 + 放大 + 美颜"""

import json, os, subprocess, tempfile
from pathlib import Path


def find_best_segment(keypoints_file, beat_frames, highlight_data, fps=30, duration=15):
    """从姿态+节拍+高光数据中找出最精彩的15秒段"""
    if not os.path.exists(keypoints_file):
        return 0.3  # fallback: 从30%位置

    with open(keypoints_file, encoding="utf-8") as f:
        data = json.load(f)
    kps = data.get("keypoints", data)

    if not beat_frames:
        return 0.3

    window = int(duration * fps)
    total_frames = max(int(k) for k in kps.keys()) + 1 if isinstance(kps, dict) else len(kps)
    total_frames = min(total_frames, 9000)

    best_start = 0
    best_score = -1

    # 滑动窗口评估每段
    for start_frame in range(0, total_frames - window, int(fps * 0.5)):
        end_frame = start_frame + window
        # 节拍密度
        beat_count = sum(1 for bf in beat_frames if start_frame <= bf < end_frame)
        # 运动量（从关键点计算）
        motion_sum = 0
        for f in range(start_frame, min(end_frame, total_frames), 3):
            entry = kps.get(str(f))
            if entry and isinstance(entry, list) and entry:
                person = entry[0]
                if isinstance(person[0], list):
                    for p in person:
                        if p[2] > 0.3:
                            # 与前一帧比较位移
                            motion_sum += 1
        score = beat_count * 3 + motion_sum * 0.1
        if score > best_score:
            best_score = score
            best_start = start_frame

    return max(best_start / fps, 0.1)


def get_coach_center(keypoints_file, frame_idx):
    """获取指定帧的教练中心位置（归一化坐标 0-1）"""
    try:
        with open(keypoints_file, encoding="utf-8") as f:
            data = json.load(f)
        kps = data.get("keypoints", data)
        entry = kps.get(str(frame_idx))
        if entry and isinstance(entry, list) and entry:
            person = entry[0]
            if isinstance(person[0], list):
                # 用髋部中点作为人体中心
                lhip = person[11]
                rhip = person[12]
                if lhip[2] > 0.3 and rhip[2] > 0.3:
                    return (lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2
    except Exception:
        pass
    return 0.5, 0.5  # 默认画面中心


def make_smart_shorts(video_path, output_dir, keypoints_file,
                      beat_frames=None, duration=15, beauty=True):
    """生成智能 Shorts"""
    import shutil as _shutil
    ffmpeg = "C:/Users/18091/ffmpeg/ffmpeg.exe"
    if not os.path.exists(ffmpeg):
        ffmpeg = _shutil.which("ffmpeg") or "ffmpeg"
    ffprobe = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
    if not os.path.exists(ffprobe):
        ffprobe = _shutil.which("ffprobe") or "ffprobe"

    # Get duration
    r = subprocess.run([ffprobe, "-v", "quiet", "-show_entries", "format=duration",
                       "-of", "csv=p=0", video_path],
                      capture_output=True, text=True, encoding="utf-8")
    total_dur = float(r.stdout.strip())

    # Find best starting time
    start_time = find_best_segment(keypoints_file, beat_frames or [], {}, fps=30,
                                   duration=duration)
    start_time = min(start_time, total_dur - duration)
    start_time = max(start_time, 0)

    # Get coach position at the middle of the segment for crop centering
    mid_frame = int((start_time + duration / 2) * 30)
    cx, cy = get_coach_center(keypoints_file, mid_frame)

    # Build filter: centered crop around coach + zoom + beautify
    # Crop a 9:16 window centered on the coach's x position
    # Original is 1920x1080. 9:16 output is 1080x1920.
    # Input: width=1920, height=1080. Target aspect = 9/16
    # Crop height = input height = 1080. Crop width = 1080 * 9/16 = 607.5 ≈ 608
    crop_w = 608
    crop_h = 1080

    # Center the crop on coach x, but keep it within frame bounds
    coach_x_px = int(cx * 1920)
    crop_x = max(0, min(1920 - crop_w, coach_x_px - crop_w // 2))

    # 放大 1.2x (zoom in on coach)
    zoom = 1.15
    zoomed_w = int(crop_w / zoom)
    zoomed_h = int(crop_h / zoom)
    zoomed_x = crop_x + (crop_w - zoomed_w) // 2

    filters = [
        f"crop={zoomed_w}:{zoomed_h}:{zoomed_x}:0",
        f"scale=1080:1920:flags=lanczos",
    ]

    if beauty:
        # Light skin smoothing + slight contrast for beauty
        filters.append("smartblur=1.5:0.8:0")

    vf = ",".join(filters)

    out_path = os.path.join(output_dir,
        f"{Path(video_path).stem}_smartshorts.mp4")

    print(f"    智能 Shorts: start={start_time:.1f}s, coach_x={cx:.2f}, zoom={zoom}")
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start_time), "-t", str(duration),
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                      encoding="utf-8", errors="replace", timeout=60)

    if r.returncode != 0:
        print(f"    Shorts 失败: {r.stderr[-200:]}")
        return None

    return out_path


def shutil_which(name):
    import shutil
    return shutil.which(name)

"""
Fitness Video Batch Publisher v2
AI SEO titles + hook overlays + Shorts clips + YouTube upload
"""

import os, sys, subprocess, time, re, glob, random, json
from pathlib import Path

# OpenCV 4.13.0 FFMPEG DLL 编译时依赖 openh264 DLL，系统缺少则 VideoWriter 初始化失败。
# 优先用 Windows Media Foundation 编码器绕过该问题。
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "MSMF")

SOURCE_DIR = r"C:\Users\18091\Desktop\短视频素材"
PIPELINE = os.path.join(os.path.dirname(__file__), "main.py")
CONFIG = os.path.join(os.path.dirname(__file__), "config.yaml")
VENV_PY = r"F:\wkspace\ComfyUI\venv\Scripts\python.exe"
DAY_COUNTER_FILE = os.path.join(os.path.dirname(__file__), "day_counter.json")

COACH_MAP = {
    "艳青": "胭脂虎", "丽丽": "腰女", "建玲": "三宝妈",
    "小红豆": "红娘子", "郭海军": "老兵不老", "枫林红": "霸道总裁",
    "李刚": "托塔天王",
    "小飞侠": "节拍战神",
}


def _get_day():
    coils = {}
    if os.path.exists(DAY_COUNTER_FILE):
        with open(DAY_COUNTER_FILE) as f:
            coils = json.load(f)
    return coils


def _save_day(coils):
    with open(DAY_COUNTER_FILE, "w") as f:
        json.dump(coils, f)


def detect_coach(filename):
    for key, nick in COACH_MAP.items():
        if key in filename:
            return key, nick
    return None, None


def generate_title(filename):
    """Challenge-style SEO titles with Day counter"""
    coach, nickname = detect_coach(filename)
    coils = _get_day()
    day = coils.get(coach, 1) if coach else 1

    if coach:
        templates = [
            f"30天暴汗打卡 Day{day} | {nickname}有氧操 | 零基础燃脂瘦全身",
            f"打工族每日功课 Day{day} | {nickname}带操 | 暴汗解压有氧健身",
            f"风雨无阻坚持Day{day} | {nickname}胭脂虎 | 男女老少燃脂操",
            f"Day{day}零基础有氧操 | {nickname}领操 | 暴汗燃脂打工族必练",
        ]
        title = random.choice(templates)
        coils[coach] = day + 1
        _save_day(coils)
        return title

    return f"胭脂虎健身团 有氧健身操 暴汗燃脂 {os.path.splitext(filename)[0]}"


def add_hook_overlay(video_path, coach_nickname):
    """Burn hook text overlay onto first 4 seconds (PNG+FFmpeg, avoids VideoWriter encoding issues)"""
    import cv2, numpy as np, subprocess, tempfile, shutil
    from PIL import Image, ImageDraw, ImageFont

    if not coach_nickname:
        return video_path
    day = _get_day().get(detect_coach(os.path.basename(video_path))[0], 1)
    hook_text = f"Day {day-1} 暴汗打卡 | 跟{coach_nickname}一起练"

    font_paths = ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc",
                  "C:/Windows/Fonts/simhei.ttf"]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 36)
            break
    if font is None:
        print("[Hook] No Chinese font, skip")
        return video_path

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hook_frames = int(fps * 4)

    # Render frames to temp PNG dir, then ffmpeg encode (avoids cv2.VideoWriter OpenH264 issues)
    tmpdir = Path(tempfile.mkdtemp(prefix="hook_"))
    try:
        for fi in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if fi < hook_frames:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                bbox = draw.textbbox((0, 0), hook_text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                margin_bottom = 80
                tx = (w - tw) // 2
                ty = h - th - margin_bottom
                bar_pad = 12
                overlay = frame.copy()
                cv2.rectangle(overlay, (tx - bar_pad, ty - bar_pad),
                             (tx + tw + bar_pad, ty + th + bar_pad), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw2 = ImageDraw.Draw(pil_frame)
                draw2.text((tx, ty), hook_text, font=font, fill=(255, 255, 255))
                frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(tmpdir / f"f_{fi:06d}.png"), frame)
        cap.release()

        output = video_path.replace(".mp4", "_hook.mp4")
        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        subprocess.run([
            ffmpeg, "-y", "-framerate", str(fps),
            "-i", str(tmpdir / "f_%06d.png"),
            "-i", video_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", "-shortest",
            output
        ], capture_output=True, check=True)
        print(f"[Hook] {os.path.basename(output)}")
        return output
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def generate_thumbnail(video_path, coach_nickname, title):
    """Generate YouTube thumbnail: extract intro frame + bold text overlay"""
    import cv2, numpy as np
    from PIL import Image, ImageDraw, ImageFont

    thumb_path = video_path.replace(".mp4", "_thumb.jpg")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Seek to ~2s for a good intro frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    # Load big bold font
    font_paths = ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"]
    font_lg = None
    for fp in font_paths:
        if os.path.exists(fp):
            font_lg = ImageFont.truetype(fp, int(h * 0.09))
            break
    if not font_lg:
        return None

    # Short punchy text for thumbnail
    short_title = title.split("|")[0].strip() if "|" in title else title[:20]
    lines = [f"{coach_nickname}带操", short_title, "暴汗燃脂·每日打卡"]

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    line_spacing = int(h * 0.04)
    total_th = 0
    line_heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_lg)
        line_heights.append(bbox[3] - bbox[1])
        total_th += line_heights[-1]
    total_th += line_spacing * (len(lines) - 1)

    # Semi-transparent dark bar across bottom third
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    bar_top = h - total_th - int(h * 0.12)
    overlay_draw.rectangle([(0, bar_top - 20), (w, h)], fill=(0, 0, 0, 160))

    # Text lines
    y = bar_top
    for i, line in enumerate(lines):
        color = (255, 220, 50) if i == 0 else (255, 255, 255)
        bbox = draw.textbbox((0, 0), line, font=font_lg)
        tw = bbox[2] - bbox[0]
        cx = (w - tw) // 2
        overlay_draw.text((cx + 1, y + 1), line, font=font_lg, fill=(0, 0, 0, 80))
        overlay_draw.text((cx, y), line, font=font_lg, fill=color)
        y += line_heights[i] + line_spacing

    result = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
    result.convert("RGB").save(thumb_path, "JPEG", quality=92)
    print(f"[Thumbnail] {os.path.basename(thumb_path)}")
    return thumb_path


def make_shorts_clip(video_path, duration=15):
    """Extract center section, crop to vertical 9:16 for Shorts (simple center crop)"""
    shorts_out = video_path.replace(".mp4", "_shorts.mp4")
    probe = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                           "-of", "csv=p=0", video_path], capture_output=True, text=True)
    total = float(probe.stdout.strip())
    start = max(total * 0.3, 1)  # 30% into video
    # Crop center to 9:16 vertical for Shorts
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-vf", "crop=ih*9/16:ih,scale=1080:1920",
        "-c:v", "libx264", "-c:a", "aac",
        shorts_out
    ], capture_output=True, check=True)
    return shorts_out


def _draw_subscribe_cta(frame, current_idx, total_frames, cta_frames):
    """Draw subscribe CTA overlay on a 1080x1920 Shorts frame."""
    from PIL import Image, ImageDraw, ImageFont
    import os as _os
    font_paths = ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc"]
    font = None
    for fp in font_paths:
        if _os.path.exists(fp):
            font = ImageFont.truetype(fp, 48)
            break
    if font is None:
        return frame

    # Progress: 0 = CTA starts, 1 = CTA ends
    remaining = total_frames - current_idx
    progress = 1.0 - (remaining / cta_frames)

    # Slide up from bottom
    offset_y = int((1.0 - min(progress * 2, 1.0)) * 120)

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    lines = ["关注本频道", "岁月不留痕 · 青春不打烊"]
    y = 1920 - 200 + offset_y
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        cx = (1080 - tw) // 2

        # Semi-transparent bg bar
        draw.rectangle([(cx - 20, y - 8), (cx + tw + 20, y + th + 8)],
                       fill=(0, 0, 0, 180))
        # Yellow text
        color = (255, 220, 50) if i == 0 else (255, 255, 255)
        draw.text((cx, y), line, font=font, fill=color)

        y += th + 12

    pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def make_lead_shorts(video_path, video_stem, duration=15):
    """Lead-focused Shorts: track lead person, crop around them, apply beauty"""
    import cv2, numpy as np, tempfile, shutil, json as _json

    # Find keypoints from output dir
    outdir = Path(video_path).parent
    kp_file = outdir / f"{video_stem}_keypoints.json"
    if not kp_file.exists():
        print("[LeadShorts] No keypoints, fallback to center crop")
        return make_shorts_clip(video_path, duration)

    with open(kp_file, encoding="utf-8") as f:
        raw = _json.load(f)
    keypoints = raw.get("keypoints", raw)

    # Find lead person (most frequently tracked person)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Use time-based keypoint lookup (final video may have intro/outro added frames)
    kp_frames = sorted([int(k) for k in keypoints.keys() if keypoints[k]])
    if len(kp_frames) < int(fps * 3):
        print("[LeadShorts] Not enough keypoint data, fallback to center crop")
        return make_shorts_clip(video_path, duration)
    # Pick middle 30% of keypoint range
    kp_start_idx = len(kp_frames) // 3
    kp_end_idx = kp_start_idx + min(int(duration * fps), len(kp_frames) - kp_start_idx - 1)
    target_kp_start = kp_frames[kp_start_idx]
    target_kp_end = kp_frames[min(kp_end_idx, len(kp_frames) - 1)]
    # Map to time in original video
    orig_fps = fps  # keypoints were extracted at original video fps
    clip_start_time = target_kp_start / orig_fps
    clip_duration = (target_kp_end - target_kp_start) / orig_fps

    # Seek in final video to ~30% in
    video_start_time = total_frames / fps * 0.3
    video_seek = max(1, video_start_time)

    # Build lead positions from target keypoint range
    lead_positions = {}
    for fi_str in sorted(keypoints.keys(), key=int):
        fi = int(fi_str)
        if fi < target_kp_start or fi > target_kp_end:
            continue
        kps = keypoints[fi_str]
        if not kps:
            continue
        person = kps[0]
        arr = np.array(person)
        vis = arr[:, 2] > 0.3
        if vis.sum() < 6:
            continue
        hip_x = (arr[11][0] + arr[12][0]) / 2 if (arr[11][2] > 0.3 and arr[12][2] > 0.3) else 0.5
        hip_y = (arr[11][1] + arr[12][1]) / 2 if (arr[11][2] > 0.3 and arr[12][2] > 0.3) else 0.5
        sh_x = (arr[5][0] + arr[6][0]) / 2 if (arr[5][2] > 0.3 and arr[6][2] > 0.3) else hip_x
        sh_y = (arr[5][1] + arr[6][1]) / 2 if (arr[5][2] > 0.3 and arr[6][2] > 0.3) else hip_y - 0.15
        cx = int((sh_x + hip_x) / 2 * w)
        cy = int(((sh_y + hip_y) / 2) * h)
        lead_positions[fi] = (cx, cy)

    if len(lead_positions) < fps * 2:
        print("[LeadShorts] Not enough tracking data, fallback to center crop")
        return make_shorts_clip(video_path, duration)

    # Smooth positions with moving average
    smoothed = {}
    sorted_frames = sorted(lead_positions.keys())
    window = int(fps * 0.5)
    for i, fi in enumerate(sorted_frames):
        w_start = max(0, i - window)
        w_end = min(len(sorted_frames), i + window + 1)
        xs = [lead_positions[sorted_frames[j]][0] for j in range(w_start, w_end)]
        ys = [lead_positions[sorted_frames[j]][1] for j in range(w_start, w_end)]
        smoothed[fi] = (np.mean(xs), np.mean(ys))

    # Crop dimensions: 9:16 vertical, zoom in on coach
    zoom = 1.15  # higher = coach appears larger
    crop_w = int(h * 9 / 16 / zoom)
    crop_h = int(h / zoom)

    print(f"[LeadShorts] Tracking lead person, {len(smoothed)} frames, {fps} fps")

    # Process frames from final video, matching keypoint frame range
    tmpdir = Path(tempfile.mkdtemp(prefix="lead_shorts_"))
    cap = cv2.VideoCapture(video_path)
    # Seek to 30% into video, then step frame by frame matching keypoint data
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_seek * fps))
    out_idx = 0
    kp_keys_sorted = sorted(smoothed.keys())

    while out_idx < len(kp_keys_sorted):
        ret, frame = cap.read()
        if not ret:
            break

        kp_fi = kp_keys_sorted[out_idx]
        if kp_fi in smoothed:
            cx, cy = smoothed[kp_fi]
            # Clamp crop window, center vertically on coach body
            left = int(max(0, min(cx - crop_w // 2, w - crop_w)))
            top = int(max(0, min(cy - crop_h // 2, h - crop_h)))

            cropped = frame[top:top + crop_h, left:left + crop_w].copy()

            # ---- Light beauty: minimal skin smooth + sharpen ----
            # Gentle bilateral (preserve edges)
            smoothed = cv2.bilateralFilter(cropped, 5, 30, 30)
            # Blend 30% smooth with 70% original (keeps texture)
            cropped = cv2.addWeighted(smoothed, 0.3, cropped, 0.7, 0)
            # Subtle sharpen to counter upscale softness
            kernel = np.array([[0, -0.3, 0],
                              [-0.3,  2.2, -0.3],
                              [0, -0.3, 0]])
            cropped = cv2.filter2D(cropped, -1, kernel)
            # Slight face brighten
            face_roi = cropped[:crop_h // 4, :]
            face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            face_hsv[:, :, 2] = np.clip(face_hsv[:, :, 2].astype(float) * 1.03, 0, 255).astype(np.uint8)
            cropped[:crop_h // 4, :] = cv2.cvtColor(face_hsv, cv2.COLOR_HSV2BGR)

            final_frame = cv2.resize(cropped, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
        else:
            left = max(0, (w - crop_w) // 2)
            top = max(0, (h - crop_h) // 2)
            cropped = frame[top:top + crop_h, left:left + crop_w]
            final_frame = cv2.resize(cropped, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)

        # Subscribe CTA on last 3 seconds
        total_out = len(kp_keys_sorted)
        cta_frames = int(fps * 3)
        if out_idx >= total_out - cta_frames and out_idx > 0:
            final_frame = _draw_subscribe_cta(final_frame, out_idx, total_out, cta_frames)

        cv2.imwrite(str(tmpdir / f"f_{out_idx:06d}.png"), final_frame)
        out_idx += 1
        fi += 1

    cap.release()

    if out_idx == 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return make_shorts_clip(video_path, duration)

    # Encode with audio from source
    shorts_out = video_path.replace(".mp4", "_shorts.mp4")
    ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
    actual_duration = out_idx / fps
    subprocess.run([
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", str(tmpdir / "f_%06d.png"),
        "-ss", str(video_seek), "-t", str(actual_duration),
        "-i", video_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        shorts_out
    ], capture_output=True, check=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"[LeadShorts] {os.path.basename(shorts_out)}")
    return shorts_out


def process_one(video_path, upload=False):
    filename = os.path.basename(video_path)
    coach, nickname = detect_coach(filename)
    title = generate_title(filename)

    print(f"\n{'='*50}")
    print(f"  {filename}")
    if coach:
        print(f"  Coach: {coach} ({nickname})")
    print(f"  Title: {title[:70]}")
    print(f"{'='*50}")

    # Run pipeline (DO NOT clear cache - incremental stages depend on intermediates)
    print("[Pipeline] Processing...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
	    env["OPENCV_VIDEOIO_PRIORITY_LIST"] = "MSMF"
    subprocess.run([VENV_PY, PIPELINE, "process", video_path, "-c", CONFIG],
                   check=True, env=env)
    # NOTE: Never delete output/YYYY-MM-DD/ intermediate files.
    # The export stage (07_export.py) chains outputs from prior stages.
    # Clearing cache breaks the reference chain → export fails.

    # Find output
    outdir = os.path.join(os.path.dirname(__file__), "output")
    videos = glob.glob(os.path.join(outdir, "**", "*_final_*.mp4"), recursive=True)
    if not videos:
        videos = glob.glob(os.path.join(outdir, "**", "*.mp4"), recursive=True)
    if not videos:
        print("[ERROR] No output generated")
        return

    latest = max(videos, key=os.path.getmtime)
    print(f"[Output] {os.path.basename(latest)}")

    # Hook overlay
    if nickname:
        try:
            latest = add_hook_overlay(latest, nickname)
        except Exception as e:
            print(f"[Hook] Skip: {e}")

    # Shorts clip (lead-focused with beauty)
    shorts_path = None
    try:
        video_stem = Path(filename).stem
        shorts_path = make_lead_shorts(latest, video_stem)
        print(f"[Shorts] {os.path.basename(shorts_path)}")
    except Exception as e:
        print(f"[Shorts] Skip: {e}, fallback to center crop")
        try:
            shorts_path = make_shorts_clip(latest)
            print(f"[Shorts] {os.path.basename(shorts_path)}")
        except Exception as e2:
            print(f"[Shorts] Center crop also failed: {e2}")

    # Thumbnail
    thumb_path = None
    try:
        thumb_path = generate_thumbnail(latest, nickname, title)
    except Exception as e:
        print(f"[Thumbnail] Skip: {e}")

    # Upload
    if upload:
        print("[YouTube] Uploading main video...")
        sys.path.insert(0, r"F:\wkspace\ComfyUI\custom_nodes")
        from youtube_upload import upload_video

        day = _get_day().get(coach, 1) - 1 if coach else ""
        description = f"""胭脂虎健身团 | {nickname or '教练'}有氧健身操 | Day{day}

零基础也能跳 男女老少不限 风雨无阻坚持 打工族每日功课

跟练打卡挑战：每天一条，30天见效果！今天你打卡了吗？

#胭脂虎健身团 #{'#'+nickname if nickname else ''} #有氧健身操 #打工族健身 #暴汗燃脂 #每日打卡 #30天挑战"""

        tags = ["胭脂虎健身团", "有氧健身操", "打工族健身", "暴汗燃脂", "每日打卡",
                "30天挑战", "零基础健身", "华人健身"]

        ytid = upload_video(latest, title, description=description, tags=tags,
                            privacy="private", channel="fitness",
                            thumbnail_path=thumb_path)
        print(f"[YouTube] https://youtube.com/watch?v={ytid}")

        # Upload Shorts
        if shorts_path and os.path.exists(shorts_path):
            shorts_title = f"Day{day} 15秒暴汗燃脂 {nickname}领操 #Shorts"
            try:
                upload_video(shorts_path, shorts_title, description="15秒暴汗挑战 完整版在频道",
                             tags=tags, privacy="private", channel="fitness")
                print(f"[Shorts] Uploaded")
            except Exception as e:
                print(f"[Shorts] Upload fail: {e}")


def batch(upload=False, limit=None):
    videos = []
    for ext in ["*.mp4", "*.MP4", "*.mov", "*.MOV"]:
        videos.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
    videos = sorted(set(videos))

    if not videos:
        print(f"No videos in {SOURCE_DIR}")
        return

    print(f"Found {len(videos)} videos")
    if limit:
        videos = videos[:limit]

    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}]")
        try:
            process_one(v, upload=upload)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Single video to process")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.video:
        process_one(args.video, upload=args.upload)
    else:
        batch(upload=args.upload, limit=args.limit)

"""独立 Shorts 生成器 v3 — 从源视频裁剪 + 音频 + 大字诗词 + CTA

v3 改进:
  - loudnorm 音频响度标准化
  - 教练个性化文字叠加
  - YouTube Shorts 推荐码率参数
"""
import cv2, numpy as np, json, os, subprocess, tempfile, ctypes, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from lib.coach_profiles import get_shorts_en, get_shorts_cta_en

FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"
FPS = 30

def _short(p):
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 256)
    return buf.value

def get_best_start(keypoints_file, total_frames, duration=15):
    if not os.path.exists(keypoints_file):
        return 10
    with open(keypoints_file, encoding="utf-8") as f:
        data = json.load(f)
    kps = data.get("keypoints", data)
    if not isinstance(kps, dict):
        return 10
    window = int(duration * FPS)
    best_start, best_motion = 10 * FPS, -1
    for s in range(0, total_frames - window, FPS):
        motion = 0
        for f in range(s, min(s + window, total_frames), 3):
            entry = kps.get(str(f))
            if entry and isinstance(entry, list) and entry:
                person = entry[0]
                if isinstance(person[0], list):
                    for i in [5, 6, 7, 8, 9, 10]:  # upper body
                        if i < len(person) and person[i][2] > 0.3:
                            motion += 1
        if motion > best_motion:
            best_motion = motion
            best_start = s
    return max(best_start / FPS, 0.5)

def get_coach_xy(keypoints_file, frame_idx):
    """教练肚脐位置 (0-1 归一化)"""
    try:
        with open(keypoints_file, encoding="utf-8") as f:
            data = json.load(f)
        kps = data.get("keypoints", data)
        entry = kps.get(str(frame_idx))
        if entry and isinstance(entry, list) and entry:
            p = entry[0]
            if isinstance(p[0], list):
                lhip, rhip = p[11], p[12]
                lsh, rsh = p[5], p[6]
                # 用肩膀+髋部中点
                pts = [v for v in [lhip, rhip, lsh, rsh] if v[2] > 0.3]
                if len(pts) >= 2:
                    cx = sum(v[0] for v in pts) / len(pts)
                    cy = sum(v[1] for v in pts) / len(pts)
                    return cx, cy
    except Exception:
        pass
    return 0.5, 0.5

def get_coach_yolo(src_path, start_sec, num_samples=5):
    """用 YOLO 检测人体边界框中心"""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(3)); h = int(cap.get(4))
    if w == 0: cap.release(); return 0.5, 0.5
    xs, ys, cnt = 0, 0, 0
    for i in range(num_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((start_sec + i) * fps))
        ok, frame = cap.read()
        if not ok: continue
        results = model(frame, verbose=False)
        if results[0].boxes:
            # 找最大的人体框
            best = max(results[0].boxes, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
            x1, y1, x2, y2 = best.xyxy[0].tolist()
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            if best.cls[0].item() == 0:  # class 0 = person
                xs += cx; ys += cy; cnt += 1
    cap.release()
    return (xs/cnt, ys/cnt) if cnt > 0 else (0.5, 0.5)


_font_cache = {}
def _get_font(size):
    if size not in _font_cache:
        for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
            if os.path.exists(fp):
                _font_cache[size] = ImageFont.truetype(fp, size)
                return _font_cache[size]
        _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

# 诗词数据统一在 lib/coach_profiles.py 中维护


def make_shorts(src_path, output_dir, keypoints_file, duration=15, audio_src=None):
    cap = cv2.VideoCapture(src_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(3)); src_h = int(cap.get(4))
    total_frames = int(cap.get(7))
    cap.release()
    fps = src_fps if src_fps > 0 else FPS

    # 1. 选段
    start_sec = get_best_start(keypoints_file, total_frames, duration)
    start_sec = min(start_sec, total_frames / fps - duration)
    start_sec = max(start_sec, 0)
    print(f"  最佳段: {start_sec:.1f}s")

    # 2. YOLO 教练位置
    coach_x, coach_y = get_coach_yolo(src_path, start_sec)
    print(f"  YOLO教练: ({coach_x:.2f}, {coach_y:.2f})")

    # 3. 裁剪: 9:16, 教练居中
    crop_h = src_h
    crop_w = int(crop_h * 9 / 16)
    coach_px = int(coach_x * src_w)
    crop_x = coach_px - crop_w // 2
    if crop_x < 0: crop_x = 0
    if crop_x + crop_w > src_w: crop_x = src_w - crop_w
    crop_y = 0
    print(f"  裁剪: {crop_w}x{crop_h} @ x={crop_x}")

    # 4. FFmpeg 裁剪+美颜+音频
    raw_out = os.path.join(output_dir, "_shorts_raw.mp4")
    audio_out = os.path.join(output_dir, "_shorts_audio.aac")

    # 提取音频（audio_src 指定有音频的源，如原始视频）
    audio_source = audio_src or src_path
    subprocess.run([
        FFMPEG, "-y", "-ss", str(start_sec), "-t", str(duration),
        "-i", audio_source, "-vn", "-c:a", "aac", "-b:a", "128k",
        audio_out
    ], capture_output=True, timeout=30)

    has_audio = os.path.exists(audio_out) and os.path.getsize(audio_out) > 1000

    cmd1 = [
        FFMPEG, "-y", "-ss", str(start_sec), "-t", str(duration),
        "-i", src_path,
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=1080:1920:flags=lanczos,smartblur=1.2:0.6:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        raw_out
    ]
    r1 = subprocess.run(cmd1, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=60)
    if r1.returncode != 0:
        print(f"  裁剪失败: {r1.stderr[-200:]}")
        return None

    # 5. 准备教练双语文字
    en_data = get_shorts_en(src_path)
    title_en = en_data["title"]
    subtitle_en = en_data["subtitle"]
    cta_lines = get_shorts_cta_en()

    # 6. PIL 叠加文字
    tmpdir = Path(tempfile.mkdtemp(prefix="st_"))
    short_p = _short(str(tmpdir))
    cap2 = cv2.VideoCapture(raw_out)
    fi = 0

    font_title = _get_font(90)
    font_sub = _get_font(48)

    while True:
        ok, frame = cap2.read()
        if not ok:
            break
        t = fi / fps
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 英文标题 + 双语教练名 — 顶部，2-22秒
        if 2.0 <= t <= 22.0:
            alpha = 255
            if t < 3.0:
                alpha = int(255 * (t - 2.0))
            elif t > 20.0:
                alpha = int(255 * (22.0 - t))

            # 英文标题（大字黄色）
            bbox = draw.textbbox((0, 0), title_en, font=font_title)
            tw = bbox[2] - bbox[0]
            tx = (1080 - tw) // 2
            draw.text((tx + 3, 62), title_en, font=font_title,
                     fill=(0, 0, 0, min(255, alpha)))
            draw.text((tx, 60), title_en, font=font_title,
                     fill=(255, 255, 0, min(255, alpha)))

            # 教练副标题（小字白色，含中文名 IP）
            bbox2 = draw.textbbox((0, 0), subtitle_en, font=font_sub)
            tw2 = bbox2[2] - bbox2[0]
            tx2 = (1080 - tw2) // 2
            draw.text((tx2 + 2, 162), subtitle_en, font=font_sub,
                     fill=(0, 0, 0, min(255, alpha)))
            draw.text((tx2, 160), subtitle_en, font=font_sub,
                     fill=(255, 255, 255, min(255, alpha)))

        # CTA — 双语，22秒后显示
        if t >= 22.0 and t < 30.0:
            alpha_cta = 255
            if t < 23.0:
                alpha_cta = int(255 * (t - 22.0))
            y = 1920 - 280
            for i, line in enumerate(cta_lines):
                sz = 72 if i == 0 else 60
                f = _get_font(sz)
                bbox = draw.textbbox((0, 0), line, font=f)
                tw = bbox[2] - bbox[0]
                tx = (1080 - tw) // 2
                color = (255, 255, 100) if i == 0 else (255, 255, 255)
                draw.text((tx + 2, y + 2), line, font=f, fill=(0, 0, 0, min(255, alpha_cta)))
                draw.text((tx, y), line, font=f, fill=(*color, min(255, alpha_cta)))
                y += 90

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{short_p}/f_{fi:06d}.png", frame)
        fi += 1
    cap2.release()
    print(f"  文字叠加: {fi} 帧")

    # 7. 编码最终视频（+音频，YouTube Shorts 推荐参数）
    out_path = os.path.join(output_dir, f"{Path(src_path).stem}_shorts_v2.mp4")
    cmd2 = [
        FFMPEG, "-y", "-framerate", str(fps),
        "-i", f"{short_p}/f_%06d.png",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-maxrate", "10M", "-bufsize", "15M",
        "-pix_fmt", "yuv420p", "-an",
        out_path
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=120)

    shutil.rmtree(tmpdir, ignore_errors=True)
    os.remove(raw_out)

    if r2.returncode != 0:
        print(f"  编码失败: {r2.stderr[-200:]}")
        return None

    # 8. 混入音频 + loudnorm 响度标准化
    if has_audio:
        mixed = out_path.replace(".mp4", "_audio.mp4")
        subprocess.run([
            FFMPEG, "-y", "-i", out_path, "-i", audio_out,
            "-c:v", "copy",
            "-af", "loudnorm=I=-14:LRA=11:TP=-1.5",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", mixed
        ], capture_output=True, timeout=60)
        os.remove(out_path)
        os.rename(mixed, out_path)
    if os.path.exists(audio_out):
        os.remove(audio_out)

    return out_path


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\18091\Desktop\短视频素材\艳玲1.mp4"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else r"F:\wkspace\fitness-video-pipeline\output\2026-05-21"
    kp_file = os.path.join(out_dir, "艳玲1_keypoints.json")

    result = make_shorts(src, out_dir, kp_file, duration=30)
    if result:
        print(f"\n完成: {result}")
        # 如需上传：
        # from lib.upload_utils import upload_video as up
        # ytid = up(result, "细柳营教练名 | 暴汗燃脂30秒 #Shorts",
        #            "完整版在频道\n细柳营·胭脂虎", tags=[...], privacy="public")
        # print(f"上传: https://youtube.com/watch?v={ytid}")
    else:
        print("失败")

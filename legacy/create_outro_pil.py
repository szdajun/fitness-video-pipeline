"""用 PIL 为横版片尾绘制文字，然后合并生成最终视频"""
import subprocess, shutil, os
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
output_dir = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-14")

def get_font(size):
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttc"]:
        try:
            return ImageFont.truetype(fp, size)
        except:
            pass
    return ImageFont.load_default()

def draw_text_pil(frame, cta_text="关注不迷路", sub_text="点击关注"):
    """在帧上绘制片尾文字"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    # 底部完全不透明的黑色衬底（y=60% 到 100%）
    substrate_top = int(h * 0.60)
    draw.rectangle([(0, substrate_top), (w, h)], fill=(0, 0, 0))

    # CTA 大字
    font_lg = get_font(int(h * 0.10))
    bbox = draw.textbbox((0, 0), cta_text, font=font_lg)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (w - tw) // 2
    ty = substrate_top + int((h - substrate_top - th) * 0.25)
    draw.text((cx, ty), cta_text, font=font_lg, fill=(255, 255, 50))

    # 小字
    font_sm = get_font(int(h * 0.06))
    bbox = draw.textbbox((0, 0), sub_text, font=font_sm)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (w - tw) // 2
    ty2 = substrate_top + int((h - substrate_top - th) * 0.65)
    draw.text((cx, ty2), sub_text, font=font_sm, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_outro_video(video_path, output_path, cta_text, sub_text, fps=30.0):
    """从视频截取最后5秒，叠加文字，输出新视频"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = 5.0
    start_frame = max(0, int(total_frames - duration * fps))
    cap.release()

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 用 FFmpeg 直接写入（avc1 codec）
    temp_dir = output_dir / "_outro_frames"
    temp_dir.mkdir(exist_ok=True)

    for i in range(int(duration * fps)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_with_text = draw_text_pil(frame, cta_text, sub_text)
        cv2.imwrite(str(temp_dir / f"f{i:04d}.jpg"), frame_with_text, [cv2.IMWRITE_JPEG_QUALITY, 95])

    cap.release()

    # 用 FFmpeg 把图片序列编码为视频
    outro_path = output_path
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(temp_dir / "f%04d.jpg"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-frames:v", str(int(duration * fps)),
        "-an", str(outro_path)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"FFmpeg encode failed: {r.stderr[-200:]}")
        return False

    # 清理临时文件
    for f in temp_dir.glob("*.jpg"):
        f.unlink()
    temp_dir.rmdir()
    return True

# Step 1: 创建片尾（含文字）
print("创建片尾（含文字）...")
video = output_dir / "丽丽1_energybar.mp4"
outro = output_dir / "丽丽1_outro_16x9.mp4"
ok = create_outro_video(video, outro, "关注不迷路", "点击关注", fps=30.0)
if not ok:
    print("片尾创建失败")
    exit(1)

# 验证
r = subprocess.run(
    [ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames,width,height", "-select_streams", "v:0", "-of", "csv=p=0", str(outro)],
    capture_output=True, text=True
)
print(f"片尾: {r.stdout.strip()}")

# Step 2: 合并视频
print("\n合并视频...")
intro = output_dir / "丽丽1_intro.mp4"
combined = output_dir / "_combined_16x9.mp4"
concat_list = output_dir / "concat_16x9.txt"
with open(concat_list, "w", encoding="utf-8", newline="\n") as f:
    for p in [intro, video, outro]:
        f.write(f"file '{p.as_posix()}'\n")

cmd1 = [
    ffmpeg, "-y",
    "-f", "concat", "-safe", "0",
    "-i", str(concat_list),
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an", str(combined)
]
r2 = subprocess.run(cmd1, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r2.returncode != 0:
    print(f"合并失败: {r2.stderr[-200:]}")
    exit(1)
print(f"合并: {combined.stat().st_size/1024/1024:.1f}MB")

# Step 3: 添加音频
print("\n添加音频...")
audio_orig = "C:/Users/18091/Desktop/短视频素材/丽丽1.mp4"
total_sec = 4.0 + 82.4 + 5.0  # 91.4s
fade_start = total_sec - 3.0

final = output_dir / "丽丽1_full_16x9.mp4"
cmd2 = [
    ffmpeg, "-y",
    "-i", str(combined),
    "-i", str(audio_orig),
    "-map", "0:v:0", "-map", "1:a:0?",
    "-af", f"afade=type=out:st={fade_start:.2f}:d=3.0",
    "-vf", f"fade=t=out:st={fade_start:.2f}:d=3.0",
    "-c:v", "libx264", "-preset", "fast", "-crf", "26",
    "-c:a", "aac", "-b:a", "96k",
    str(final)
]
r3 = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r3.returncode != 0:
    print(f"音频合并失败: {r3.stderr[-300:]}")
else:
    print(f"\n完成: {final.name} ({final.stat().st_size/1024/1024:.1f}MB)")

    r4 = subprocess.run(
        [ffmpeg, "-v", "error", "-show_entries", "format=duration,size", "-show_entries", "stream=nb_frames,width,height", "-of", "csv=p=0", str(final)],
        capture_output=True, text=True
    )
    print(f"验证: {r4.stdout.strip()}")
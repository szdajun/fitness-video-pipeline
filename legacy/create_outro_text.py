"""用 PIL 绘制文字在片尾帧上，然后用 FFmpeg 编码"""
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

def draw_text_pil(frame, cta="关注不迷路", sub="点击关注"):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    # 底部完全不透明黑色衬底（y=60% 到 100%）
    substrate_top = int(h * 0.60)
    draw.rectangle([(0, substrate_top), (w, h)], fill=(0, 0, 0))

    # CTA 大字（亮黄色）
    font_lg = get_font(int(h * 0.10))
    bbox = draw.textbbox((0, 0), cta, font=font_lg)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (w - tw) // 2
    ty = substrate_top + int((h - substrate_top - th) * 0.25)
    draw.text((cx, ty), cta, font=font_lg, fill=(255, 255, 50))

    # 小字（白色）
    font_sm = get_font(int(h * 0.06))
    bbox = draw.textbbox((0, 0), sub, font=font_sm)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (w - tw) // 2
    ty2 = substrate_top + int((h - substrate_top - th) * 0.65)
    draw.text((cx, ty2), sub, font=font_sm, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Step 1: 从 energybar 提取最后 5 秒的帧，用 PIL 添加文字，再编码为视频
print("Step 1: 生成带文字的片尾...")
energybar = str(output_dir / "丽丽1_energybar.mp4")
outro_text = str(output_dir / "outro_with_text.mp4")

# 提取帧
temp_jpg_dir = output_dir / "_outro_jpgs"
temp_jpg_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(energybar)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = max(0, int(total - 5 * fps))
cap.release()

cap = cv2.VideoCapture(energybar)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frame_count = 0
max_count = int(5 * fps)

print(f"  提取帧: 从帧{start_frame}开始，共{max_count}帧...")
for i in range(max_count):
    ret, frame = cap.read()
    if not ret:
        break
    frame_with_text = draw_text_pil(frame)
    cv2.imwrite(str(temp_jpg_dir / f"f{i:04d}.jpg"), frame_with_text, [cv2.IMWRITE_JPEG_QUALITY, 95])
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"  已处理 {frame_count}/{max_count} 帧")

cap.release()
print(f"  已保存 {frame_count} 帧到临时目录")

# 用 FFmpeg 编码为视频
cmd_encode = [
    ffmpeg, "-y",
    "-framerate", str(fps),
    "-i", str(temp_jpg_dir / "f%04d.jpg"),
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-frames:v", str(frame_count),
    "-an", outro_text
]
r = subprocess.run(cmd_encode, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print(f"编码失败: {r.stderr[-200:]}")
else:
    print(f"  片尾(含文字)已生成: {outro_text}")
    r2 = subprocess.run([ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames", "-select_streams", "v:0", "-of", "csv=p=0", outro_text], capture_output=True, text=True)
    print(f"  帧数: {r2.stdout.strip()}")

# 清理临时文件
for f in temp_jpg_dir.glob("*.jpg"):
    f.unlink()
temp_jpg_dir.rmdir()

# Step 2: 合并视频（filter_complex concat）
print("\nStep 2: 合并视频...")
intro = str(output_dir / "丽丽1_intro.mp4")
main_part = str(output_dir / "main_part.mp4")
combined = str(output_dir / "_combined_16x9.mp4")

cmd_concat = [
    ffmpeg, "-y",
    "-i", intro,
    "-i", main_part,
    "-i", outro_text,
    "-filter_complex",
    "[0:v][1:v][2:v]concat=n=3:v=1:a=0[outv]",
    "-map", "[outv]",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an", combined
]
r3 = subprocess.run(cmd_concat, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r3.returncode != 0:
    print(f"合并失败: {r3.stderr[-300:]}")
else:
    r4 = subprocess.run([ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames,duration", "-select_streams", "v:0", "-of", "csv=p=0", combined], capture_output=True, text=True)
    print(f"  合并后: {r4.stdout.strip()}")

    # Step 3: 添加音频
    print("\nStep 3: 添加音频...")
    audio_orig = "C:/Users/18091/Desktop/短视频素材/丽丽1.mp4"
    total_sec = 4.0 + 73.4 + 5.0
    fade_start = total_sec - 3.0
    final = str(output_dir / "丽丽1_full_16x9.mp4")

    cmd_audio = [
        ffmpeg, "-y",
        "-i", combined,
        "-i", audio_orig,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-af", f"afade=type=out:st={fade_start:.2f}:d=3.0",
        "-vf", f"fade=t=out:st={fade_start:.2f}:d=3.0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "26",
        "-c:a", "aac", "-b:a", "96k",
        final
    ]
    r5 = subprocess.run(cmd_audio, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r5.returncode != 0:
        print(f"音频失败: {r5.stderr[-300:]}")
    else:
        size = os.path.getsize(final) / 1024 / 1024
        print(f"\n完成: 丽丽1_full_16x9.mp4 ({size:.1f}MB)")
        r6 = subprocess.run([ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames,duration,width,height", "-select_streams", "v:0", "-of", "csv=p=0", final], capture_output=True, text=True)
        print(f"验证: {r6.stdout.strip()}")
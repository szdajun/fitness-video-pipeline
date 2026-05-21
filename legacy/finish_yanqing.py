"""从 warped 文件继续：Ken Burns dual + 色彩 + 导出"""
import cv2, subprocess, math, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ffmpeg = "C:/Users/18091/ffmpeg/ffmpeg.exe"
output_dir = Path("output")
warped = Path("output/yanqing4_h2v_warped.mp4")
src = Path("C:/Users/18091/Desktop/短视频素材/艳青4.mp4")

# 读取视频信息
cap = cv2.VideoCapture(str(warped))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
total_time = total / fps
print(f"输入: {w}x{h}, {total}帧, {fps:.1f}fps, {total_time:.1f}秒")

# ============ Step1: Ken Burns dual (OpenCV) ============
target_h = h
target_w = int(h * 9 / 16)
if target_w % 2:
    target_w -= 1
print(f"\n[Step1] Ken Burns dual -> {target_w}x{target_h}")

cycle_s = 8.0
kb_out = output_dir / "temp_kb.mp4"
writer = cv2.VideoWriter(str(kb_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

cap_in = cv2.VideoCapture(str(warped))
frame_idx = 0
while True:
    ret, frame = cap_in.read()
    if not ret:
        break
    H, W = frame.shape[:2]
    t = frame_idx / total * total_time
    scene_factor = 0.5 - 0.5 * math.cos(2 * math.pi * t / cycle_s)
    zoom = 1.0 + 0.1 * scene_factor
    pan_x = scene_factor * 6 * math.sin(2.5 * math.pi * t / cycle_s)
    pan_y = scene_factor * 4 * math.sin(2.0 * math.pi * t / cycle_s)
    new_w = int(W * zoom)
    new_h = int(H * zoom)
    scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    cx = int(new_w / 2 + pan_x - target_w / 2)
    cy = int(new_h / 2 + pan_y - target_h / 2)
    cx = max(0, min(cx, new_w - target_w))
    cy = max(0, min(cy, new_h - target_h))
    cropped = scaled[cy:cy + target_h, cx:cx + target_w]
    writer.write(cropped)
    frame_idx += 1
    if frame_idx % 200 == 0:
        print(f"    {frame_idx / total * 100:.0f}% ({frame_idx}/{total})")

cap_in.release()
writer.release()
print(f"Ken Burns 完成: {kb_out.stat().st_size // 1024 // 1024}MB")

# ============ Step2: 色彩增强 (FFmpeg) ============
print("\n[Step2] 色彩增强...")
color_out = output_dir / "temp_color.mp4"
r = subprocess.run([
    ffmpeg, "-y", "-i", str(kb_out),
    "-vf", "eq=brightness=0.03:contrast=1.05:saturation=1.10,vibrance=0.8",
    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
    "-pix_fmt", "yuv420p", "-an", str(color_out)
], capture_output=True, text=True, errors="replace")
if r.returncode != 0:
    print(f"色彩失败: {r.stderr[-200:]}")
else:
    print(f"色彩完成: {color_out.stat().st_size // 1024 // 1024}MB")

    # ============ Step3: 最终导出 (合并音频) ============
    print("\n[Step3] 导出最终视频...")
    final_out = output_dir / "艳青4_sexy_final.mp4"
    r = subprocess.run([
        ffmpeg, "-y",
        "-i", str(color_out),
        "-i", str(src),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        str(final_out)
    ], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"导出失败: {r.stderr[-300:]}")
    else:
        cap2 = cv2.VideoCapture(str(final_out))
        frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap2.release()
        size = final_out.stat().st_size // 1024 // 1024
        print(f"\n完成: {final_out.name} ({size}MB, {frames}帧, {frames/fps:.1f}秒)")

    if color_out.exists():
        color_out.unlink()

if kb_out.exists():
    kb_out.unlink()
print("清理完成")
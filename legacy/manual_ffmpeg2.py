"""直接用 FFmpeg 完成 pipeline (Windows 兼容)"""
import subprocess, shutil, os, cv2
from pathlib import Path

output_dir = Path("output")

# 找最新 h2v_warped 文件
files = list(output_dir.glob("*h2v_warped.mp4"))
files.sort(key=os.path.getmtime)
new_warped = files[-1]
print(f"warped: {new_warped} ({os.path.getsize(new_warped)//1024//1024}MB)")

ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"

# 获取 warped 视频信息
cap = cv2.VideoCapture(str(new_warped))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"warped视频: {w}x{h}, {total}帧, fps={fps:.2f}")

# Windows: FFmpeg 需要正斜杠路径，避免中文路径问题
def to_fwd(p):
    return str(p).replace("\\", "/")

# ========== Stage 4: ken_burns (直接 scale，无运镜) ==========
print("\n[Stage 4] ken_burns (pass-through)...")
kb_out = output_dir / "temp_kenburns.mp4"
if kb_out.exists():
    kb_out.unlink()
cmd = [
    ffmpeg, "-y",
    "-i", to_fwd(new_warped),
    "-vf", f"scale={w}:{h}",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an", to_fwd(kb_out)
]
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
if result.returncode != 0:
    print(f"ken_burns 失败: {result.stderr[-200:]}")
else:
    print(f"ken_burns OK: {os.path.getsize(kb_out)//1024//1024}MB")

# ========== Stage 5: color_grade ==========
print("\n[Stage 5] color_grade...")
color_out = output_dir / "temp_color.mp4"
if color_out.exists():
    color_out.unlink()
cmd = [
    ffmpeg, "-y",
    "-i", to_fwd(kb_out),
    "-vf", "eq=brightness=0.05:contrast=1.05:saturation=1.05",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an", to_fwd(color_out)
]
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
if result.returncode != 0:
    print(f"color 失败: {result.stderr[-200:]}")
else:
    print(f"color OK: {os.path.getsize(color_out)//1024//1024}MB")

# ========== Stage 6: export with audio ==========
print("\n[Stage 6] export...")
final_out = output_dir / "艳青和丽丽_final_new2.mp4"
input_video = Path("C:/Users/18091/Desktop/短视频素材/艳青和丽丽.mp4")

cmd = [
    ffmpeg, "-y",
    "-i", to_fwd(color_out),
    "-i", to_fwd(input_video),
    "-map", "0:v:0", "-map", "1:a:0?",
    "-vf", "scale=1080:1920:flags=lanczos",
    "-c:v", "libx264", "-preset", "medium", "-crf", "23",
    "-c:a", "aac", "-b:a", "128k", "-shortest",
    to_fwd(final_out)
]
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
if result.returncode != 0:
    print(f"export 失败: {result.stderr[-300:]}")
else:
    size = os.path.getsize(final_out) // 1024 // 1024
    cap = cv2.VideoCapture(str(final_out))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_f = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"export OK: {final_out.name} ({size}MB, {total_f}帧, fps={fps_f:.2f}, {total_f/fps_f:.1f}秒)")
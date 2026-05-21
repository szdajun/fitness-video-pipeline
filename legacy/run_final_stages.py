"""用 FFmpeg 直接完成 pipeline 剩余阶段（跳过 body_warp 和 ken_burns）"""
import subprocess, shutil, os, cv2
from pathlib import Path

output_dir = Path("output")

# 找最新 h2v 文件
h2v_files = list(output_dir.glob("*_h2v.mp4"))
h2v_files.sort(key=os.path.getmtime)
new_h2v = h2v_files[-1]
print(f"h2v: {new_h2v.name} ({os.path.getsize(new_h2v)//1024//1024}MB)")

ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"

# 获取 h2v 视频信息
cap = cv2.VideoCapture(str(new_h2v))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"h2v视频: {w}x{h}, {total}帧, fps={fps:.2f}")

def to_fwd(p):
    return str(p).replace("\\", "/")

# ========== Stage: color_grade ==========
print("\n[Stage] color_grade...")
color_out = output_dir / "temp_color.mp4"
if color_out.exists():
    color_out.unlink()
cmd = [
    ffmpeg, "-y",
    "-i", to_fwd(new_h2v),
    "-vf", "eq=brightness=0.05:contrast=1.05:saturation=1.05",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an", to_fwd(color_out)
]
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
if result.returncode != 0:
    print(f"color 失败: {result.stderr[-200:]}")
else:
    print(f"color OK: {os.path.getsize(color_out)//1024//1024}MB")

# ========== Stage: export with audio ==========
print("\n[Stage] export...")
final_out = output_dir / "艳青和丽丽_final_new3.mp4"
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

# 清理临时文件
if color_out.exists():
    color_out.unlink()
    print(f"\n已清理临时文件")

"""手动生成横版最终视频"""
import subprocess, shutil
from pathlib import Path

ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
output_dir = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-14")

intro = output_dir / "丽丽1_intro.mp4"
main_video = output_dir / "丽丽1_energybar.mp4"
outro = output_dir / "丽丽1_outro.mp4"
audio_orig = "C:/Users/18091/Desktop/短视频素材/丽丽1.mp4"
concat_list = output_dir / "concat_16x9.txt"

# 创建 concat list
with open(concat_list, "w", encoding="utf-8", newline="\n") as f:
    for p in [intro, main_video, outro]:
        f.write(f"file '{p.as_posix()}'\n")

# Step 1: 合并视频
combined = output_dir / "_combined_16x9.mp4"
cmd1 = [
    ffmpeg, "-y",
    "-f", "concat", "-safe", "0",
    "-i", str(concat_list),
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an", str(combined)
]
print("合并视频...")
r1 = subprocess.run(cmd1, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r1.returncode != 0:
    print(f"FAILED: {r1.stderr[-200:]}")
else:
    print(f"合并成功: {combined.stat().st_size/1024/1024:.1f}MB")

    # 手动计算总时长：intro(4s) + energybar(82.4s) + outro(5s) = 91.4s
    total_sec = 4.0 + 82.4 + 5.0
    fade_start = total_sec - 3.0
    print(f"总时长: {total_sec:.1f}s, 淡出: {fade_start:.1f}s起")

    # Step 2: 添加音频
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
    print("添加音频...")
    r2 = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r2.returncode != 0:
        print(f"FAILED: {r2.stderr[-300:]}")
    else:
        print(f"\n完成: {final.name} ({final.stat().st_size/1024/1024:.1f}MB)")
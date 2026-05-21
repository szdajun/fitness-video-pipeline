"""手动合并：intro + energybar + outro_v2 生成最终视频"""
import subprocess, shutil, os
from pathlib import Path

ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
output_dir = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-14")

# 文件路径
intro = output_dir / "丽丽1_intro.mp4"
main_video = output_dir / "丽丽1_energybar.mp4"
outro = output_dir / "丽丽1_outro_v2.mp4"
audio_orig = "C:/Users/18091/Desktop/短视频素材/丽丽1.mp4"

# 检查文件
for f in [intro, main_video, outro, Path(audio_orig)]:
    exists = f.exists()
    size = f.stat().st_size / 1024 / 1024 if exists else 0
    print(f"  {f.name}: {'OK' if exists else 'MISSING'} ({size:.1f}MB)")

# 创建 concat list
concat_list = output_dir / "concat_final.txt"
with open(concat_list, "w", encoding="utf-8", newline="\n") as f:
    for p in [intro, main_video, outro]:
        f.write(f"file '{p.as_posix()}'\n")

print(f"\nConcat list:\n{open(concat_list).read()}")

# Step 1: 合并视频（无音频）
combined = output_dir / "_combined_final.mp4"
cmd = [
    ffmpeg, "-y",
    "-f", "concat", "-safe", "0",
    "-i", str(concat_list),
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an",
    str(combined)
]
r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print(f"Video concat failed: {r.stderr[-300:]}")
else:
    print(f"Video concat OK: {combined} ({combined.stat().st_size/1024/1024:.1f}MB)")

# Step 2: 添加音频（带淡出）
total_frames = 2472 + 120 + 150  # energybar + intro + outro
fps = 30.0
total_sec = total_frames / fps
fade_start = total_sec - 2.0  # 最后2秒淡出
audio_fade = f"afade=type=out:st={fade_start:.2f}:d=2.0"

final = output_dir / "丽丽1_full_9x16_v2.mp4"
cmd2 = [
    ffmpeg, "-y",
    "-i", str(combined),
    "-i", str(audio_orig),
    "-map", "0:v:0", "-map", "1:a:0?",
    "-af", audio_fade,
    "-vf", f"fade=t=out:st={fade_start:.2f}:d=2.0",
    "-c:v", "libx264", "-preset", "fast", "-crf", "26",
    "-c:a", "aac", "-b:a", "96k",
    str(final)
]
r2 = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r2.returncode != 0:
    print(f"Final encode failed: {r2.stderr[-300:]}")
else:
    print(f"\n最终视频: {final} ({final.stat().st_size/1024/1024:.1f}MB)")

    # 验证
    r3 = subprocess.run(
        [ffmpeg, "-v", "error", "-show_entries", "format=duration,size",
         "-show_entries", "stream=nb_frames", "-of", "json", str(final)],
        capture_output=True, text=True
    )
    import json
    try:
        data = json.loads(r3.stdout)
        s = data["streams"][0]
        f = data["format"]
        print(f"验证: {s['width']}x{s['height']}, {s['nb_frames']}帧, {float(f['duration']):.2f}秒")
    except:
        print(f"验证输出: {r3.stdout[:200]}")
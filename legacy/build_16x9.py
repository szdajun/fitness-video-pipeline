"""正确构建横版最终视频：intro(4s) + energybar(去掉前4s和后5s) + outro(最后5s)"""
import subprocess, shutil
from pathlib import Path

ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
output_dir = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-14")
audio_orig = "C:/Users/18091/Desktop/短视频素材/丽丽1.mp4"

energybar = output_dir / "丽丽1_energybar.mp4"
intro = output_dir / "丽丽1_intro.mp4"

# Step 1: 提取主体（前4秒到77.4秒，即去掉intro时段和outro时段）
# intro可能是任意位置，所以主体 = 去掉前4秒和后5秒的energybar
print("Step 1: 提取主体（去掉前4s和后5s）...")
main_part = output_dir / "main_part.mp4"
r = subprocess.run([
    ffmpeg, "-y",
    "-i", str(energybar),
    "-ss", "4.0",           # 跳过前4秒（intro）
    "-t", "73.4",            # 到77.4秒为止（保留后5秒给outro）
    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
    str(main_part)
], capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print(f"FAILED: {r.stderr[-200:]}")
else:
    r2 = subprocess.run([ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames,duration", "-select_streams", "v:0", "-of", "csv=p=0", str(main_part)], capture_output=True, text=True)
    print(f"主体: {r2.stdout.strip()}")

# Step 2: 提取片尾（最后5秒）
print("\nStep 2: 提取片尾（最后5秒）...")
outro_raw = output_dir / "outro_raw.mp4"
r = subprocess.run([
    ffmpeg, "-y",
    "-i", str(energybar),
    "-ss", "77.4", "-t", "5.0",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
    str(outro_raw)
], capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print(f"FAILED: {r.stderr[-200:]}")
else:
    r2 = subprocess.run([ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames", "-select_streams", "v:0", "-of", "csv=p=0", str(outro_raw)], capture_output=True, text=True)
    print(f"片尾: {r2.stdout.strip()}")

# Step 3: 合并 intro + main_part + outro_raw
print("\nStep 3: 合并视频...")
combined = output_dir / "_combined_16x9.mp4"
concat_list = output_dir / "concat_manual.txt"
with open(concat_list, "w", encoding="utf-8", newline="\n") as f:
    f.write(f"file '{intro.as_posix()}'\n")
    f.write(f"file '{main_part.as_posix()}'\n")
    f.write(f"file '{outro_raw.as_posix()}'\n")

r = subprocess.run([
    ffmpeg, "-y",
    "-f", "concat", "-safe", "0",
    "-i", str(concat_list),
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an", str(combined)
], capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print(f"合并失败: {r.stderr[-200:]}")
else:
    r2 = subprocess.run([ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames,duration", "-select_streams", "v:0", "-of", "csv=p=0", str(combined)], capture_output=True, text=True)
    print(f"合并后: {r2.stdout.strip()}")

    # Step 4: 添加音频
    print("\nStep 4: 添加音频...")
    total_sec = 4.0 + 73.4 + 5.0  # 82.4s
    fade_start = total_sec - 3.0

    final = output_dir / "丽丽1_full_16x9.mp4"
    r2 = subprocess.run([
        ffmpeg, "-y",
        "-i", str(combined),
        "-i", str(audio_orig),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-af", f"afade=type=out:st={fade_start:.2f}:d=3.0",
        "-vf", f"fade=t=out:st={fade_start:.2f}:d=3.0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "26",
        "-c:a", "aac", "-b:a", "96k",
        str(final)
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r2.returncode != 0:
        print(f"音频失败: {r2.stderr[-300:]}")
    else:
        final_size = final.stat().st_size / 1024 / 1024
        print(f"\n完成: {final.name} ({final_size:.1f}MB)")
        r3 = subprocess.run([ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames,width,height", "-select_streams", "v:0", "-of", "csv=p=0", str(final)], capture_output=True, text=True)
        print(f"验证: {r3.stdout.strip()}")
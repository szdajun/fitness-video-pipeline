"""用 FFmpeg concat filter 正确合并三个视频片段"""
import subprocess, shutil
from pathlib import Path

ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
output_dir = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-14")
audio_orig = "C:/Users/18091/Desktop/短视频素材/丽丽1.mp4"

intro = str(output_dir / "丽丽1_intro.mp4")
main_part = str(output_dir / "main_part.mp4")
outro = str(output_dir / "outro_raw.mp4")
combined = str(output_dir / "_combined_16x9.mp4")

# 用 filter_complex concat（可处理时间戳不连续的情况）
# intro: 4s, main_part: 73.4s, outro: 5s, 总计 82.4s
cmd = [
    ffmpeg, "-y",
    "-i", intro,
    "-i", main_part,
    "-i", outro,
    "-filter_complex",
    "[0:v][1:v][2:v]concat=n=3:v=1:a=0[outv]",
    "-map", "[outv]",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an",
    combined
]
print("合并视频（filter_complex）...")
r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print(f"FAILED: {r.stderr[-300:]}")
else:
    r2 = subprocess.run(
        [ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames,duration,width,height", "-select_streams", "v:0", "-of", "csv=p=0", combined],
        capture_output=True, text=True
    )
    print(f"合并后: {r2.stdout.strip()}")

    # 添加音频
    total_sec = 4.0 + 73.4 + 5.0  # 82.4s
    fade_start = total_sec - 3.0
    final = str(output_dir / "丽丽1_full_16x9.mp4")

    cmd2 = [
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
    print("\n添加音频...")
    r3 = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r3.returncode != 0:
        print(f"音频失败: {r3.stderr[-300:]}")
    else:
        import os
        size = os.path.getsize(final) / 1024 / 1024
        print(f"\n完成: 丽丽1_full_16x9.mp4 ({size:.1f}MB)")
        r4 = subprocess.run(
            [ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames,duration,width,height", "-select_streams", "v:0", "-of", "csv=p=0", final],
            capture_output=True, text=True
        )
        print(f"验证: {r4.stdout.strip()}")
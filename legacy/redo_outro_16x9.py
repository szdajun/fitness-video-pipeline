"""重新生成横版片尾（含可见文字），然后合并最终视频"""
import subprocess, shutil
from pathlib import Path

ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
output_dir = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-14")

video = output_dir / "丽丽1_energybar.mp4"
outro = output_dir / "丽丽1_outro_16x9.mp4"

# 用 FFmpeg drawbox + drawtext 生成片尾
# 底部覆盖黑色衬底 + 黄色大字 CTA + 白色小字"点击关注"
cmd_outro = [
    ffmpeg, "-y",
    "-ss", "77.4",    # 从77.4s开始，共5秒（到82.4s结束）
    "-i", str(video),
    "-t", "5.0",
    # 底部50%区域填充纯黑（完全不透明）
    "-vf", (
        "drawbox=0:540:1280:540:color=black:t=fill,"
        "drawtext=text='关注不迷路':fontsize=72:fontcolor=yellow:fontfile='C\\:/Windows/Fonts/msyh.ttc':"
        "x=(w-text_w)/2:y=680,"
        "drawtext=text='点击关注':fontsize=48:fontcolor=white:fontfile='C\\:/Windows/Fonts/msyh.ttc':"
        "x=(w-text_w)/2:y=800"
    ),
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-an", str(outro)
]
print("生成片尾（含文字）...")
r = subprocess.run(cmd_outro, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print(f"FAILED: {r.stderr[-300:]}")
else:
    size = outro.stat().st_size / 1024 / 1024
    print(f"片尾生成: {size:.1f}MB")
    r2 = subprocess.run(
        [ffmpeg, "-v", "error", "-show_entries", "stream=nb_frames,duration", "-select_streams", "v:0", "-of", "csv=p=0", str(outro)],
        capture_output=True, text=True
    )
    print(f"片尾信息: {r2.stdout.strip()}")

    # Step 1: concat intro + energybar + outro
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
    print("\n合并视频...")
    r3 = subprocess.run(cmd1, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r3.returncode != 0:
        print(f"FAILED concat: {r3.stderr[-200:]}")
    else:
        print(f"合并: {combined.stat().st_size/1024/1024:.1f}MB")

        # Step 2: 添加音频（带淡出）
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
        print("\n添加音频...")
        r4 = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r4.returncode != 0:
            print(f"FAILED audio: {r4.stderr[-300:]}")
        else:
            print(f"\n完成: {final.name} ({final.stat().st_size/1024/1024:.1f}MB)")

            # 验证
            r5 = subprocess.run(
                [ffmpeg, "-v", "error", "-show_entries", "format=duration,size", "-show_entries", "stream=nb_frames,width,height", "-of", "csv=p=0", str(final)],
                capture_output=True, text=True
            )
            print(f"验证: {r5.stdout.strip()}")
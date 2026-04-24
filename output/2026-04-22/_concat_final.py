"""合并横版16x9视频：片头+主视频+片尾 + 音频"""
import subprocess, shutil, os
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
FFPROBE = shutil.which("ffprobe") or FFMPEG
OUT = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-22")
TEMP = os.environ.get("TEMP", "F:/wkspace/temp")

def to_short(p):
    import ctypes
    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    GetShortPathNameW.restype = ctypes.c_uint
    buf_size = GetShortPathNameW(str(p), None, 0)
    if buf_size == 0:
        return str(p)
    buf = ctypes.create_unicode_buffer(buf_size)
    GetShortPathNameW(str(p), buf, buf_size)
    return buf.value

def get_dur(p):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                       "-of", "csv=p=0", str(p)], capture_output=True, text=True)
    return float(r.stdout.strip())

def run(cmd, desc=""):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"失败: {desc} - {r.stderr[-200:]}")
    return r.returncode == 0

# Step 1: 片段加静音音轨
print("准备片段...")
for name, path in [("intro", OUT / "_h_intro_new.mp4"),
                   ("main",  OUT / "_h_main_new.mp4"),
                   ("outro", OUT / "_h_outro_new.mp4")]:
    dur = get_dur(path)
    out = OUT / f"_{name}_a.mp4"
    run([
        FFMPEG, "-y",
        "-i", str(path),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", str(dur),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
        to_short(out)
    ], f"{name} 加音轨")

# Step 2: concat 视频（带静音音轨）
concat_list = Path(TEMP) / "cl.txt"
with open(concat_list, "w") as f:
    for seg in ["_intro_a.mp4", "_main_a.mp4", "_outro_a.mp4"]:
        f.write(f"file '{to_short(OUT / seg)}'\n")

video_concat = OUT / "_vcon.mp4"
run([
    FFMPEG, "-y", "-f", "concat", "-safe", "0",
    "-i", to_short(concat_list),
    "-c", "copy",
    to_short(video_concat)
], "视频concat")

# Step 3: 去掉 concat 视频的音轨
video_no_audio = OUT / "_vnoa.mp4"
run([
    FFMPEG, "-y",
    "-i", to_short(video_concat),
    "-c", "copy", "-an",
    to_short(video_no_audio)
], "去除静音音轨")

# Step 4: 合并视频 + 音频
final = OUT / "丽丽3_16x9_final4.mp4"
audio = OUT / "_audio_faded.mp3"
run([
    FFMPEG, "-y",
    "-i", to_short(video_no_audio),
    "-i", to_short(audio),
    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
    "-shortest",
    to_short(final)
], "合并音视频")

if os.path.exists(final):
    size = os.path.getsize(final) / 1024 / 1024
    print(f"完成: {final.name} ({size:.1f}MB)")
    # 验证
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name", "-of", "csv=p=0",
                        str(final)], capture_output=True, text=True)
    print(f"流: {r.stdout.strip().replace(chr(10), ' | ')}")
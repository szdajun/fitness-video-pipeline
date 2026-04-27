# -*- coding: utf-8 -*-
import subprocess, shutil
from pathlib import Path

ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
ffprobe = shutil.which("ffprobe") or "C:/Users/18091/ffmpeg/ffprobe.exe"

BASE = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-14")
SRC_ORIG = Path("C:/Users/18091/Desktop/短视频素材/丽丽1.mp4")

def get_duration(fp):
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries",
        "format=duration", "-of", "csv=p=0", str(fp)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return float(r.stdout.strip())

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"  FAIL: {r.stderr[-200:]}")
    return r

# Check if original source still exists
print(f"Original source exists: {SRC_ORIG.exists()}")

for video in ["9x16", "16x9"]:
    suffix = f"_full_{video}_v3.mp4"
    orig_file = BASE / f"丽丽1{suffix}"
    if not orig_file.exists():
        print(f"Not found: {orig_file}")
        continue

    dur = get_duration(orig_file)
    print(f"\n{video}: dur={dur:.1f}s")

    # Cut to 57s (safely under 60s), re-add fade in/out
    fade_start = 0.0
    fade_dur_in = 3.0
    fade_dur_out = 3.0
    cut_dur = 57.0  # stay under 60s

    out_file = BASE / f"丽丽1{suffix.replace('.mp4', '_60s.mp4')}"

    if SRC_ORIG.exists():
        # Use original source for audio (higher quality)
        # Video: first 57s of the processed video (which is the concat of intro+main+outro)
        # But we need audio from the orig source - it has different duration
        # Simpler: just trim the final mp4 and re-encode audio from itself
        cmd = [
            ffmpeg, "-y",
            "-i", str(orig_file),
            "-t", str(cut_dur),
            "-vf", f"fade=t=in:st={fade_start}:d={fade_dur_in},fade=t=out:st={cut_dur-fade_dur_out}:d={fade_dur_out}",
            "-af", f"afade=t=in:st={fade_start}:d={fade_dur_in},afade=t=out:st={cut_dur-fade_dur_out}:d={fade_dur_out}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            str(out_file)
        ]
    else:
        # No original source - use the processed video directly
        cmd = [
            ffmpeg, "-y",
            "-i", str(orig_file),
            "-t", str(cut_dur),
            "-vf", f"fade=t=in:st={fade_start}:d={fade_dur_in},fade=t=out:st={cut_dur-fade_dur_out}:d={fade_dur_out}",
            "-af", f"afade=t=in:st={fade_start}:d={fade_dur_in},afade=t=out:st={cut_dur-fade_dur_out}:d={fade_dur_out}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "96k",
            str(out_file)
        ]

    run(cmd)
    if out_file.exists():
        new_dur = get_duration(out_file)
        size_mb = out_file.stat().st_size // 1024 // 1024
        print(f"  -> {out_file.name}: {size_mb}MB, {new_dur:.1f}s")

print("\nDone!")
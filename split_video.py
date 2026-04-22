"""视频切割工具: 将长视频按指定时长切割成多段短视频"""

import subprocess
import sys
from pathlib import Path


def split_video(input_path, segment_seconds=45, ffmpeg="C:\\Users\\18091\\ffmpeg\\ffmpeg.exe"):
    """将视频切割为多段短视频

    Args:
        input_path: 输入视频路径
        segment_seconds: 每段时长（秒）
        ffmpeg: ffmpeg 路径
    """
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        return

    # 获取视频时长
    probe = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
    cmd_probe = [
        probe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(input_path),
    ]
    result = subprocess.run(cmd_probe, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    duration = float(result.stdout.strip())
    fps_result = subprocess.run(
        [probe, "-v", "error", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", str(input_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    fps_str = fps_result.stdout.strip()
    fps = float(fps_str.split("/")[0]) if "/" in fps_str else float(fps_str)

    print(f"视频: {input_path.name}")
    print(f"时长: {duration:.1f}s, 帧率: {fps:.1f}fps")

    # 输出目录
    output_dir = input_path.parent / f"{input_path.stem}_segments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 切割
    start = 0
    segment_idx = 1

    while start < duration:
        end = min(start + segment_seconds, duration)
        segment_dur = end - start

        if segment_dur < 5:
            print(f"    最后一段不足5秒，跳过")
            break

        # 输出文件名
        output_path = output_dir / f"{input_path.stem}_part{segment_idx:02d}.mp4"

        cmd = [
            ffmpeg, "-y",
            "-i", str(input_path),
            "-ss", str(start),
            "-t", str(segment_dur),
            "-c", "copy",  # 不重新编码，直接切割，速度快
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")

        size_mb = output_path.stat().st_size / 1024 / 1024
        ok = "OK" if result.returncode == 0 else "FAIL"
        print(f"  Part {segment_idx:02d}: {start:.0f}s ~ {end:.0f}s "
              f"({segment_dur:.1f}s, {size_mb:.1f}MB) {ok}")

        start = end
        segment_idx += 1

    print(f"\n共 {segment_idx - 1} 段，保存在: {output_dir}")
    return output_dir


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else input("输入视频路径: ")
    seg_len = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    split_video(video, seg_len)

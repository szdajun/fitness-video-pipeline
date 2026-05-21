"""视频循环脚本 — 重复播放加长视频（适合跳操重复动作）

用法:
    python loop_video.py "视频.mp4" -o "输出.mp4" --count 3
    python loop_video.py --auto  # 自动处理所有_final.mp4
"""

import argparse, subprocess, shutil, uuid
from pathlib import Path


FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"


def loop_video(video_path, output_path, count=3):
    """将视频重复 N 次"""
    if count <= 1:
        shutil.copy2(str(video_path), str(output_path))
        return True

    uid = uuid.uuid4().hex[:8]
    tmp_dir = Path(f"F:/wkspace/fitness-video-pipeline/output/tmp_loop_{uid}")
    tmp_dir.mkdir(exist_ok=True)

    # 复制到临时目录（解决中文路径）
    src = tmp_dir / "src.mp4"
    shutil.copy2(str(video_path), str(src))

    # 写 concat list（N 次）
    list_file = tmp_dir / "list.txt"
    with open(list_file, "w") as f:
        for _ in range(count):
            f.write(f"file '{src}'\n")

    # 视频 concat
    v_out = tmp_dir / "v.mp4"
    r = subprocess.run([
        FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-an", str(v_out)
    ], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"  视频循环失败: {r.stderr[-200:]}")
        shutil.rmtree(tmp_dir)
        return False

    # 音频 concat
    a_out = tmp_dir / "a.aac"
    r = subprocess.run([
        FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-vn", "-c:a", "aac", "-b:a", "128k", str(a_out)
    ], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"  音频循环失败: {r.stderr[-200:]}")
        shutil.rmtree(tmp_dir)
        return False

    # 合并
    r = subprocess.run([
        FFMPEG, "-y",
        "-i", str(v_out), "-i", str(a_out),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "copy",
        "-shortest", str(output_path)
    ], capture_output=True, text=True, errors="replace")

    shutil.rmtree(tmp_dir)

    if r.returncode != 0:
        print(f"  最终合并失败: {r.stderr[-200:]}")
        return False

    return True


def auto_loop(output_base, count=3, min_duration=0, max_duration=300):
    """自动处理：时长不足 max_duration 的视频循环补足"""
    videos = list(output_base.glob("**/*_final.mp4"))
    # 排除已循环的视频
    videos = [v for v in videos if "循环" not in v.name]

    print(f"找到 {len(videos)} 个视频，检查时长...")

    for v in sorted(videos):
        # 检查时长
        r = subprocess.run([
            FFMPEG, "-y",
            "-i", str(v),
            "-vn", "-c:a", "aac", "-t", "1",
            "-f", "null", "-"
        ], capture_output=True, text=True, errors="replace")

        # 获取实际时长
        r = subprocess.run([
            FFMPEG, "-v", "quiet", "-print_format", "json", "-show_format", str(v)
        ], capture_output=True, text=True, errors="replace")
        try:
            import json
            data = json.loads(r.stdout)
            duration = float(data["format"]["duration"])
        except:
            duration = 0

        duration_sec = duration
        print(f"  {v.name}: {duration_sec:.0f}s", end="")

        if duration_sec >= max_duration:
            print(f" — 时长足够，跳过")
            continue

        # 需要循环
        loop_count = max(2, (max_duration // int(duration_sec)) + 1)
        if loop_count < 2:
            print(f" — 时长足够，跳过")
            continue

        out_name = v.name.replace("_final", "_循环_final")
        out_file = v.parent / out_name

        print(f" — 时长{duration_sec:.0f}s x {loop_count}次")
        if loop_video(v, out_file, count=loop_count):
            print(f"  完成: {out_file.name} ({out_file.stat().st_size/1024/1024:.1f}MB)")


def main():
    parser = argparse.ArgumentParser(description="视频循环（适合跳操）")
    parser.add_argument("input", nargs="?", help="输入视频")
    parser.add_argument("-o", "--output", help="输出视频")
    parser.add_argument("-c", "--count", type=int, default=3,
                        help="重复次数（默认3次）")
    parser.add_argument("--auto", action="store_true",
                        help="自动处理 output 目录下所有视频")
    parser.add_argument("--max-duration", type=int, default=300,
                        help="目标时长（秒），不足则循环补足（默认300s=5分钟）")
    args = parser.parse_args()

    if args.auto:
        auto_loop(Path("F:/wkspace/fitness-video-pipeline/output"),
                  count=args.count, max_duration=args.max_duration)
    elif args.input and args.output:
        loop_video(Path(args.input), Path(args.output), count=args.count)
    else:
        print("用法:")
        print("  python loop_video.py '视频.mp4' -o '输出.mp4' --count 3")
        print("  python loop_video.py --auto                    # 自动循环不足5分钟的视频")


if __name__ == "__main__":
    main()

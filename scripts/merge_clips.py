"""视频合并脚本 — 同一天同一教练合并"""

import argparse, subprocess, shutil, uuid
from pathlib import Path
from collections import defaultdict


FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"


def get_coach(name):
    if "艳青" in name or "胭脂" in name or "海军" in name: return "艳青"
    if "丽丽" in name or "腰女" in name: return "丽丽"
    if "小红豆" in name or "妙女" in name: return "小红豆"
    if "建玲" in name or "带队" in name: return "建玲"
    return "unknown"


def get_date(stem):
    for i in range(len(stem) - 8):
        part = stem[i:i+8]
        if part.isdigit() and 20200101 <= int(part) <= 20301231:
            from datetime import datetime
            try:
                return datetime(int(part[:4]), int(part[4:6]), int(part[6:8]))
            except:
                pass
    return None


def merge_videos(video_paths, output_path):
    """合并多个视频：逐个转码 + stream concat（最可靠方案）"""
    if len(video_paths) == 1:
        shutil.copy2(str(video_paths[0]), str(output_path))
        return True

    uid = uuid.uuid4().hex[:8]
    tmp = Path(f"F:/wkspace/fitness-video-pipeline/output/tmp_{uid}")
    tmp.mkdir(exist_ok=True)

    # 1. 复制到临时目录（解决中文路径）
    clips = []
    for i, vp in enumerate(video_paths):
        dst = tmp / f"c{i}.mp4"
        shutil.copy2(str(vp), str(dst))
        clips.append(dst)

    # 2. 逐个转码为统一 H.264（归一化）
    encoded = []
    for c in clips:
        enc = tmp / f"e{clips.index(c)}.mp4"
        r = subprocess.run([
            FFMPEG, "-y", "-i", str(c),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-an",  # 去音频
            str(enc)
        ], capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(f"    转码失败: {r.stderr[-200:]}")
            shutil.rmtree(tmp)
            return False
        encoded.append(enc)

    # 3. 视频 stream concat
    list_file = tmp / "list.txt"
    with open(list_file, "w") as f:
        for e in encoded:
            f.write(f"file '{e}'\n")

    v_out = tmp / "v.mp4"
    r = subprocess.run([
        FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(v_out)
    ], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"    视频拼接失败: {r.stderr[-200:]}")
        shutil.rmtree(tmp)
        return False

    # 4. 音频 stream concat
    with open(list_file, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")

    a_out = tmp / "a.aac"
    r = subprocess.run([
        FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-vn", "-c:a", "aac", "-b:a", "128k", str(a_out)
    ], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"    音频拼接失败: {r.stderr[-200:]}")
        shutil.rmtree(tmp)
        return False

    # 5. 合并
    r = subprocess.run([
        FFMPEG, "-y",
        "-i", str(v_out), "-i", str(a_out),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "copy",
        "-shortest", str(output_path)
    ], capture_output=True, text=True, errors="replace")

    shutil.rmtree(tmp)

    if r.returncode != 0:
        print(f"    最终合并失败: {r.stderr[-200:]}")
        return False

    return True


def auto_merge(output_base, min_clips=2):
    videos = list(output_base.glob("**/*_final.mp4"))
    groups = defaultdict(list)
    for v in videos:
        date = get_date(v.stem)
        if date:
            key = (f"{date.year:04d}-{date.month:02d}-{date.day:02d}", get_coach(v.stem))
            groups[key].append(v)

    print(f"找到 {len(videos)} 个视频，分成 {len(groups)} 组（每天每教练 >= {min_clips} 才合并）")
    print()

    for (date_str, coach), clips in sorted(groups.items()):
        if len(clips) < min_clips:
            print(f"[{date_str} {coach}] {len(clips)} 个，跳过（<{min_clips}）")
            continue
        clips.sort()
        out_dir = output_base / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"合并_{coach}_{date_str}.mp4"
        print(f"[{date_str} {coach}] {len(clips)} 个 -> {out_file.name}")
        for c in clips:
            print(f"  + {c.name}")
        if merge_videos(clips, out_file):
            print(f"  完成: {out_file.name} ({out_file.stat().st_size/1024/1024:.1f}MB)")
        else:
            print(f"  失败")


def main():
    parser = argparse.ArgumentParser(description="视频合并")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--min-clips", type=int, default=2)
    parser.add_argument("--input-dir", default="F:/wkspace/fitness-video-pipeline/output")
    args = parser.parse_args()
    auto_merge(Path(args.input_dir), args.min_clips)


if __name__ == "__main__":
    main()

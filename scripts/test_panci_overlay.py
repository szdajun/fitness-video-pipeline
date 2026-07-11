"""判词水墨插画测试视频合成 (4 张 → 8s Ken Burns + 叠加到小红豆1_2_merged opening).

不接主管线, 只验视觉.

输出: output/2026-07-12/_test_xhd_panci.mp4
"""
import json
import subprocess
from pathlib import Path

PAINT_DIR = Path(r"F:\wkspace\fitness-video-pipeline\tools\panci_paint\xiaohongdou")
MAIN_VIDEO = Path(r"F:\wkspace\fitness-video-pipeline\output\2026-07-12\小红豆1_2_merged_full_16x9_1920x1080.mp4")
OUT_DIR = Path(r"F:\wkspace\fitness-video-pipeline\output\2026-07-12")
TMP_DIR = Path(r"F:\wkspace\fitness-video-pipeline\_tmp_panci")

# 8s, 4 张每张 2s
PANCI_TEXT = [
    {"img": "1.png", "text": "红豆生来俏模样", "color": "yellow"},
    {"img": "2.png", "text": "香汗淋漓透红妆", "color": "yellow"},
    {"img": "3.png", "text": "娇喘微微惹人怜", "color": "yellow"},
    {"img": "4.png", "text": "花枝乱颤舞霓裳", "color": "yellow"},
]
FPS = 30
WIDTH = 1080
HEIGHT = 1920  # 9:16 竖版
SEG_DUR = 2.0  # 每段 2s
TOTAL_DUR = 8.0


def make_paint_clip():
    """4 张静态图 → 8s 视频, 每段 Ken Burns 慢推 (从大到小/小到大 + 平移)."""
    TMP_DIR.mkdir(exist_ok=True)
    out = TMP_DIR / "paint_clip.mp4"

    # 用 ffmpeg concat demuxer + 每段独立 scale/zoompan
    # 简化: 用 zoompan filter 把单张图变 2s 视频
    seg_files = []
    for i, p in enumerate(PANCI_TEXT):
        seg = TMP_DIR / f"seg_{i}.mp4"
        img = PAINT_DIR / p["img"]
        # zoompan: 60 帧 (2s@30fps), 从 1.0 缓推到 1.08
        # 单图变视频: ffmpeg -loop 1 -i img.png -vf "zoompan=z='1.0+0.08*on/60':d=60:s=1080x1920" -t 2 -r 30 seg.mp4
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(img),
            "-vf", (
                # zoompan 单 filter 直接放大+缓推
                f"zoompan=z='min(1.0+0.06*on/60\\,1.08)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d=60:s={WIDTH}x{HEIGHT}:fps=30,"
                f"format=yuv420p"
            ),
            "-t", "2",
            "-r", "30",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(seg),
        ]
        print(f"[seg] {i+1}/4: {p['text']}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[err] {r.stderr[-500:]}")
            return None
        seg_files.append(seg)

    # concat 4 段
    list_file = TMP_DIR / "concat.txt"
    list_file.write_text("\n".join(f"file '{s}'" for s in seg_files), encoding="utf-8")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[err concat] {r.stderr[-500:]}")
        return None

    print(f"[seg] all 4 segments concatenated → {out}")
    return out


def overlay_on_main(paint_clip: Path):
    """把 paint_clip 叠加到 main video t=4-12s 的位置.

    用 ffmpeg overlay filter + enable=between(t,4,12)
    """
    out = OUT_DIR / "_test_xhd_panci.mp4"

    # main video crop 到 9:16 1080x1920 (跟 paint 同样竖版, 叠加更顺)
    # 取 t=4-12s (8s 替换原片头 hook/opening)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(MAIN_VIDEO),
        "-i", str(paint_clip),
        "-filter_complex",
        (
            # step 1: main video t=4-12s crop 9:16 1080x1920
            f"[0:v]trim=start=4:duration=8,setpts=PTS-STARTPTS,"
            f"scale=1080:1920:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,"
            f"format=yuv420p[main];"
            # step 2: paint clip 已经 9:16, 直接用
            f"[1:v]format=yuv420p[paint];"
            # step 3: 半透明叠加 paint 在 main 上, 50% alpha
            f"[main][paint]overlay=0:0:format=auto:eof_action=pass[out]"
        ),
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-t", "8",
        "-an",  # 测试版不要音频
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[err overlay] {r.stderr[-800:]}")
        return None

    print(f"[out] test video → {out}")
    return out


def main():
    paint_clip = make_paint_clip()
    if not paint_clip:
        return
    test = overlay_on_main(paint_clip)
    if test:
        print(f"[done] {test} ({test.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
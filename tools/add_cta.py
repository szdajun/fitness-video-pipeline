"""tools/add_cta.py — 给已有 Shorts 视频后处理加 CTA (PIL + ffmpeg overlay)

绕开 ffmpeg 8.1 drawtext UTF-8 parser bug. 直接 ffmpeg overlay PIL PNG.
"""
import argparse
import os
import sys
import subprocess

# 加项目根到 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.render_cta_overlay import render_cta_png


def add_cta_to_video(video_path: str, output_path: str = None,
                     start_sec: float = 26.0, end_sec: float = 30.0) -> str:
    """给 9:16 视频加 CTA overlay (start_sec~end_sec 显示).

    Args:
        video_path: 输入 Shorts 视频 (1080x1920, ≤30s)
        output_path: 输出路径 (None = 覆盖输入)
        start_sec: CTA 出现时间点 (秒)
        end_sec: CTA 消失时间点 (秒)

    Returns:
        output_path
    """
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_with_cta{ext}"

    # 渲染 CTA PNG
    cta_png = video_path.replace(".mp4", "_cta.png")
    render_cta_png(cta_png)

    # ffmpeg overlay (用 gte*lte 替代 between 避开 ffmpeg 8.1 , bug)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", cta_png,
        "-filter_complex",
        f"[0:v][1:v]overlay=0:0:enable='gte(t,{start_sec})*lte(t,{end_sec})'",
        "-c:a", "copy",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    if r.returncode != 0:
        print(f"[FAIL] {r.stderr[-500:]}")
        return None
    print(f"[OK] {output_path} ({os.path.getsize(output_path)//1024//1024}MB)")
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="输入 9:16 视频")
    ap.add_argument("-o", "--output", default=None, help="输出路径 (None=原文件+_with_cta 后缀)")
    ap.add_argument("--start", type=float, default=26.0, help="CTA 出现秒")
    ap.add_argument("--end", type=float, default=30.0, help="CTA 消失秒")
    args = ap.parse_args()
    add_cta_to_video(args.video, args.output, args.start, args.end)

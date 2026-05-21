"""阶段33: 电影滤镜

FFmpeg 色彩分级实现怀旧/电影色调。
"""

import subprocess, shutil
from pathlib import Path
from lib.utils import path_exists


PRESETS = {
    "warm_vintage": "eq=brightness=0.02:saturation=0.9:contrast=1.05,"
                    "colorbalance=rs=0.05:gs=-0.02:bs=-0.08,"
                    "vignette=PI/4",
    "cool_cinema": "eq=saturation=0.85:contrast=1.1,"
                   "colorbalance=rs=-0.03:gs=0.02:bs=0.08",
    "bw_classic": "hue=s=0,eq=contrast=1.15:saturation=0",
    "golden_hour": "eq=brightness=0.03:saturation=1.05:contrast=0.95,"
                   "colorbalance=rs=0.08:gs=0.02:bs=-0.05",
    "teal_orange": "eq=contrast=1.1,"
                   "colorbalance=rs=0.04:gs=-0.01:bs=-0.06:rh=0.03:gh=0:bh=-0.03,"
                   "vignette=PI/5",
}


class FilmLookStage:
    def run(self, ctx):
        if ctx.get("filmlook_path") and path_exists(ctx.get("filmlook_path")):
            print("    已存在，跳过")
            return

        cfg = ctx.config.get("film_look", {})
        if not cfg.get("enabled", False):
            return

        preset = cfg.get("preset", "warm_vintage")
        filter_str = PRESETS.get(preset, PRESETS["warm_vintage"])

        input_path = (ctx.get("burst_path") or
                     ctx.get("danmaku_path") or
                     ctx.get("pip_path") or
                     ctx.get("speedramp_path") or
                     ctx.get("mascot_path") or
                     ctx.get("watermark_path") or
                     ctx.get("energybar_path") or
                     str(ctx.input_path))
        if not input_path or not path_exists(input_path):
            return

        out_path = ctx.output_dir / f"{Path(input_path).stem}_film.mp4"
        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"

        # GPU 编码检测
        use_nvenc = False
        r_test = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "1M", "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=5)
        use_nvenc = r_test.returncode == 0

        enc_name = "h264_nvenc" if use_nvenc else "libx264"
        print(f"    电影滤镜: {preset}, encoder={enc_name}")

        if use_nvenc:
            cmd = [
                ffmpeg, "-y", "-i", input_path,
                "-vf", filter_str,
                "-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "21", "-b:v", "0",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                str(out_path)
            ]
        else:
            cmd = [
                ffmpeg, "-y", "-i", input_path,
                "-vf", filter_str,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                str(out_path)
            ]

        r = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)

        if r.returncode != 0:
            print(f"    滤镜失败: {r.stderr[-200:]}")
            ctx.set("filmlook_path", input_path)
            return

        ctx.set("filmlook_path", str(out_path))
        print(f"    输出: {out_path.name}")

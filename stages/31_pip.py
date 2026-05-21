"""阶段31: 画中画

原始画面缩到角落，处理后的主画面全屏。
"""

import subprocess, shutil
from pathlib import Path
from lib.utils import path_exists


class PiPStage:
    def run(self, ctx):
        if ctx.get("pip_path") and path_exists(ctx.get("pip_path")):
            print("    已存在，跳过")
            return

        cfg = ctx.config.get("pip", {})
        if not cfg.get("enabled", False):
            return

        # 主视频（处理过的）
        main_path = (ctx.get("filmlook_path") or
                     ctx.get("burst_path") or
                     ctx.get("speedramp_path") or
                     ctx.get("mascot_path") or
                     ctx.get("watermark_path") or
                     ctx.get("energybar_path") or
                     ctx.get("beatflash_path") or
                     str(ctx.input_path))
        if not main_path or not path_exists(main_path):
            print("    跳过: 无主视频")
            return

        # 原片
        source_path = str(ctx.input_path)
        if not path_exists(source_path):
            print("    跳过: 无原始视频")
            return

        video_info = ctx.get("video_info")
        main_w = video_info.get("width", 1280)
        main_h = video_info.get("height", 720)

        # 画中画配置
        pos = cfg.get("position", "bottom-right")
        pip_scale = cfg.get("scale", 0.25)  # 缩到25%
        margin = cfg.get("margin", 10)
        border = cfg.get("border", 3)

        pip_w = int(main_w * pip_scale)
        pip_h = int(main_h * pip_scale)

        # 位置坐标
        if pos == "bottom-right":
            x = main_w - pip_w - margin
            y = main_h - pip_h - margin
        elif pos == "bottom-left":
            x = margin
            y = main_h - pip_h - margin
        elif pos == "top-right":
            x = main_w - pip_w - margin
            y = margin
        elif pos == "top-left":
            x = margin
            y = margin
        else:
            x = main_w - pip_w - margin
            y = main_h - pip_h - margin

        out_path = ctx.output_dir / f"{Path(main_path).stem}_pip.mp4"
        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"

        print(f"    画中画: {pos}, scale={pip_scale}, {pip_w}x{pip_h}")

        cmd = [
            ffmpeg, "-y",
            "-i", main_path,
            "-i", source_path,
            "-filter_complex",
            # 子画面缩放到指定大小
            f"[1:v]scale={pip_w}:{pip_h}:flags=lanczos,"
            # 加边框
            f"pad={pip_w+border*2}:{pip_h+border*2}:{border}:{border}:white,"
            f"setsar=1[sub];"
            # 叠加到主画面
            f"[0:v][sub]overlay={x}:{y}:shortest=1[v]",
            "-map", "[v]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "1",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(out_path)
        ]

        r = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)

        if r.returncode != 0:
            print(f"    画中画失败: {r.stderr[-300:]}")
            ctx.set("pip_path", main_path)
            return

        ctx.set("pip_path", str(out_path))
        print(f"    输出: {out_path.name}")

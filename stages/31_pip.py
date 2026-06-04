"""阶段31: 画中画

原始画面缩到角落，处理后的主画面全屏。
"""

import subprocess, shutil
from pathlib import Path
import cv2
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

        # 关键: 用 main_path 实际尺寸, 不是 video_info 原始尺寸
        # 否则 h2v 后 (1080x1920) 用原始 (1280x720) 算坐标, PIP 跑到画布外
        cap_main = cv2.VideoCapture(main_path)
        if not cap_main.isOpened():
            print(f"    跳过: 无法打开主视频 {main_path}")
            return
        main_w = int(cap_main.get(cv2.CAP_PROP_FRAME_WIDTH))
        main_h = int(cap_main.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_main.release()
        if main_w <= 0 or main_h <= 0:
            # 兜底回 video_info
            video_info = ctx.get("video_info") or {}
            main_w = video_info.get("width", 1280)
            main_h = video_info.get("height", 720)

        # 画中画配置
        pos = cfg.get("position", "bottom-right")
        pip_scale = cfg.get("scale", 0.25)  # 缩到25%
        margin = cfg.get("margin", 10)
        border = cfg.get("border", 3)

        pip_w = int(main_w * pip_scale)
        pip_h = int(main_h * pip_scale)

        # 边界钳制: 防止 pip_w+margin 超过 main_w (避免溢出)
        if pip_w + 2 * margin > main_w:
            pip_w = main_w - 2 * margin
        if pip_h + 2 * margin > main_h:
            pip_h = main_h - 2 * margin

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

        print(f"    画中画: {pos}, scale={pip_scale}, {pip_w}x{pip_h} at ({x},{y}) [main {main_w}x{main_h}]")

        cmd = [
            ffmpeg, "-y",
            "-i", main_path,
            "-i", source_path,
            "-filter_complex",
            # 子画面缩放
            f"[1:v]scale={pip_w}:{pip_h}:flags=lanczos,setsar=1[sub];"
            # 叠加
            f"[0:v][sub]overlay={x}:{y}:shortest=1,"
            # 淡黄细边框
            f"drawbox=x={x-3}:y={y-3}:w={pip_w+6}:h={pip_h+6}:color=green@0.5:t=2[v]",
            "-map", "[v]",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
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

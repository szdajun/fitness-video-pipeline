"""阶段24: 水印字幕

在视频上添加文字水印（姓名、日期等），可选位置和样式。

配置:
  watermark_text: 水印文字（默认空=不显示）
  watermark_position: bottom-right | bottom-left | top-right | top-left | bottom-center
  watermark_size: 字体大小（默认24）
  watermark_color: (B,G,R) 颜色（默认(255,255,255)白色）
  watermark_alpha: 透明度 0~1（默认0.7）
  watermark_margin: 边距像素（默认20）
  show_date: true 显示日期（默认True）
"""

import cv2
import numpy as np
import ctypes
import subprocess
import shutil
import tempfile
import os
from pathlib import Path


class WatermarkStage:
    def run(self, ctx):
        if ctx.get("watermark_path") and Path(ctx.get("watermark_path")).exists():
            print("    已存在，跳过")
            return

        input_path = (
            ctx.get("color_path") or
            ctx.get("skin_tone_filter_path") or
            ctx.get("denoise_path") or
            ctx.get("ken_burns_path") or
            ctx.get("face_warp_path") or
            ctx.get("warped_path") or
            ctx.get("h2v_path")
        )
        if not input_path or not Path(input_path).exists():
            print("    跳过: 无输入视频")
            ctx.set("watermark_path", None)
            return

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        cap_check = cv2.VideoCapture(input_path)
        if not cap_check.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")
        orig_w = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_check.release()

        cfg = ctx.config.get("watermark", {})
        text = cfg.get("watermark_text", "")
        position = cfg.get("watermark_position", "bottom-right")
        font_size = cfg.get("watermark_size", 24)
        margin = cfg.get("watermark_margin", 20)
        show_date = cfg.get("show_date", True)

        # 颜色配置
        color_cfg = cfg.get("watermark_color", (255, 255, 255))
        if isinstance(color_cfg, str):
            color = (255, 255, 255)
        else:
            color = tuple(max(0, min(255, int(c))) for c in color_cfg)
        alpha = cfg.get("watermark_alpha", 0.7)

        if not text and not show_date:
            print("    跳过: 无水印文字")
            ctx.set("watermark_path", input_path)
            return

        print(f"    水印: pos={position}, text='{text}', size={font_size}")

        out_path = ctx.output_dir / f"{Path(input_path).stem}_watermark.mp4"
        tmpdir = Path(tempfile.mkdtemp(prefix="wm_"))

        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint

        def to_short(p):
            buf_size = GetShortPathNameW(str(p), None, 0)
            if buf_size == 0:
                return str(p)
            buf = ctypes.create_unicode_buffer(buf_size)
            GetShortPathNameW(str(p), buf, buf_size)
            return buf.value

        tmpdir_short = to_short(str(tmpdir))

        # 字体：尝试多种中文字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 中文字体fallback
        import os
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
        ]
        chinese_font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    chinese_font = cv2.FONT_HERSHEY_SIMPLEX
                    # freetype 加载（如果可用）
                    import cv2
                    chinese_font = None
                except Exception:
                    pass

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0

        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 构建水印文字
            lines = []
            if text:
                lines.append(text)
            if show_date:
                lines.append(date_str)

            if lines:
                # 计算文字区域大小
                max_w, max_h = 0, 0
                for line in lines:
                    (tw, th), _ = cv2.getTextSize(line, font, font_size / 28, 2)
                    max_w = max(max_w, tw)
                    max_h += th + 10

                # 背景区域
                pad = 8
                bg_h = max_h + pad * 2
                bg_w = max_w + pad * 2

                # 计算位置
                if position == "bottom-right":
                    x = orig_w - bg_w - margin
                    y = orig_h - bg_h - margin
                elif position == "bottom-left":
                    x = margin
                    y = orig_h - bg_h - margin
                elif position == "bottom-center":
                    x = (orig_w - bg_w) // 2
                    y = orig_h - bg_h - margin
                elif position == "top-right":
                    x = orig_w - bg_w - margin
                    y = margin
                elif position == "top-left":
                    x = margin
                    y = margin
                else:
                    x = orig_w - bg_w - margin
                    y = orig_h - bg_h - margin

                # 画半透明背景
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x + bg_w, y + bg_h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0)

                # 画文字
                cy = y + pad
                for line in lines:
                    cv2.putText(frame, line, (x + pad, cy + font_size),
                                font, font_size / 28, color, 2)
                    cy += font_size + 10

            cv2.imwrite(f"{tmpdir_short}/f_{frame_idx:06d}.png", frame)
            frame_idx += 1
            if frame_idx % 30 == 0:
                pct = frame_idx / max_frames * 100
                print(f"    进度: {pct:.0f}% ({frame_idx}/{max_frames})")

        cap.release()

        tmp_fd, tmp_path_tmp = tempfile.mkstemp(suffix='.mp4')
        os.close(tmp_fd)
        tmp_path_tmp = Path(tmp_path_tmp)

        print(f"    写入完成: {frame_idx} 帧，调用 FFmpeg 编码...")
        ffmpeg_bin = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        cmd = [ffmpeg_bin, "-y", "-v", "info",
               "-framerate", str(fps),
               "-i", f"{tmpdir_short}/f_%06d.png",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-an", str(tmp_path_tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"    FFmpeg 错误: {r.stderr[:500]}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(f"FFmpeg 编码失败")

        shutil.rmtree(tmpdir, ignore_errors=True)

        try:
            shutil.move(str(tmp_path_tmp), str(out_path))
        except Exception:
            alt_path = ctx.output_dir / f"{Path(input_path).stem}_watermark_new.mp4"
            shutil.move(str(tmp_path_tmp), str(alt_path))
            out_path = alt_path

        ctx.set("watermark_path", str(out_path))
        print(f"    输出: {out_path.name}")

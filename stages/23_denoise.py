"""阶段23: 视频降噪

使用 OpenCV 的快速非局部均值去噪，适合健身房等暗光环境下的视频。
降噪强度越高越平滑，但细节损失越多。

配置:
  denoise_strength: 0~20, 降噪强度（默认3，夜景建议8~15）
  denoise_mode: fastNlMeans (默认) | GaussianBlur
"""

import cv2
from lib.utils import path_exists
import numpy as np
import ctypes
import subprocess
import shutil
import tempfile
import os
from pathlib import Path


class DenoiseStage:
    def run(self, ctx):
        if ctx.get("denoise_path") and path_exists(ctx.get("denoise_path")):
            print("    已存在，跳过")
            return

        input_path = (
            ctx.get("color_path") or
            ctx.get("skin_tone_filter_path") or
            ctx.get("ken_burns_path") or
            ctx.get("face_warp_path") or
            ctx.get("warped_path") or
            ctx.get("h2v_path") or
            str(ctx.input_path)  # 横屏 fallback
        )
        if not input_path or not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("denoise_path", None)
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

        cfg = ctx.config.get("denoise", {})
        strength = cfg.get("denoise_strength", 3)
        mode = cfg.get("denoise_mode", "fastNlMeans")

        if strength <= 0:
            print("    跳过: denoise_strength=0")
            ctx.set("denoise_path", input_path)
            return

        print(f"    降噪: mode={mode}, strength={strength}")

        out_path = ctx.output_dir / f"{Path(input_path).stem}_denoise.mp4"
        tmpdir = Path(tempfile.mkdtemp(prefix="dn_"))

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

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if mode == "fastNlMeans":
                # h: filter strength (higher = stronger denoising, more detail loss)
                # hForColorComponents: same for color images
                # templateWindowSize: must be odd (7 default)
                # searchWindowSize: must be odd (21 default)
                frame = cv2.fastNlMeansDenoisingColored(
                    frame, None,
                    h=strength,
                    hColor=strength,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            elif mode == "GaussianBlur":
                # 高斯模糊降噪，sigma 根据 strength 计算
                sigma = max(1, strength // 3)
                kernel = max(3, (sigma // 2) * 2 + 1)
                frame = cv2.GaussianBlur(frame, (kernel, kernel), sigma)

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
            alt_path = ctx.output_dir / f"{Path(input_path).stem}_denoise_new.mp4"
            shutil.move(str(tmp_path_tmp), str(alt_path))
            out_path = alt_path

        ctx.set("denoise_path", str(out_path))
        print(f"    输出: {out_path.name}")

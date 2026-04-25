"""阶段22: 肤色滤镜（粉色/暖色/冷色）

独立于色彩调整的肤色美化滤镜，基于HSV色彩空间调整。
支持多种风格叠加，适配健身视频的灯光环境。

配置:
  pink_filter: 0~1, 粉色滤镜强度（色相偏红+饱和度+亮度）
  warm_filter: 0~1, 暖色调强度（色温偏暖）
  cool_filter: 0~1, 冷色调强度（色温偏冷）
  soft_glow: 0~1, 柔光效果（轻微模糊叠加）
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


class SkinToneFilterStage:
    def run(self, ctx):
        # 增量跳过
        if ctx.get("skin_tone_filter_path") and path_exists(ctx.get("skin_tone_filter_path")):
            print("    已存在，跳过")
            return

        input_path = (
            ctx.get("color_path") or
            ctx.get("ken_burns_path") or
            ctx.get("face_warp_path") or
            ctx.get("warped_path") or
            ctx.get("h2v_path") or
            str(ctx.input_path)  # 横屏 fallback
        )
        if not input_path or not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("skin_tone_filter_path", None)
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

        cfg = ctx.config.get("skin_tone_filter", {})
        pink = cfg.get("pink_filter", 1.0)
        warm = cfg.get("warm_filter", 0.0)
        cool = cfg.get("cool_filter", 0.0)
        soft_glow = cfg.get("soft_glow", 0.0)

        enabled = pink > 0 or warm > 0 or cool > 0 or soft_glow > 0
        if not enabled:
            print("    跳过: 无肤色滤镜参数")
            ctx.set("skin_tone_filter_path", input_path)
            return

        print(f"    pink={pink}, warm={warm}, cool={cool}, soft_glow={soft_glow}")

        out_path = ctx.output_dir / f"{Path(input_path).stem}_skin_tone.mp4"
        tmpdir = Path(tempfile.mkdtemp(prefix="stf_"))

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

            # 1. 粉色滤镜
            if pink > 0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 0] = np.clip(hsv[:, :, 0] - 2 * pink, 0, 179)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] + 10 * pink, 0, 255)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 8 * pink, 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 2. 暖色调
            if warm > 0:
                frame = frame.astype(np.float32)
                frame[:, :, 2] = np.clip(frame[:, :, 2] + 15 * warm, 0, 255)  # R+
                frame[:, :, 0] = np.clip(frame[:, :, 0] - 8 * warm, 0, 255)   # B-
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            # 3. 冷色调
            if cool > 0:
                frame = frame.astype(np.float32)
                frame[:, :, 2] = np.clip(frame[:, :, 2] - 15 * cool, 0, 255)  # R-
                frame[:, :, 0] = np.clip(frame[:, :, 0] + 8 * cool, 0, 255)   # B+
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            # 4. 柔光效果（轻微高斯模糊叠加）
            if soft_glow > 0:
                blur = cv2.GaussianBlur(frame, (0, 0), 15)
                frame = cv2.addWeighted(frame, 1.0 - soft_glow * 0.5, blur, soft_glow * 0.5, 0)

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
            alt_path = ctx.output_dir / f"{Path(input_path).stem}_skin_tone_new.mp4"
            shutil.move(str(tmp_path_tmp), str(alt_path))
            out_path = alt_path

        ctx.set("skin_tone_filter_path", str(out_path))
        print(f"    输出: {out_path.name}")

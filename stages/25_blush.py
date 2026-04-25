"""阶段25: 腮红/局部美白

在面部的颧骨区域添加自然的红润效果，或对全脸进行局部美白提亮。
基于肤色检测（HSV/YCrCb）定位皮肤区域，叠加粉色色调。

配置:
  blush_strength: 0~1, 腮红强度（默认0.3）
  brighten_strength: 0~1, 局部美白强度（默认0.15）
  blush_color: (B,G,R) 腮红颜色（默认偏粉(180,80,100)）
"""

import cv2
import numpy as np
import ctypes
import subprocess
import shutil
import tempfile
import os
from pathlib import Path


class BlushStage:
    def run(self, ctx):
        if ctx.get("blush_path") and Path(ctx.get("blush_path")).exists():
            print("    已存在，跳过")
            return

        input_path = (
            ctx.get("color_path") or
            ctx.get("skin_tone_filter_path") or
            ctx.get("denoise_path") or
            ctx.get("ken_burns_path") or
            ctx.get("face_warp_path") or
            ctx.get("warped_path") or
            ctx.get("h2v_path") or
            str(ctx.input_path)  # 横屏 fallback
        )
        if not input_path or not Path(input_path).exists():
            print("    跳过: 无输入视频")
            ctx.set("blush_path", None)
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

        cfg = ctx.config.get("blush", {})
        blush_strength = cfg.get("blush_strength", 0.3)
        brighten = cfg.get("brighten_strength", 0.15)

        if blush_strength <= 0 and brighten <= 0:
            print("    跳过: 腮红和美白强度都为0")
            ctx.set("blush_path", input_path)
            return

        print(f"    腮红: blush={blush_strength}, brighten={brighten}")

        out_path = ctx.output_dir / f"{Path(input_path).stem}_blush.mp4"
        tmpdir = Path(tempfile.mkdtemp(prefix="blush_"))

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

            if blush_strength > 0 or brighten > 0:
                frame = self._apply_blush(frame, blush_strength, brighten)

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
            alt_path = ctx.output_dir / f"{Path(input_path).stem}_blush_new.mp4"
            shutil.move(str(tmp_path_tmp), str(alt_path))
            out_path = alt_path

        ctx.set("blush_path", str(out_path))
        print(f"    输出: {out_path.name}")

    def _apply_blush(self, frame, blush_strength, brighten_strength):
        """基于肤色检测的腮红+局部美白"""
        h, w = frame.shape[:2]

        # 转换为 YCrCb 检测皮肤
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1].astype(np.float32)
        cb = ycrcb[:, :, 2].astype(np.float32)

        # 皮肤掩码（Cr 135~180, Cb 85~135）
        skin_mask = np.zeros_like(cr)
        skin_mask[(cr >= 133) & (cr <= 180) & (cb >= 80) & (cb <= 135)] = 255
        skin_mask = skin_mask.astype(np.uint8)

        # 模糊掩码（让边缘过渡自然）
        skin_mask = cv2.erode(skin_mask, np.ones((5, 5), np.uint8), iterations=1)
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)

        # 只在上半部分（脸部区域）应用腮红
        face_mask = skin_mask.copy()
        face_region_top = int(h * 0.55)  # 只在画面上半部
        face_mask[:face_region_top, :] = 0  # 清除过高的区域

        # ========== 腮红 ==========
        if blush_strength > 0:
            # 腮红颜色：偏粉红
            blush_hsv = np.array([0, 80, 120], dtype=np.float32)  # HSV
            # 转 BGR
            pink_patch = np.zeros((1, 1, 3), dtype=np.uint8)
            pink_patch[0, 0] = [blush_hsv[0], blush_hsv[1], blush_hsv[2]]
            # 简单转换估算
            blush_bgr = np.array([[[120, 80, 180]]], dtype=np.uint8)  # 偏红的颜色

            # 在掩码区域内叠加粉色
            overlay = frame.copy().astype(np.float32)
            blush_color = np.array([80, 60, 200], dtype=np.float32)  # BGR 偏粉
            for c in range(3):
                overlay[:, :, c] = np.clip(
                    overlay[:, :, c] + blush_color[c] * blush_strength * 0.3,
                    0, 255
                )
            frame = np.clip(overlay, 0, 255).astype(np.uint8)
            # 叠加到皮肤掩码区域
            mask_f = face_mask.astype(np.float32) / 255.0 * blush_strength * 0.15
            for c in range(3):
                frame[:, :, c] = np.clip(
                    frame[:, :, c] + (blush_color[c] - frame[:, :, c]) * mask_f,
                    0, 255
                ).astype(np.uint8)

        # ========== 局部美白 ==========
        if brighten_strength > 0:
            # 只在皮肤区域提亮
            bright_mask = skin_mask.astype(np.float32) / 255.0 * brighten_strength
            frame_f = frame.astype(np.float32)
            # V通道提亮
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + bright_mask * 0.5), 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return frame

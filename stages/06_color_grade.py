"""阶段06: 色彩处理

亮度/对比度/饱和度/色温/CLAHE 自适应直方图均衡。
"""

import cv2
import numpy as np
from pathlib import Path

import ctypes, subprocess, shutil, tempfile, os


class ColorGradeStage:
    def run(self, ctx):
        # 增量跳过：输出已存在则跳过
        if ctx.get("color_path") and Path(ctx.get("color_path")).exists():
            print("    已存在，跳过")
            return

        # 找到输入视频
        input_path = (ctx.get("face_warp_path") or ctx.get("ken_burns_path") or
                      ctx.get("warped_path") or ctx.get("h2v_path") or
                      str(ctx.input_path))  # 横屏 fallback
        if not input_path or not Path(input_path).exists():
            print("    跳过: 无可处理的视频")
            return

        video_info = ctx.get("video_info")
        h2v_size = ctx.get("h2v_size")
        crop_w, crop_h = h2v_size if h2v_size else (video_info["width"], video_info["height"])
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])
        cfg = ctx.config.get("color_grade", {})

        brightness = cfg.get("brightness", 0)
        contrast = cfg.get("contrast", 1.0)
        saturation = cfg.get("saturation", 1.0)
        warmth = cfg.get("warmth", 0)
        use_clahe = cfg.get("clahe", False)
        shadow = cfg.get("shadow", 0)
        auto_wb = cfg.get("auto_wb", False)
        adaptive_contrast = cfg.get("adaptive_contrast", 0)

        needs_grade = any(v != 0 and v != 1.0 and v is not False
                          for k, v in cfg.items()
                          if k in ("brightness", "contrast", "saturation", "warmth",
                                   "clahe", "shadow", "auto_wb", "adaptive_contrast"))
        if not needs_grade:
            print("    跳过: 无色彩参数")
            ctx.set("color_path", input_path)
            return

        print(f"    参数: bright={brightness}, contrast={contrast:.2f}, "
              f"sat={saturation:.2f}, warm={warmth}, clahe={use_clahe}, "
              f"shadow={shadow}, auto_wb={auto_wb}, ad_contrast={adaptive_contrast}")

        # 创建 CLAHE (只创建一次)
        clahe_obj = None
        if use_clahe:
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        out_path = ctx.output_dir / f"{Path(input_path).stem}_color.mp4"
        tmp_fd, tmp_path_tmp = tempfile.mkstemp(suffix='.mp4')
        os.close(tmp_fd)
        tmp_path_tmp = Path(tmp_path_tmp)
        tmpdir = Path(tempfile.mkdtemp(prefix="cg_"))

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

        frame_idx = 0
        prev_frame = None
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 亮度
            if brightness != 0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness * 2.55, 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 2. 对比度
            if contrast != 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)

            # 3. 饱和度
            if saturation != 1.0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 4. 色温 (warm > 0 暖, < 0 冷)
            if warmth != 0:
                result = frame.astype(np.float32)
                result[:, :, 2] = np.clip(result[:, :, 2] + warmth * 0.5, 0, 255)  # R+
                result[:, :, 0] = np.clip(result[:, :, 0] - warmth * 0.3, 0, 255)  # B-
                frame = result.astype(np.uint8)

            # 5. 阴影修正 (提亮暗部，不影响亮部)
            if shadow > 0:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                shadow_mask = np.clip((80 - l) / 80, 0, 1) * shadow
                l = np.clip(l + shadow_mask * 40, 0, 255).astype(np.uint8)
                lab = cv2.merge([l, a, b])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 6. 自动白平衡 (gray world)
            if auto_wb:
                result = frame.astype(np.float32)
                avg_b = np.mean(result[:, :, 0])
                avg_g = np.mean(result[:, :, 1])
                avg_r = np.mean(result[:, :, 2])
                gray = (avg_b + avg_g + avg_r) / 3
                result[:, :, 0] = np.clip(result[:, :, 0] * gray / max(avg_b, 1), 0, 255)
                result[:, :, 1] = np.clip(result[:, :, 1] * gray / max(avg_g, 1), 0, 255)
                result[:, :, 2] = np.clip(result[:, :, 2] * gray / max(avg_r, 1), 0, 255)
                frame = result.astype(np.uint8)

            # 7. 自适应对比度 (直方图裁剪)
            if adaptive_contrast > 0:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe_ac = cv2.createCLAHE(
                    clipLimit=adaptive_contrast * 4.0,
                    tileGridSize=(4, 4)
                )
                l = clahe_ac.apply(l)
                lab = cv2.merge([l, a, b])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 8. CLAHE 自适应直方图均衡 (提亮暗部)
            if clahe_obj is not None:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 9. 锐化 (unsharp mask)
            sharpen = cfg.get("sharpen", 0)
            if sharpen > 0:
                blurred = cv2.GaussianBlur(frame, (0, 0), 3)
                frame = cv2.addWeighted(frame, 1.0 + sharpen, blurred, -sharpen, 0)

            # 10. 时间平滑 (帧间混合，消除色块跳动)
            temporal_smooth = cfg.get("temporal_smooth", 0)
            if temporal_smooth > 0 and prev_frame is not None:
                frame = cv2.addWeighted(frame, 1.0 - temporal_smooth,
                                        prev_frame, temporal_smooth, 0)
            prev_frame = frame.copy()

            fname = f"{tmpdir_short}/f_{frame_idx:06d}.png"
            cv2.imwrite(fname, frame)
            frame_idx += 1
            if frame_idx % 30 == 0:
                pct = frame_idx / max_frames * 100
                print(f"    进度: {pct:.0f}% ({frame_idx}/{max_frames})")

        cap.release()

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
            raise RuntimeError(f"FFmpeg 编码失败: {r.stderr[:200]}")
        print(f"    编码完成")

        try:
            shutil.move(str(tmp_path_tmp), str(out_path))
        except Exception:
            alt_path = ctx.output_dir / f"{Path(input_path).stem}_color_new.mp4"
            shutil.move(str(tmp_path_tmp), str(alt_path))
            out_path = alt_path

        ctx.set("color_path", str(out_path))
        print(f"    输出: {Path(out_path).name}")

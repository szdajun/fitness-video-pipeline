"""阶段06: 色彩处理

亮度/对比度/饱和度/色温/CLAHE 自适应直方图均衡。
"""

import cv2
from lib.utils import path_exists
from lib.highlight_protect import optimize_night_highlights
from lib.lut_utils import load_cube, apply_lut, get_builtin_lut, write_lut_cube
import numpy as np
from pathlib import Path

import ctypes, subprocess, shutil, tempfile, os


class ColorGradeStage:
    def run(self, ctx):
        # 增量跳过：输出已存在则跳过
        if ctx.get("color_path") and path_exists(ctx.get("color_path")):
            print("    已存在，跳过")
            return

        # 找到输入视频
        input_path = (ctx.get("face_warp_path") or ctx.get("ken_burns_path") or
                      ctx.get("warped_path") or ctx.get("h2v_path") or
                      str(ctx.input_path))  # 横屏 fallback
        if not input_path or not path_exists(input_path):
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
                                   "clahe", "shadow", "auto_wb", "adaptive_contrast",
                                   "vignette_strength", "film_grain_strength"))
        if not needs_grade:
            lut_path = cfg.get("lut_path", "")
            lut_preset = cfg.get("lut_preset", "")
            if lut_path or lut_preset:
                needs_grade = True
        if not needs_grade:
            print("    跳过: 无色彩参数")
            ctx.set("color_path", input_path)
            return

        print(f"    参数: bright={brightness}, contrast={contrast:.2f}, "
              f"sat={saturation:.2f}, warm={warmth}, clahe={use_clahe}, "
              f"shadow={shadow}, auto_wb={auto_wb}, ad_contrast={adaptive_contrast}")

        # ==== LUT 预加载 + FFmpeg 加速检测 ====
        lut_path = cfg.get("lut_path", "")
        lut_intensity = cfg.get("lut_intensity", 1.0)
        lut_preset = cfg.get("lut_preset", "")
        skin_protect = cfg.get("skin_protect", 0)
        use_ffmpeg_lut = False

        if lut_preset or lut_path:
            _lut_data = None
            if lut_preset:
                try:
                    _lut_data, _ = get_builtin_lut(lut_preset)
                    print(f"    加载 LUT 预设: {lut_preset}")
                except ValueError as e:
                    print(f"    警告: {e}")
            elif Path(lut_path).exists():
                try:
                    _lut_data, _ = load_cube(lut_path)
                    print(f"    加载 LUT 文件: {lut_path}")
                except Exception as e:
                    print(f"    警告: LUT 加载失败: {e}")

            if _lut_data is not None and lut_intensity > 0:
                if skin_protect == 0:
                    use_ffmpeg_lut = True
                    self._lut_data_for_ffmpeg = _lut_data
                    print(f"    LUT 加速: FFmpeg lut3d 编码时应用")
                else:
                    self._current_lut = _lut_data

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

            # 2. 自动曝光补偿 (场景自适应亮度)
            auto_exposure = cfg.get("auto_exposure", 0)
            if auto_exposure > 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_mean = np.mean(gray)
                ae_target = cfg.get("ae_target", 128)
                ae_speed = cfg.get("ae_speed", 0.05)
                error = ae_target - current_mean
                if abs(error) > 3:  # 死区避免微颤
                    if not hasattr(self, '_ae_integral'):
                        self._ae_integral = 0.0
                    self._ae_integral = (self._ae_integral * (1 - ae_speed) +
                                         error * ae_speed)
                    comp = self._ae_integral * auto_exposure * 0.4
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + comp, 0, 255)
                    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 3. 对比度
            if contrast != 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)

            # 4. 饱和度
            if saturation != 1.0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # 5. 色温 (warm > 0 暖, < 0 冷)
            if warmth != 0:
                result = frame.astype(np.float32)
                result[:, :, 2] = np.clip(result[:, :, 2] + warmth * 0.5, 0, 255)  # R+
                result[:, :, 0] = np.clip(result[:, :, 0] - warmth * 0.3, 0, 255)  # B-
                frame = result.astype(np.uint8)

            # 6. 阴影修正 (提亮暗部，不影响亮部)
            if shadow > 0:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                shadow_mask = np.clip((80 - l) / 80, 0, 1) * shadow
                l = np.clip(l + shadow_mask * 40, 0, 255).astype(np.uint8)
                lab = cv2.merge([l, a, b])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 7. 自动白平衡 (gray world)
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

            # 8. 自适应对比度 (场景感知)
            if adaptive_contrast > 0:
                # 分析场景亮度：计算平均亮度
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray) / 255.0  # 0=黑, 1=白

                if mean_brightness < 0.35:
                    # 暗场：强提亮暗部
                    clip_limit = adaptive_contrast * 5.0
                    tile_size = (4, 4)
                elif mean_brightness > 0.65:
                    # 亮场：保护高光，轻度对比度
                    clip_limit = adaptive_contrast * 2.0
                    tile_size = (8, 8)
                else:
                    # 正常场：标准处理
                    clip_limit = adaptive_contrast * 3.5
                    tile_size = (6, 6)

                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe_ac = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
                l = clahe_ac.apply(l)

                # 根据亮度调整 L 通道
                if mean_brightness < 0.35:
                    l = np.clip(l * 1.15, 0, 255).astype(np.uint8)
                elif mean_brightness > 0.65:
                    l = np.clip(l * 0.92, 0, 255).astype(np.uint8)

                lab = cv2.merge([l, a, b])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 9. 夜景高光三层处理（压灯区/护白衣/抑灯斑）
            night_highlight_cfg_raw = {
                "highlight_protect": cfg.get("highlight_protect", 0.0),
                "highlight_threshold": cfg.get("highlight_threshold", 185),
                "highlight_blur": cfg.get("highlight_blur", 5),
                "white_protect": cfg.get("white_protect", 0.0),
                "white_value_threshold": cfg.get("white_value_threshold", 200),
                "white_sat_threshold": cfg.get("white_sat_threshold", 60),
                "white_protect_blur": cfg.get("white_protect_blur", 5),
                "light_region_protect": cfg.get("light_region_protect", 0.0),
                "light_region_threshold": cfg.get("light_region_threshold", 235),
                "light_region_min_area": cfg.get("light_region_min_area", 2500),
                "light_region_blur": cfg.get("light_region_blur", 21),
            }
            if any(v != 0.0 for k, v in night_highlight_cfg_raw.items() if "protect" in k or "threshold" in k):
                frame = optimize_night_highlights(frame, night_highlight_cfg_raw)

            # 10. CLAHE 自适应直方图均衡 (提亮暗部)
            if clahe_obj is not None:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 11. 运动自适应锐化 (unsharp mask + 帧差法)
            sharpen = cfg.get("sharpen", 0)
            if sharpen > 0:
                blurred = cv2.GaussianBlur(frame, (0, 0), 3)
                if prev_frame is not None:
                    # 帧差法检测运动区域：运动越大锐化越强（补偿运动模糊）
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    diff_smooth = cv2.GaussianBlur(diff_gray, (15, 15), 5)
                    motion = diff_smooth.astype(np.float32) / 255.0  # 0~1
                    # 运动区域: 强度×3, 静止区域: 强度×0.3
                    adapt_map = sharpen * (0.3 + motion * 2.7)
                    adapt_map = np.clip(adapt_map, sharpen * 0.3, sharpen * 3.0)
                    mask_3ch = np.stack([adapt_map] * 3, axis=-1)
                    frame = np.clip(
                        frame.astype(np.float32) * (1.0 + mask_3ch) -
                        blurred.astype(np.float32) * mask_3ch,
                        0, 255
                    ).astype(np.uint8)
                else:
                    # 首帧：全局锐化
                    frame = cv2.addWeighted(frame, 1.0 + sharpen, blurred, -sharpen, 0)

            # 12. 时间平滑 (帧间混合，消除色块跳动)
            temporal_smooth = cfg.get("temporal_smooth", 0)
            if temporal_smooth > 0 and prev_frame is not None:
                frame = cv2.addWeighted(frame, 1.0 - temporal_smooth,
                                        prev_frame, temporal_smooth, 0)
            prev_frame = frame.copy()

            # 13. 暗角 (vignette) — 预计算径向掩码
            vignette_strength = cfg.get("vignette_strength", 0)
            if vignette_strength > 0:
                if not hasattr(self, '_vignette_mask'):
                    h, w = frame.shape[:2]
                    cx, cy = w / 2, h / 2
                    dist = np.sqrt((np.arange(w)[None, :] - cx) ** 2 +
                                   (np.arange(h)[:, None] - cy) ** 2)
                    max_dist = np.sqrt(cx ** 2 + cy ** 2)
                    vignette_radius = cfg.get("vignette_radius", 0.75)
                    vignette_feather = cfg.get("vignette_feather", 0.3)
                    r_thresh = max_dist * vignette_radius
                    feather = max_dist * vignette_feather
                    mask = np.clip((dist - r_thresh) / max(feather, 1), 0, 1)
                    self._vignette_mask = (1.0 - mask * vignette_strength).astype(np.float32)
                frame = (frame.astype(np.float32) * self._vignette_mask[:, :, None]).astype(np.uint8)

            # 14. 胶片颗粒 (film grain) — 确定性噪声避免闪烁
            film_grain_strength = cfg.get("film_grain_strength", 0)
            if film_grain_strength > 0:
                rng = np.random.RandomState(frame_idx)
                h, w = frame.shape[:2]
                grain_size = cfg.get("film_grain_size", 2)
                noise = rng.randn(h, w, 3).astype(np.float32) * film_grain_strength * 60
                if grain_size > 1:
                    noise = cv2.GaussianBlur(noise, (grain_size * 2 + 1,) * 2, grain_size * 0.5)
                frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # 15. 3D LUT 调色 (Python fallback：FFmpeg 加速时跳过)
            if hasattr(self, '_current_lut') and not use_ffmpeg_lut:
                lut_frame = apply_lut(frame, self._current_lut, lut_intensity)
                # 肤色保护: LUT 调色后还原肤色区域，避免偏色
                if skin_protect > 0:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    lower = np.array([0, 15, 40])
                    upper = np.array([25, 150, 255])
                    skin_mask = cv2.inRange(hsv, lower, upper)
                    skin_mask = cv2.GaussianBlur(skin_mask.astype(np.float32),
                                                  (31, 31), 15) / 255.0
                    strength = (1.0 - skin_mask * skin_protect)[..., None]
                    frame = (lut_frame.astype(np.float32) * strength +
                             frame.astype(np.float32) * (1 - strength)).astype(np.uint8)
                else:
                    frame = lut_frame

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
               "-i", f"{tmpdir_short}/f_%06d.png"]
        # LUT FFmpeg 加速：通过 lut3d 滤镜在编码时应用 LUT（用相对路径避免 Windows 盘符问题）
        if use_ffmpeg_lut and hasattr(self, '_lut_data_for_ffmpeg'):
            cube_path = tmpdir / "_color_lut.cube"
            write_lut_cube(self._lut_data_for_ffmpeg, lut_intensity, path=str(cube_path))
            cube_rel = os.path.relpath(str(cube_path), os.getcwd()).replace("\\", "/")
            cmd += ["-vf", f"lut3d={cube_rel}"]
            print(f"    LUT FFmpeg 加速: lut3d={cube_rel}")
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "1",
                "-pix_fmt", "yuv444p", "-an", str(tmp_path_tmp)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"    FFmpeg 错误: {r.stderr[:500]}")
            raise RuntimeError(f"FFmpeg 编码失败: {r.stderr[:500]}")
        print(f"    编码完成")

        try:
            shutil.move(str(tmp_path_tmp), str(out_path))
        except Exception:
            alt_path = ctx.output_dir / f"{Path(input_path).stem}_color_new.mp4"
            shutil.move(str(tmp_path_tmp), str(alt_path))
            out_path = alt_path

        ctx.set("color_path", str(out_path))
        print(f"    输出: {Path(out_path).name}")

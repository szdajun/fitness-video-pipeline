"""阶段04: Ken Burns 运镜效果

在竖版画面内添加微缩放/平移，增加视频动感。
支持两种模式:
  - smooth: 正弦函数微幅运镜
  - dual: 全景/特写双场景平滑切换（需要 H2V 加宽裁切）

注意: body_warp 在本阶段之前运行，本阶段只做画面裁切，不影响体型。
"""

import os
import cv2
import numpy as np
import subprocess
import shutil
from pathlib import Path

from lib.utils import path_exists, create_writer


class KenBurnsStage:
    def run(self, ctx):
        # 增量跳过：输出已存在则跳过
        if ctx.get("ken_burns_path") and path_exists(ctx.get("ken_burns_path")):
            print("    已存在，跳过")
            return

        # 优先读 warped_path（body_warp 已处理），否则 h2v_path，再否则 stabilized_path
        # 横屏模式：也支持直接用原始输入视频（无 h2v/warp 中间步骤）
        input_path = (ctx.get("warped_path") or
                      ctx.get("h2v_path") or
                      ctx.get("stabilized_path") or
                      str(ctx.input_path))
        if not input_path or not path_exists(input_path):
            print("    跳过: 无输入视频")
            return

        video_info = ctx.get("video_info")
        h2v_size = ctx.get("h2v_size")
        if h2v_size is None or h2v_size[0] == 0 or h2v_size[1] == 0:
            # h2v_size 无效（_scan_existing_outputs 读到 corrupt 0x0 文件），从实际输入重新读取
            fallback_src = input_path
            cap_check = cv2.VideoCapture(fallback_src)
            if cap_check.isOpened():
                w = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
                h2v_size = (w, h)
                cap_check.release()
                print(f"    警告: h2v_size 无效({ctx.get('h2v_size')})，从输入重新读取: {h2v_size}")
            else:
                cap_check.release()
                print(f"    跳过: 无法读取视频尺寸: {fallback_src}")
                return
        crop_w, crop_h = h2v_size
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])
        cfg = ctx.config.get("ken_burns", {})
        mode = cfg.get("mode", "smooth")
        # 记录输入尺寸供 dual mode 使用
        input_w, input_h = crop_w, crop_h

        # 检测最终输出分辨率，决定裁切比例
        out_cfg = ctx.config.get("output", {})
        out_w = out_cfg.get("width")
        out_h = out_cfg.get("height")
        if out_w and out_h:
            is_vertical = out_h > out_w
        else:
            is_vertical = True  # 默认竖版

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"    ken_burns 输入: {input_path}, h2v_size={h2v_size}, video_info frames={video_info.get('frames')}, max_frames={max_frames}, 实际帧数={actual_frames}")
        if actual_frames != max_frames:
            print(f"    警告: 实际帧数 {actual_frames} 与预期 {max_frames} 不符，使用实际帧数")
            max_frames = actual_frames

        # 输出文件名基于输入 stem，加上分辨率后缀避免覆盖
        stem = Path(input_path).stem
        if mode == "dual":
            # 根据目标宽高比决定裁切尺寸
            if is_vertical:
                # 竖版 9:16：裁切宽度 = height * 9/16（和原来一样）
                target_h = crop_h
                target_w = int(target_h * 9.0 / 16.0)
                ratio_suffix = "_9x16"
            else:
                # 横版 16:9：输出高度 = 输入宽度（即竖版裁切后的宽度，作为横版的高度）
                # 这样横版和竖版用同一个输入视频，crop 不会出界
                target_h = crop_w
                target_w = int(target_h * 16.0 / 9.0)
                ratio_suffix = "_16x9"
            target_w = target_w if target_w % 2 == 0 else target_w - 1
            out_path = ctx.output_dir / f"{stem}_kenburns{ratio_suffix}.mp4"
            # 用系统临时目录避免输出目录被锁的文件影响
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(tmp_fd)
            tmp_path = Path(tmp_path)
            cropped_keypoints = ctx.get("cropped_keypoints", {})
            if not cropped_keypoints:
                # 从 JSON 文件加载（stage 增量跳过场景）
                import json as _json
                kp_file = Path(ctx.output_dir) / f"{Path(ctx.input_path).stem}_cropped_keypoints.json"
                if kp_file.exists():
                    with open(kp_file) as f:
                        cropped_keypoints = _json.load(f)
                    print(f"    从文件加载关键点: {len(cropped_keypoints)} 帧")
            self._run_dual_ffmpeg(cap, str(tmp_path), crop_w, crop_h, target_w, max_frames, fps, cfg,
                           cropped_keypoints, is_vertical)
            # 完成后移动到最终路径（shutil.move 跨驱动器会做 copy+delete）
            try:
                shutil.move(str(tmp_path), str(out_path))
            except Exception:
                # 目标被锁或跨驱动器失败：改用不同文件名
                alt_path = ctx.output_dir / f"{stem}_kenburns{ratio_suffix}_new.mp4"
                shutil.move(str(tmp_path), str(alt_path))
                ctx.set("ken_burns_path", str(alt_path))
                print(f"    注意: {out_path.name} 被占用，输出改为 {alt_path.name}")
            ctx.set("h2v_size", (target_w, crop_h))
            ctx.set("ken_burns_ratio", ratio_suffix)
        else:
            out_path = ctx.output_dir / f"{stem}_kenburns.mp4"
            import tempfile, ctypes
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
            tmpdir = Path(tempfile.mkdtemp(prefix="kb_smooth_"))
            tmpdir_short = to_short(str(tmpdir))
            tmp_fd, tmp_out_path = tempfile.mkstemp(suffix='.mp4')
            os.close(tmp_fd)
            tmp_out_path = Path(tmp_out_path)
            tmp_out_short = to_short(str(tmp_out_path))
            fps_val = fps
            self._run_smooth_ffmpeg(cap, str(tmpdir_short), str(tmp_out_short), crop_w, crop_h, max_frames, cfg, fps_val)
            shutil.move(str(tmp_out_path), str(out_path))
            shutil.rmtree(tmpdir, ignore_errors=True)
            ctx.set("ken_burns_ratio", "")

        cap.release()
        if 'writer' in locals():
            writer.release()
        ctx.set("ken_burns_path", str(out_path))
        print(f"    输出: {out_path.name}")

    def _run_smooth(self, cap, writer, crop_w, crop_h, max_frames, cfg):
        """smooth 模式: 正弦微幅运镜"""
        zoom_range = cfg.get("zoom_range", [1.0, 1.05])
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / max_frames
            zoom = zoom_range[0] + (zoom_range[1] - zoom_range[0]) * (0.5 + 0.5 * np.sin(t * 2 * np.pi))
            pan_x = int(5 * np.sin(t * 3 * np.pi))
            pan_y = 0  # 竖版不做垂直移动，防止底部砖缝线被裁切产生"鼓起"视觉效果

            h, w = frame.shape[:2]
            new_w, new_h = int(w * zoom), int(h * zoom)
            scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            cx = (new_w - w) // 2 + pan_x
            cy = (new_h - h) // 2
            cx = max(0, min(cx, new_w - w))
            cy = max(0, min(cy, new_h - h))
            frame = scaled[cy:cy + h, cx:cx + w]

            writer.write(frame)
            frame_idx += 1

    def _run_smooth_ffmpeg(self, cap, tmpdir_short, tmp_out_short, crop_w, crop_h, max_frames, cfg, fps):
        """smooth 模式: PNG 序列 + FFmpeg 编码"""
        zoom_range = cfg.get("zoom_range", [1.0, 1.05])
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / max_frames
            zoom = zoom_range[0] + (zoom_range[1] - zoom_range[0]) * (0.5 + 0.5 * np.sin(t * 2 * np.pi))
            pan_x = int(5 * np.sin(t * 3 * np.pi))
            pan_y = 0  # 竖版不做垂直移动，防止底部砖缝线被裁切产生"鼓起"视觉效果

            h, w = frame.shape[:2]
            new_w, new_h = int(w * zoom), int(h * zoom)
            scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            cx = (new_w - w) // 2 + pan_x
            cy = (new_h - h) // 2
            cx = max(0, min(cx, new_w - w))
            cy = max(0, min(cy, new_h - h))
            frame_cropped = scaled[cy:cy + h, cx:cx + w]

            fname = f"{tmpdir_short}/f_{frame_idx:06d}.png"
            cv2.imwrite(fname, frame_cropped)
            frame_idx += 1

        ffmpeg_bin = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        cmd = [ffmpeg_bin, "-y", "-v", "info",
               "-framerate", str(fps),
               "-i", f"{tmpdir_short}/f_%06d.png",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-an", tmp_out_short]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg smooth error: {r.stderr[-300:]}")

    def _run_dual(self, cap, writer, input_w, input_h, target_w, max_frames, fps, cfg,
                  cropped_keypoints, is_vertical=True):
        """dual mode: dual scene switching with near-equal aspect scaling.

        Maintains stable body proportions by tracking lead center position.
        Vertical mode tracks subject vertically; horizontal mode aligns to bottom.
        Motion response: zoom out when active, zoom in when stationary.
        """
        close_zoom_h = cfg.get("dual_close_zoom", 1.1)
        close_zoom_v = cfg.get("dual_close_zoom_v", 1.05)
        cycle_seconds = cfg.get("dual_cycle_seconds", 8)
        pan_amplitude = cfg.get("dual_pan_amplitude", 8)
        motion_response = cfg.get("dual_motion_response", 0.5)  # 运动响应系数
        dual_dwell = cfg.get("dual_dwell", 0.3)  # 特写停留系数 (0=无, 0.4=强)
        motion_zoom_response = cfg.get("dual_motion_zoom_response", 0.3)  # 运动对zoom的影响
        target_h = input_h

        total_time = max_frames / fps
        frame_idx = 0
        smooth_cx = None  # 帧间平滑裁切中心
        smooth_cy = None  # 帧间平滑裁切纵向中心
        prev_lead_cx = None  # 上一帧领操人水平中心（归一化）
        prev_lead_cy = None  # 上一帧领操人纵向中心（归一化）
        motion_smooth = 0.0  # 平滑后的运动量
        prev_scene_factor = 0.0  # 上一帧 scene_factor（用于平滑）

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            time_s = frame_idx / max_frames * total_time

            # -cos 振荡: t=0 时全景(0)，周期中点特写(1)
            raw_sf = 0.5 - 0.5 * np.cos(2 * np.pi * time_s / cycle_seconds)
            # 特写停留: 幂曲线使峰值/谷值区间拉长
            dwell_power = 1.0 / (1.0 + dual_dwell * 2.0)
            scene_factor = np.power(raw_sf, dwell_power)

            # 从关键点获取领操人位置
            lead_cx_norm = self._get_lead_center_x(cropped_keypoints, frame_idx, input_w)
            lead_cy_norm = self._get_lead_center_y(cropped_keypoints, frame_idx, input_h)

            # 计算运动量（领操人帧间位移）
            raw_motion = 0.0
            if prev_lead_cx is not None:
                raw_motion = abs(lead_cx_norm - prev_lead_cx) + abs(lead_cy_norm - prev_lead_cy)
            prev_lead_cx = lead_cx_norm
            prev_lead_cy = lead_cy_norm

            # 指数平滑运动量
            motion_smooth = motion_smooth * 0.7 + raw_motion * 0.3
            # 运动响应系数：将运动量映射到 pan 增益 [0.3, 1.5]
            motion_gain = 0.3 + min(motion_smooth * 30, 1.0) * motion_response * 1.5

            # 运动响应 zoom：运动大时略缩小（显示更多），静止时略放大
            # zoom 在 scene_factor 基础上有 motion 调制
            motion_zoom_bias = 1.0 - motion_smooth * motion_zoom_response * 2.0
            motion_zoom_bias = max(0.9, min(motion_zoom_bias, 1.05))

            # 横纵分别缩放，比例接近，体型视觉稳定
            # zoom 基础值：确保 scaled 尺寸 ≥ target 尺寸
            zoom_base_h = max(1.0, target_w / input_w)
            zoom_base_v = max(1.0, target_h / input_h)
            zoom_h = zoom_base_h * (1.0 + (close_zoom_h - 1.0) * scene_factor) * motion_zoom_bias
            zoom_v = zoom_base_v * (1.0 + (close_zoom_v - 1.0) * scene_factor) * motion_zoom_bias

            h, w = frame.shape[:2]
            new_w = int(w * zoom_h)
            new_h = int(h * zoom_v)
            scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # 水平裁切中心: 跟踪领操人位置
            lead_cx_pixel = lead_cx_norm * new_w  # 缩放后的像素位置
            cx = int(lead_cx_pixel - target_w / 2)

            # 帧间平滑
            if smooth_cx is not None:
                cx = int(smooth_cx * 0.85 + cx * 0.15)
            smooth_cx = cx

            # 微平移：受 scene_factor 和运动量双重调制
            base_pan = pan_amplitude * (0.5 + 0.5 * scene_factor)
            pan_x = int(base_pan * motion_gain *
                        np.sin(2.5 * np.pi * time_s / cycle_seconds))
            max_pan_right = new_w - (cx + target_w)
            max_pan_left = -cx
            pan_x = max(max_pan_right, min(pan_x, max_pan_left))
            if is_vertical:
                pan_x = 0  # 竖版不做水平pan
            cx += pan_x

            # 纵向裁切: 竖版固定水平中心，横版底部对齐
            if is_vertical:
                # 竖版：cy 提高到 30%，地面直接裁在画面之外，底部 margin 25%
                cy_target = int(new_h * 0.30)
                if smooth_cy is not None:
                    cy_target = int(smooth_cy * 0.8 + cy_target * 0.2)
                smooth_cy = cy_target
                bottom_margin = int(target_h * 0.25)
                cy = max(bottom_margin, min(cy_target, new_h - target_h - bottom_margin))
            else:
                cy = new_h - target_h
            cy = max(0, min(cy, new_h - target_h))

            cropped = scaled[cy:cy + target_h, cx:cx + target_w].copy()

            if frame_idx == 0:
                print(f"    target: {target_w}x{target_h}, input: {input_w}x{input_h}, vertical_track={is_vertical}")

            writer.write(cropped)
            frame_idx += 1

        print(f"    _run_dual 实际写入帧数: {frame_idx}")

    def _run_dual_ffmpeg(self, cap, output_path, input_w, input_h, target_w, max_frames, fps, cfg,
                         cropped_keypoints, is_vertical=True):
        """dual 模式: 临时图片文件 + FFmpeg concat 可靠编码"""
        import subprocess, shutil, ctypes, tempfile, os

        ffmpeg_bin = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
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

        target_h = input_h
        tmpdir = Path(tempfile.mkdtemp(prefix="kb_"))
        tmpdir_short = to_short(str(tmpdir))

        close_zoom_h = cfg.get("dual_close_zoom", 1.1)
        close_zoom_v = cfg.get("dual_close_zoom_v", 1.05)
        cycle_seconds = cfg.get("dual_cycle_seconds", 8)
        pan_amplitude = cfg.get("dual_pan_amplitude", 8)
        motion_response = cfg.get("dual_motion_response", 0.5)
        dual_dwell = cfg.get("dual_dwell", 0.3)
        motion_zoom_response = cfg.get("dual_motion_zoom_response", 0.3)

        total_time = max_frames / fps
        frame_idx = 0
        smooth_cx = None
        smooth_cy = None  # 新增 cy 平滑
        prev_lead_cx = None
        prev_lead_cy = None
        motion_smooth = 0.0

        print(f"    target: {target_w}x{target_h}, input: {input_w}x{input_h}, vertical_track={is_vertical}")
        print(f"    临时目录: {tmpdir.name}")

        BATCH = 100

        while frame_idx < max_frames:
            batch_frames = []
            batch_indices = []
            for _ in range(BATCH):
                ret, frame = cap.read()
                if not ret:
                    break

                time_s = frame_idx / max_frames * total_time
                raw_sf = 0.5 - 0.5 * np.cos(2 * np.pi * time_s / cycle_seconds)
                dwell_power = 1.0 / (1.0 + dual_dwell * 2.0)
                scene_factor = np.power(raw_sf, dwell_power)

                lead_cx_norm = self._get_lead_center_x(cropped_keypoints, frame_idx, input_w)
                lead_cy_norm = self._get_lead_center_y(cropped_keypoints, frame_idx, input_h)

                raw_motion = 0.0
                if prev_lead_cx is not None:
                    raw_motion = abs(lead_cx_norm - prev_lead_cx) + abs(lead_cy_norm - prev_lead_cy)
                prev_lead_cx = lead_cx_norm
                prev_lead_cy = lead_cy_norm
                motion_smooth = motion_smooth * 0.7 + raw_motion * 0.3
                motion_gain = 0.3 + min(motion_smooth * 30, 1.0) * motion_response * 1.5
                motion_zoom_bias = 1.0 - motion_smooth * motion_zoom_response * 2.0
                motion_zoom_bias = max(0.9, min(motion_zoom_bias, 1.05))

                # zoom 基础值：确保 scaled 尺寸 ≥ target 尺寸
                zoom_base_h = max(1.0, target_w / input_w)
                zoom_base_v = max(1.0, target_h / input_h)
                zoom_h = zoom_base_h * (1.0 + (close_zoom_h - 1.0) * scene_factor) * motion_zoom_bias
                zoom_v = zoom_base_v * (1.0 + (close_zoom_v - 1.0) * scene_factor) * motion_zoom_bias

                h, w = frame.shape[:2]
                new_w = int(w * zoom_h)
                new_h = int(h * zoom_v)
                scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

                lead_cx_pixel = lead_cx_norm * new_w
                cx = int(lead_cx_pixel - target_w / 2)
                if smooth_cx is not None:
                    cx = int(smooth_cx * 0.93 + cx * 0.07)
                smooth_cx = cx

                base_pan = pan_amplitude * (0.5 + 0.5 * scene_factor)
                pan_x = int(base_pan * motion_gain * np.sin(2.5 * np.pi * time_s / cycle_seconds))
                max_pan_right = new_w - (cx + target_w)
                max_pan_left = -cx
                pan_x = max(max_pan_right, min(pan_x, max_pan_left))
                if is_vertical:
                    pan_x = 0
                cx += pan_x
                cx = max(0, min(cx, new_w - target_w))

                if is_vertical:
                    # 竖版：cy 提高到 30%，地面直接裁在画面之外
                    # 底部 margin=25%，确保即使 zoom 大时底部也不会显示地面区域
                    cy_target = int(new_h * 0.30)
                    if smooth_cy is not None:
                        cy_target = int(smooth_cy * 0.85 + cy_target * 0.15)  # 更强平滑
                    smooth_cy = cy_target
                    bottom_margin = int(target_h * 0.25)
                    cy = max(bottom_margin, min(cy_target, new_h - target_h - bottom_margin))
                else:
                    cy = new_h - target_h
                cy = max(0, min(cy, new_h - target_h))

                # 确保 crop 不超出 scaled 边界（scaled 可能因 zoom<1 而比 target 小）
                crop_right = min(cx + target_w, scaled.shape[1])
                crop_bottom = min(cy + target_h, scaled.shape[0])
                cropped = scaled[cy:crop_bottom, cx:crop_right]

                if frame_idx < 5:
                    print(f"    DEBUG frame {frame_idx}: scaled={scaled.shape}, cx={cx}, cy={cy}, target={target_w}x{target_h}, crop_right={crop_right}, cropped={cropped.shape}")
                batch_frames.append(cropped)
                batch_indices.append(frame_idx)
                frame_idx += 1

            # 批量写入 PNG
            for fi, cropped in zip(batch_indices, batch_frames):
                fname = f"{tmpdir_short}/f_{fi:06d}.png"
                cv2.imwrite(fname, cropped)

            # 每批后验证第一个PNG的尺寸（用 tmpdir 完整路径，不用 short path）
            if frame_idx >= 500 and frame_idx < 600:
                test_png = str(tmpdir / f"f_{batch_indices[0]:06d}.png")
                import cv2 as _cv2
                _test = _cv2.imread(test_png)
                if _test is not None:
                    print(f"    PNG {batch_indices[0]}: shape={_test.shape}")

            if frame_idx % 500 == 0:
                pct = frame_idx / max_frames * 100
                print(f"    进度: {pct:.0f}% ({frame_idx}/{max_frames})")

        print(f"    写入完成: {frame_idx} 帧, 调用 FFmpeg 编码...")

        output_short = to_short(str(output_path))
        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(fps),
            "-i", f"{tmpdir_short}/f_%06d.png",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            output_short
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"    FFmpeg 错误: {result.stderr[-300:]}")
            raise RuntimeError("FFmpeg concat 失败")

        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"    编码完成: {frame_idx} 帧")

    def _get_lead_center_x(self, cropped_keypoints, frame_idx, frame_w):
        """获取领操人水平中心（归一化 0~1），默认 0.5"""
        pose_data = cropped_keypoints.get(frame_idx)
        if not pose_data or not pose_data[0]:
            return 0.5

        lead_kps = pose_data[0]
        # 肩髋中点 x
        kps = np.array(lead_kps)
        shoulder_cx = (kps[11][0] + kps[12][0]) / 2
        hip_cx = (kps[23][0] + kps[24][0]) / 2
        return (shoulder_cx + hip_cx) / 2

    def _get_lead_center_y(self, cropped_keypoints, frame_idx, frame_h):
        """获取领操人纵向中心（归一化 0~1），默认 0.5"""
        pose_data = cropped_keypoints.get(frame_idx)
        if not pose_data or not pose_data[0]:
            return 0.5

        lead_kps = pose_data[0]
        kps = np.array(lead_kps)
        # 肩髋中点 y
        shoulder_cy = (kps[11][1] + kps[12][1]) / 2
        hip_cy = (kps[23][1] + kps[24][1]) / 2
        return (shoulder_cy + hip_cy) / 2

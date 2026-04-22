"""阶段05: 身体比例调整

对领操人物应用身材塑形：长腿、瘦腰、整体瘦身等。
基于 cv2.remap 坐标缩放映射，帧间平滑防抖。
"""

import cv2
import numpy as np
from pathlib import Path

from lib.warp import create_displacement_map, apply_warp


class BodyWarpStage:
    def run(self, ctx):
        # 增量跳过：输出已存在则跳过
        if ctx.get("warped_path") and Path(ctx.get("warped_path")).exists():
            print("    已存在，跳过")
            return

        # 当 h2v_convert 被禁用时，fallback 到 stabilized_path 或原始视频
        h2v_path = (ctx.get("ken_burns_path") or
                    ctx.get("warped_path") or
                    ctx.get("h2v_path") or
                    ctx.get("stabilized_path") or
                    str(ctx.input_path))
        cropped_keypoints = ctx.get("cropped_keypoints")
        h2v_size = ctx.get("h2v_size")
        video_info = ctx.get("video_info")

        if not h2v_path or not Path(h2v_path).exists():
            print("    跳过: 无可处理的视频")
            return

        # 如果 h2v 被跳过（cropped_keypoints 为 None），检查是否需要变形
        if not cropped_keypoints:
            warp_cfg = ctx.config.get("body_warp", {})
            needs_warp = any(v != 1.0 for k, v in warp_cfg.items()
                           if k in ("leg_lengthen", "waist_slim", "overall_slim",
                                    "leg_slim", "head_ratio", "chest_enlarge",
                                    "neck_lengthen"))
            if not needs_warp:
                # 没有变形需求，传递原视频路径
                ctx.set("warped_path", h2v_path)
                print("    跳过: 无变形参数（h2v已禁用）")
                return
            else:
                print("    跳过: 无裁剪后关键点")
                return

        crop_w, crop_h = h2v_size
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])
        warp_cfg = ctx.config.get("body_warp", {})

        # 检查是否有实际需要变形的参数
        needs_warp = any(v != 1.0 for k, v in warp_cfg.items()
                         if k in ("leg_lengthen", "waist_slim", "overall_slim",
                                  "leg_slim", "head_ratio", "chest_enlarge",
                                  "neck_lengthen"))
        if not needs_warp:
            print("    跳过: 无变形参数")
            ctx.set("warped_path", h2v_path)
            return

        print(f"    参数: leg={warp_cfg.get('leg_lengthen', 1.0):.2f}, "
              f"waist={warp_cfg.get('waist_slim', 1.0):.2f}, "
              f"slim={warp_cfg.get('overall_slim', 1.0):.2f}, "
              f"chest={warp_cfg.get('chest_enlarge', 1.0):.2f}")

        # 读取 h2v 视频
        cap = cv2.VideoCapture(h2v_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {h2v_path}")

        # 输出: 用 PNG+FFmpeg 替代 cv2.VideoWriter（避免中文路径 codec 问题）
        out_path = ctx.output_dir / f"{Path(h2v_path).stem}_warped.mp4"
        import tempfile, subprocess, shutil, ctypes, os
        tmp_fd, tmp_path_tmp = tempfile.mkstemp(suffix='.mp4')
        os.close(tmp_fd)
        tmp_path_tmp = Path(tmp_path_tmp)
        tmpdir = Path(tempfile.mkdtemp(prefix="bw_"))

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
        prev_map_x = None
        prev_map_y = None
        alpha = 0.8  # 帧间权重: 0.8当前帧 + 0.2前一帧，减少边界跳动

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            pose_data = cropped_keypoints.get(frame_idx)

            if pose_data and pose_data[0]:
                lead_kps = pose_data[0]
                map_x, map_y = create_displacement_map(lead_kps, crop_h, crop_w, warp_cfg)

                # 帧间平滑
                if prev_map_x is not None:
                    map_x = cv2.addWeighted(map_x, alpha, prev_map_x, 1 - alpha, 0)
                    map_y = cv2.addWeighted(map_y, alpha, prev_map_y, 1 - alpha, 0)

                prev_map_x = map_x.copy()
                prev_map_y = map_y.copy()

                frame = apply_warp(frame, map_x, map_y)

                # 双边滤波: 平滑变形接缝，保留边缘细节（默认关闭，极慢）
                if warp_cfg.get("bilateral_filter", False):
                    frame = cv2.bilateralFilter(frame, d=3, sigmaColor=8, sigmaSpace=8)

                # 脸部区域轻锐化：变形后脸部可能变模糊，用 unsharp mask 恢复清晰度
                frame = self._sharpen_face(frame, lead_kps, crop_w, crop_h)

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

        # 移动到最终路径
        try:
            shutil.move(str(tmp_path_tmp), str(out_path))
        except Exception:
            alt_path = ctx.output_dir / f"{Path(h2v_path).stem}_warped_new.mp4"
            shutil.move(str(tmp_path_tmp), str(alt_path))
            out_path = alt_path

        ctx.set("warped_path", str(out_path))
        print(f"    输出: {Path(out_path).name}")

    def _sharpen_face(self, frame, kps, crop_w, crop_h):
        """对脸部区域轻锐化，防止变形后脸部模糊

        MediaPipe face keypoints: 0=鼻子, 1-4=眼部, 5-9=口部/下颌
        扩展一个安全区域覆盖整张脸。
        """
        if not kps:
            return frame

        kps = np.array(kps)
        # 脸部关键点范围 0-9
        face_kps = kps[:10]
        vis = face_kps[:, 2] > 0.3
        if not vis.any():
            return frame

        xs = face_kps[vis, 0] * crop_w
        ys = face_kps[vis, 1] * crop_h
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        # 脸部区域扩展 20%，防止切边
        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)
        x_min = max(0, x_min - pad_x)
        x_max = min(crop_w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(crop_h, y_max + pad_y)

        if x_max - x_min < 10 or y_max - y_min < 10:
            return frame

        face_roi = frame[y_min:y_max, x_min:x_max]
        # Unsharp mask: 轻度锐化（sigma=1, amount=0.8），不破坏肤色
        blurred = cv2.GaussianBlur(face_roi, (0, 0), sigmaX=1)
        sharpened = cv2.addWeighted(face_roi, 1 + 0.8, blurred, -0.8, 0)
        frame[y_min:y_max, x_min:x_max] = sharpened
        return frame

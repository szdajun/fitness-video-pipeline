"""阶段19: 运动强度能量条

在视频右侧显示一个垂直能量条，实时反映当前运动的激烈程度。
基于领操人物的关键点帧间位移计算运动强度。
"""

import cv2
import numpy as np
import json
import subprocess
import shutil
import ctypes
import tempfile
from pathlib import Path


GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
GetShortPathNameW.restype = ctypes.c_uint

def _to_short(path_str):
    buf_size = GetShortPathNameW(str(path_str), None, 0)
    if buf_size == 0:
        return str(path_str)
    buf = ctypes.create_unicode_buffer(buf_size)
    GetShortPathNameW(str(path_str), buf, buf_size)
    return buf.value


class EnergyBarStage:
    def run(self, ctx):
        if ctx.get("energybar_path") and Path(ctx.get("energybar_path")).exists():
            print("    已存在，跳过")
            return

        # 输入视频：优先 ken_burns 输出，其次 h2v
        input_path = (ctx.get("ken_burns_path") or
                     ctx.get("beatflash_path") or
                     ctx.get("color_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     str(ctx.input_path))
        # Windows pathlib bug: Path.exists() 返回 False 但 cv2.VideoCapture 能打开
        if not cv2.VideoCapture(input_path).isOpened():
            print("    跳过: 无输入视频")
            ctx.set("energybar_path", None)
            return

        # 关键点：优先用 cropped_keypoints（竖版对齐），其次原始
        raw_kp = ctx.get("cropped_keypoints")
        if raw_kp:
            keypoints = raw_kp
        else:
            kp_path = ctx.output_dir / f"{ctx.input_path.stem}_cropped_keypoints.json"
            if kp_path.exists():
                with open(kp_path, encoding="utf-8") as f:
                    keypoints = json.load(f)
            else:
                kp_path2 = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
                if not kp_path2.exists():
                    print("    跳过: 无关键点数据")
                    ctx.set("energybar_path", None)
                    return
                with open(kp_path2, encoding="utf-8") as f:
                    raw = json.load(f)
                keypoints = raw.get("keypoints", raw)

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        # 从输入视频读取实际分辨率
        cap_check = cv2.VideoCapture(input_path)
        if not cap_check.isOpened():
            print(f"    跳过: 无法打开视频 {input_path}")
            ctx.set("energybar_path", None)
            return
        orig_w = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_check.release()

        # 领操人追踪
        lead_tid = ctx.get("lead_tid")
        lead_cx = ctx.get("lead_cx")
        if lead_tid is None:
            tracks = self._track_people(keypoints)
            if not tracks:
                print("    跳过: 无法追踪人员")
                ctx.set("energybar_path", None)
                return
            lead_tid = max(tracks, key=lambda tid: tracks[tid]["count"])
            lead_cx = np.median(tracks[lead_tid]["cx_list"])
            ctx.set("lead_tid", lead_tid)
            ctx.set("lead_cx", lead_cx)

        bar_cfg = ctx.config.get("energy_bar", {})
        bar_width = bar_cfg.get("width", 16)
        bar_margin_right = bar_cfg.get("margin_right", 20)
        bar_margin_bottom = bar_cfg.get("margin_bottom", 60)
        bar_height = bar_cfg.get("height", 300)
        smoothing = bar_cfg.get("smoothing", 0.85)
        min_fill = bar_cfg.get("min_fill_ratio", 0.15)
        motion_scale = bar_cfg.get("motion_scale", 200)

        bar_x = orig_w - bar_margin_right - bar_width
        bar_bottom = orig_h - bar_margin_bottom
        bar_top = bar_bottom - bar_height

        print(f"    能量条: {orig_w}x{orig_h}, bar=({bar_x},{bar_top})-({bar_x+bar_width},{bar_bottom}), lead_tid={lead_tid}")

        # PNG + FFmpeg concat（避免 cv2.VideoWriter mp4v 截断问题）
        ffmpeg_bin = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        tmpdir = Path(tempfile.mkdtemp(prefix="eb_"))
        tmpdir_short = _to_short(str(tmpdir))

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0
        motion_history = []
        max_motion = 10.0
        prev_kps = None
        smoothed_motion = 0.0
        BATCH = 100

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_kps = keypoints.get(frame_idx)
            curr_kps = None
            if frame_kps:
                for person_kps in frame_kps:
                    kps_arr = np.array(person_kps)
                    vis = kps_arr[:, 2] > 0.5
                    if vis.sum() < 6:
                        continue
                    shoulders_cx = (kps_arr[5][0] + kps_arr[6][0]) / 2
                    hips_cx = (kps_arr[11][0] + kps_arr[12][0]) / 2
                    cx = (shoulders_cx + hips_cx) / 2
                    if abs(cx - lead_cx) < 0.15:
                        curr_kps = person_kps
                        break

            motion = 0.0
            if curr_kps and prev_kps:
                curr_arr = np.array(curr_kps)
                prev_arr = np.array(prev_kps)
                vis = (prev_arr[:, 2] > 0.3) & (curr_arr[:, 2] > 0.3)
                if vis.sum() >= 4:
                    dx = curr_arr[vis, 0] - prev_arr[vis, 0]
                    dy = curr_arr[vis, 1] - prev_arr[vis, 1]
                    motion = float(np.mean(np.sqrt(dx * dx + dy * dy))) * motion_scale

            smoothed_motion = smoothing * smoothed_motion + (1 - smoothing) * motion
            motion_history.append(smoothed_motion)
            if len(motion_history) > 300:
                motion_history.pop(0)
            max_motion = max(np.percentile(motion_history, 95) if len(motion_history) > 10 else 10.0,
                             max_motion * 0.95, 3.0)

            raw_ratio = min(smoothed_motion / max_motion, 1.0)
            fill_ratio = max(raw_ratio, min_fill)

            # 颜色：绿→黄→红
            if fill_ratio < 0.5:
                color = (0, 255, 0)
            elif fill_ratio < 0.8:
                t = (fill_ratio - 0.5) / 0.3
                color = (0, 255, int(255 * (1 - t)))
            else:
                t = (fill_ratio - 0.8) / 0.2
                color = (int(t * 255), int(255 * (1 - t)), 255)

            fill_h = max(4, int(bar_height * fill_ratio))
            y1 = bar_bottom - fill_h
            y2 = bar_bottom

            cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_width, bar_bottom), (35, 35, 48), -1)
            cv2.rectangle(frame, (bar_x, y1), (bar_x + bar_width, y2), color, -1)
            cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_width, bar_bottom), (200, 200, 215), 1)
            for tick in range(1, 4):
                tick_y = bar_top + int(bar_height * tick / 4)
                cv2.line(frame, (bar_x - 3, tick_y), (bar_x, tick_y), (160, 160, 175), 1)

            fname = f"{tmpdir_short}/f_{frame_idx:06d}.png"
            cv2.imwrite(fname, frame)
            prev_kps = curr_kps
            frame_idx += 1

            if frame_idx % 500 == 0:
                print(f"    进度: {frame_idx}/{max_frames}")

        cap.release()
        print(f"    写入完成: {frame_idx} 帧, 调用 FFmpeg 编码...")

        temp_out = ctx.output_dir / f"{ctx.input_path.stem}_energybar.mp4"
        output_short = _to_short(str(temp_out))
        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(fps),
            "-i", f"{tmpdir_short}/f_%06d.png",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", output_short
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"    FFmpeg 错误: {result.stderr[-300:]}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            ctx.set("energybar_path", None)
            return

        shutil.rmtree(tmpdir, ignore_errors=True)
        ctx.set("energybar_path", str(temp_out))
        print(f"    输出: {temp_out.name} ({frame_idx} 帧)")

    def _track_people(self, keypoints):
        """简单追踪人员"""
        tracks = {}
        for fi, frame_data in sorted(keypoints.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
            if not frame_data:
                continue
            for person_kps in frame_data:
                kps = np.array(person_kps)
                vis = kps[:, 2] > 0.5
                if vis.sum() < 6:
                    cx = 0.5
                else:
                    shoulders_cx = (kps[5][0] + kps[6][0]) / 2
                    hips_cx = (kps[11][0] + kps[12][0]) / 2
                    cx = (shoulders_cx + hips_cx) / 2

                best_tid, best_dist = None, float('inf')
                for tid, trk in tracks.items():
                    prev_cx = np.median(trk["cx_list"]) if trk["cx_list"] else cx
                    dist = abs(cx - prev_cx)
                    if dist < best_dist:
                        best_dist, best_tid = dist, tid

                if best_tid is not None and best_dist < 0.2:
                    tracks[best_tid]["cx_list"].append(cx)
                    tracks[best_tid]["count"] += 1
                else:
                    new_tid = len(tracks)
                    tracks[new_tid] = {"cx_list": [cx], "count": 1}
        return tracks

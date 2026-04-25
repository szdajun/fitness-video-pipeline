"""阶段15: 运动热力图

追踪领操人的运动轨迹，在画面上叠加彩色热力图显示运动强度区域。
运动越多、幅度越大的地方颜色越亮（黄/红色），静止区域偏蓝/绿。
热力图随时间累积，清洗节奏感。
"""

import cv2
import numpy as np
import json
from pathlib import Path

from lib.utils import path_exists, create_writer


class MotionHeatmapStage:
    def run(self, ctx):
        # 增量跳过
        if ctx.get("heatmap_path") and path_exists(ctx.get("heatmap_path")):
            print("    已存在，跳过")
            return

        # 找输入视频
        input_path = (ctx.get("faceblur_path") or
                     ctx.get("ghost_path") or
                     ctx.get("leadbox_path") or
                     ctx.get("count_path") or
                     ctx.get("skeleton_path") or
                     ctx.get("color_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     ctx.get("stabilized_path") or
                     str(ctx.input_path))
        if not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("heatmap_path", None)
            return

        # 加载关键点
        keypoints_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        if not keypoints_path.exists():
            print("    跳过: 无关键点数据")
            ctx.set("heatmap_path", None)
            return

        with open(keypoints_path, encoding="utf-8") as f:
            raw = json.load(f)
        keypoints = raw.get("keypoints", raw)

        # 复用领操人追踪结果
        lead_tid = ctx.get("lead_tid")
        lead_cx = ctx.get("lead_cx")
        if lead_tid is None:
            tracks = self._track_people(keypoints)
            if not tracks:
                print("    跳过: 无法追踪人员")
                ctx.set("heatmap_path", None)
                return
            lead_tid = max(tracks, key=lambda tid: tracks[tid]["count"])
            lead_cx = np.median(tracks[lead_tid]["cx_list"])
            ctx.set("lead_tid", lead_tid)
            ctx.set("lead_cx", lead_cx)

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        orig_w = video_info["width"]
        orig_h = video_info["height"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        print(f"    领操人: tid={lead_tid}")
        print(f"    热力图: {orig_w}x{orig_h}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_heatmap.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        # 热力图参数
        heatmap = np.zeros((orig_h, orig_w), dtype=np.float32)
        decay = 0.97          # 每帧热力图衰减
        motion_gain = 120.0  # 运动放大系数

        # 上一帧领操人位置（归一化坐标）
        prev_lead_center = None

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 读取当前帧关键点
            frame_kps = keypoints.get(str(frame_idx))
            lead_center = None

            if frame_kps:
                for pi, person_kps in enumerate(frame_kps):
                    kps_arr = np.array(person_kps)
                    vis = kps_arr[:, 2] > 0.5
                    if vis.sum() < 6:
                        continue

                    shoulders_cx = (kps_arr[5][0] + kps_arr[6][0]) / 2
                    hips_cx = (kps_arr[11][0] + kps_arr[12][0]) / 2
                    cx = (shoulders_cx + hips_cx) / 2

                    if abs(cx - lead_cx) < 0.15:
                        vis_kps = kps_arr[vis]
                        lead_center = (vis_kps[:, 0].mean(), vis_kps[:, 1].mean())
                        break

            # 如果有前一帧数据，计算运动量并叠加
            if prev_lead_center is not None and lead_center is not None:
                dx = lead_center[0] - prev_lead_center[0]
                dy = lead_center[1] - prev_lead_center[1]
                motion = np.sqrt(dx * dx + dy * dy)

                if motion > 0.005:  # 过滤微小抖动
                    cx_px = int(lead_center[0] * orig_w)
                    cy_px = int(lead_center[1] * orig_h)
                    sigma = max(20, int(motion * 400))

                    # 用 numpy slice + Gaussian权重快速叠加（避免逐像素循环）
                    x1 = max(0, cx_px - sigma * 2)
                    x2 = min(orig_w, cx_px + sigma * 2)
                    y1 = max(0, cy_px - sigma * 2)
                    y2 = min(orig_h, cy_px + sigma * 2)

                    if x2 > x1 and y2 > y1:
                        ys = np.arange(y1, y2)
                        xs = np.arange(x1, x2)
                        yy, xx = np.meshgrid(ys, xs, indexing='ij')
                        dist2 = ((xx - cx_px) ** 2 + (yy - cy_px) ** 2).astype(np.float32)
                        gaussian = np.exp(-dist2 / (2 * sigma ** 2)) * motion_gain
                        heatmap[y1:y2, x1:x2] += gaussian

            # 衰减
            heatmap *= decay
            heatmap = np.clip(heatmap, 0, 255)

            # 绘制热力图叠加
            if heatmap.max() > 5:
                h_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                h_smooth = cv2.medianBlur(h_norm, 9)
                h_color = cv2.applyColorMap(h_smooth, cv2.COLORMAP_JET)
                cv2.addWeighted(frame, 0.75, h_color, 0.25, 0, frame)

            writer.write(frame)
            prev_lead_center = lead_center
            frame_idx += 1

            if frame_idx % 300 == 0:
                pct = frame_idx / max_frames * 100
                print(f"    进度: {pct:.0f}% ({frame_idx}/{max_frames})")

        cap.release()
        writer.release()

        ctx.set("heatmap_path", str(temp_path))
        print(f"    输出: {temp_path.name}")

    def _track_people(self, keypoints):
        """简单追踪人员"""
        tracks = {}
        for fi, frame_data in sorted(keypoints.items()):
            fi = int(fi)
            if not frame_data:
                continue
            for pi, person_kps in enumerate(frame_data):
                kps = np.array(person_kps)
                vis = kps[:, 2] > 0.5
                if vis.sum() < 6:
                    cx = 0.5
                else:
                    shoulders_cx = (kps[5][0] + kps[6][0]) / 2
                    hips_cx = (kps[11][0] + kps[12][0]) / 2
                    cx = (shoulders_cx + hips_cx) / 2

                best_tid = None
                best_dist = float('inf')
                for tid, trk in tracks.items():
                    prev_cx = np.median(trk["cx_list"]) if trk["cx_list"] else cx
                    dist = abs(cx - prev_cx)
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None and best_dist < 0.2:
                    tracks[best_tid]["cx_list"].append(cx)
                    tracks[best_tid]["count"] += 1
                else:
                    new_tid = len(tracks)
                    tracks[new_tid] = {"cx_list": [cx], "count": 1}

        return tracks

"""阶段13: 领操人残影

在当前帧上叠加领操人过去数帧的位置残影，产生运动轨迹拖尾效果。
残影仅显示领操人，不显示其他人，增强领操人的动作延续感。
"""

import cv2
import numpy as np
import json
from collections import deque
from pathlib import Path

from lib.utils import path_exists, create_writer


# COCO 17 骨架连接（用于绘制残影轮廓）
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


class LeadGhostStage:
    def run(self, ctx):
        # 增量跳过
        if ctx.get("ghost_path") and path_exists(ctx.get("ghost_path")):
            print("    已存在，跳过")
            return

        # 找输入视频
        input_path = (ctx.get("leadbox_path") or
                     ctx.get("count_path") or
                     ctx.get("skeleton_path") or
                     ctx.get("color_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     ctx.get("stabilized_path") or
                     str(ctx.input_path))
        if not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("ghost_path", None)
            return

        # 加载关键点
        keypoints_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        if not keypoints_path.exists():
            print("    跳过: 无关键点数据")
            ctx.set("ghost_path", None)
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
                ctx.set("ghost_path", None)
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

        # 残影参数
        ghost_frames = 6      # 保留多少帧的残影
        ghost_alpha = 0.35   # 残影透明度
        decay_factor = 0.75  # 越旧的残影越淡

        print(f"    领操人: tid={lead_tid}")
        print(f"    残影: {orig_w}x{orig_h}, {ghost_frames}帧, alpha={ghost_alpha}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_ghost.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        # 帧缓冲队列：存储过去 N 帧的关键点
        # 每个元素: (frame_idx, {lead_kps or None})
        ghost_buffer = deque(maxlen=ghost_frames)
        ghost_buffer_append = ghost_buffer.append

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 获取当前帧领操人关键点
            frame_kps = keypoints.get(str(frame_idx))
            lead_kps_current = None
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
                        lead_kps_current = kps_arr
                        break

            # 加入缓冲
            ghost_buffer_append((frame_idx, lead_kps_current))

            # 绘制所有残影（从旧到新，越新的在最上层）
            for gi, (fi, kps) in enumerate(ghost_buffer):
                if kps is None:
                    continue
                age = len(ghost_buffer) - 1 - gi  # 0=最新, N-1=最旧
                if age == 0:
                    continue  # 当前帧在原帧显示，不重复绘制

                # 透明度随年龄递减
                alpha = ghost_alpha * (decay_factor ** age)
                color = (0, int(200 * (decay_factor ** age)), 0)  # 绿色系，越旧越暗

                self._draw_skeleton(frame, kps, orig_w, orig_h, color, int(1 * decay_factor ** age))

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        ctx.set("ghost_path", str(temp_path))
        print(f"    输出: {temp_path.name}")

    def _draw_skeleton(self, frame, kps_arr, w, h, color, thickness):
        """在帧上绘制骨架"""
        vis = kps_arr[:, 2] > 0.3
        # 绘制连接线
        for (a, b) in SKELETON_CONNECTIONS:
            if vis[a] and vis[b]:
                pt_a = (int(kps_arr[a][0] * w), int(kps_arr[a][1] * h))
                pt_b = (int(kps_arr[b][0] * w), int(kps_arr[b][1] * h))
                cv2.line(frame, pt_a, pt_b, color, max(1, thickness), cv2.LINE_AA)
        # 绘制关键点
        for i in range(len(kps_arr)):
            if vis[i]:
                pt = (int(kps_arr[i][0] * w), int(kps_arr[i][1] * h))
                cv2.circle(frame, pt, max(2, 3 + thickness), color, -1, cv2.LINE_AA)

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

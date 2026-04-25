"""阶段10: Pose骨架叠加

在视频上叠加显示检测到的人体骨架。
使用领操人骨架 + 其他人骨架区分颜色。
"""

import cv2
import numpy as np
import json
from pathlib import Path

from lib.utils import path_exists, create_writer


# COCO 17 keypoint connections
SKELETON_CONNECTIONS = [
    # Head/face
    (0, 1), (0, 2), (1, 3), (2, 4),  # face outline
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),  # shoulders + hips
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]

# Keypoint names for reference
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


class SkeletonOverlayStage:
    def run(self, ctx):
        # 增量跳过：输出已存在则跳过
        if ctx.get("skeleton_path") and path_exists(ctx.get("skeleton_path")):
            print("    已存在，跳过")
            return

        # 找输入视频
        input_path = (ctx.get("color_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     ctx.get("stabilized_path") or
                     str(ctx.input_path))
        if not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("skeleton_path", None)
            return

        # 加载关键点
        keypoints_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        if not keypoints_path.exists():
            print("    跳过: 无关键点数据")
            ctx.set("skeleton_path", None)
            return

        with open(keypoints_path, encoding="utf-8") as f:
            raw = json.load(f)

        # JSON 结构: {"keypoints": {...}, "video_info": {...}}
        keypoints = raw.get("keypoints", raw)

        # 追踪领操人
        tracks = self._track_people(keypoints)
        if not tracks:
            print("    跳过: 无法追踪人员")
            ctx.set("skeleton_path", None)
            return

        lead_tid = max(tracks, key=lambda tid: tracks[tid]["count"])
        lead_cx = np.median(tracks[lead_tid]["cx_list"])

        # 共享追踪结果（后续阶段复用，避免跳变）
        ctx.set("lead_tid", lead_tid)
        ctx.set("lead_cx", lead_cx)

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        orig_w = video_info["width"]
        orig_h = video_info["height"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        print(f"    领操人: tid={lead_tid}, x_center={lead_cx:.3f}")
        print(f"    叠加骨架: {orig_w}x{orig_h}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_skeleton.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        # 判断是否为竖屏（9:16）
        is_portrait = orig_h > orig_w

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 绘制骨架
            frame_kps = keypoints.get(str(frame_idx))
            if frame_kps:
                for pi, person_kps in enumerate(frame_kps):
                    # 判断是否领操人（位置最接近lead_cx）
                    kps_arr = np.array(person_kps)
                    vis = kps_arr[:, 2] > 0.5
                    if vis.sum() < 6:
                        continue

                    # 计算此人的中心x
                    shoulders_cx = (kps_arr[5][0] + kps_arr[6][0]) / 2
                    hips_cx = (kps_arr[11][0] + kps_arr[12][0]) / 2
                    cx = (shoulders_cx + hips_cx) / 2

                    is_lead = abs(cx - lead_cx) < 0.15

                    # 颜色: 领操人=绿色，其他人=浅蓝色
                    if is_lead:
                        color = (0, 255, 0)  # 绿色
                        thickness = 2
                    else:
                        color = (255, 200, 100)  # 浅蓝
                        thickness = 1

                    # 画连接线
                    for (a, b) in SKELETON_CONNECTIONS:
                        if vis[a] and vis[b]:
                            pt_a = (int(kps_arr[a][0] * orig_w), int(kps_arr[a][1] * orig_h))
                            pt_b = (int(kps_arr[b][0] * orig_w), int(kps_arr[b][1] * orig_h))
                            cv2.line(frame, pt_a, pt_b, color, thickness)

                    # 画关键点
                    for i in range(len(kps_arr)):
                        if vis[i]:
                            pt = (int(kps_arr[i][0] * orig_w), int(kps_arr[i][1] * orig_h))
                            cv2.circle(frame, pt, 3, color, -1)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        ctx.set("skeleton_path", str(temp_path))
        print(f"    输出: {temp_path.name}")

    def _track_people(self, keypoints):
        """追踪人员（领操人版）

        匹配策略：综合考虑 x 位置和人体大小。
        领操人在前排，人体框更大（y 方向跨度更大）。
        使用匈牙利算法风格的加权评分，每次同一帧内匹配一次。
        """
        # 第一遍：收集所有帧的人体特征
        frame_features = []
        for fi, frame_data in sorted(keypoints.items()):
            fi = int(fi)
            if not frame_data:
                frame_features.append((fi, []))
                continue
            persons = []
            for pi, person_kps in enumerate(frame_data):
                kps = np.array(person_kps)
                vis = kps[:, 2] > 0.5
                if vis.sum() < 4:
                    cx, cy, body_h = 0.5, 0.5, 0.0
                else:
                    shoulders_cx = (kps[5][0] + kps[6][0]) / 2
                    hips_cx = (kps[11][0] + kps[12][0]) / 2
                    cx = (shoulders_cx + hips_cx) / 2
                    # 身体高度（归一化）：可见关键点中 y 的范围
                    vis_y = kps[vis, 1]
                    body_h = vis_y.max() - vis_y.min()
                    cy = (vis_y.max() + vis_y.min()) / 2
                persons.append({"cx": cx, "cy": cy, "body_h": body_h, "pi": pi})
            frame_features.append((fi, persons))

        # 第二遍：按帧顺序追踪
        tracks = {}  # tid -> {"cx_list": [], "body_h_list": [], "count": int}

        for fi, persons in frame_features:
            if not persons:
                continue

            # 计算当前帧每个人与已有轨迹的匹配分数
            matched = set()
            for pi, pers in enumerate(persons):
                best_tid = None
                best_score = float('inf')

                for tid, trk in tracks.items():
                    prev_cx = np.median(trk["cx_list"][-10:]) if trk["cx_list"] else pers["cx"]
                    # 位置距离（归一化）
                    dist_cx = abs(pers["cx"] - prev_cx)
                    # 体型：领操人通常更大（比历史均值略大或相当）
                    avg_h = np.median(trk["body_h_list"][-10:]) if trk["body_h_list"] else pers["body_h"]
                    size_diff = abs(pers["body_h"] - avg_h) / max(avg_h, 0.05)

                    # 综合分数：位置为主，体型为辅
                    # 领操人在前排（cy 偏下/中间，body_h 偏大）
                    score = dist_cx * 1.0 + size_diff * 0.3

                    if score < best_score:
                        best_score = score
                        best_tid = tid

                if best_tid is not None and best_score < 0.25:
                    tracks[best_tid]["cx_list"].append(pers["cx"])
                    tracks[best_tid]["body_h_list"].append(pers["body_h"])
                    tracks[best_tid]["count"] += 1
                    matched.add(pi)

            # 未匹配的人创建新轨迹
            for pi, pers in enumerate(persons):
                if pi not in matched:
                    new_tid = len(tracks)
                    tracks[new_tid] = {
                        "cx_list": [pers["cx"]],
                        "body_h_list": [pers["body_h"]],
                        "count": 1
                    }

        return tracks

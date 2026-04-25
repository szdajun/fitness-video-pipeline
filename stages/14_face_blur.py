"""阶段14: 背景人脸模糊

对画面中非领操人的人物应用高斯模糊，保护隐私。
领操人保持清晰，背景中的跟操人员（尤其是面部区域）做模糊处理。
使用关键点定位人脸大致位置（头顶区域）作为模糊中心。
"""

import cv2
import numpy as np
import json
from pathlib import Path

from lib.utils import path_exists, create_writer


# 人体关键点索引（MediaPipe / COCO 通用）
# 0=鼻子, 1-4=眼部/耳部, 5-6=肩膀
# 头顶估计位置在 nose 和 shoulders 之间偏上
HEAD_KEYPOINTS = [0, 1, 2, 3, 4]  # 鼻子+眼部+耳部


class FaceBlurStage:
    def run(self, ctx):
        # 增量跳过
        if ctx.get("faceblur_path") and path_exists(ctx.get("faceblur_path")):
            print("    已存在，跳过")
            return

        # 找输入视频
        input_path = (ctx.get("ghost_path") or
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
            ctx.set("faceblur_path", None)
            return

        # 加载关键点
        keypoints_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        if not keypoints_path.exists():
            print("    跳过: 无关键点数据")
            ctx.set("faceblur_path", None)
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
                ctx.set("faceblur_path", None)
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

        # 模糊参数
        blur_radius = 25  # 高斯模糊半径（越大越模糊）
        head_radius = 40  # 头部区域半径（像素）

        print(f"    领操人: tid={lead_tid}")
        print(f"    背景模糊: {orig_w}x{orig_h}, blur={blur_radius}px, head_radius={head_radius}px")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_faceblur.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 获取当前帧关键点
            frame_kps = keypoints.get(str(frame_idx))

            # 全帧模糊（先模糊再恢复领操人）
            blurred = cv2.GaussianBlur(frame, (blur_radius, blur_radius), 0)

            if frame_kps:
                # 创建领操人恢复 mask
                lead_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

                for pi, person_kps in enumerate(frame_kps):
                    kps_arr = np.array(person_kps)
                    vis = kps_arr[:, 2] > 0.5
                    if vis.sum() < 6:
                        continue

                    # 判断是否为领操人
                    shoulders_cx = (kps_arr[5][0] + kps_arr[6][0]) / 2
                    hips_cx = (kps_arr[11][0] + kps_arr[12][0]) / 2
                    cx = (shoulders_cx + hips_cx) / 2
                    is_lead = abs(cx - lead_cx) < 0.15

                    if is_lead:
                        # 领操人：恢复整个人体区域
                        vis_kps = kps_arr[vis]
                        x_min = vis_kps[:, 0].min()
                        x_max = vis_kps[:, 0].max()
                        y_min = vis_kps[:, 1].min()
                        y_max = vis_kps[:, 1].max()

                        x1 = max(0, int(x_min * orig_w) - 15)
                        y1 = max(0, int(y_min * orig_h) - 15)
                        x2 = min(orig_w, int(x_max * orig_w) + 15)
                        y2 = min(orig_h, int(y_max * orig_h) + 15)

                        cv2.rectangle(lead_mask, (x1, y1), (x2, y2), 255, -1)

                        # 额外恢复面部区域（更精准）
                        for kp_idx in HEAD_KEYPOINTS:
                            if kps_arr[kp_idx][2] > 0.3:
                                fx = int(kps_arr[kp_idx][0] * orig_w)
                                fy = int(kps_arr[kp_idx][1] * orig_h)
                                cv2.circle(lead_mask, (fx, fy), head_radius, 255, -1)
                    else:
                        # 背景人：仅模糊面部区域
                        for kp_idx in HEAD_KEYPOINTS:
                            if kps_arr[kp_idx][2] > 0.3:
                                fx = int(kps_arr[kp_idx][0] * orig_w)
                                fy = int(kps_arr[kp_idx][1] * orig_h)
                                # 在模糊图上覆盖原图的面部区域
                                y_start = max(0, fy - head_radius)
                                y_end = min(orig_h, fy + head_radius)
                                x_start = max(0, fx - head_radius)
                                x_end = min(orig_w, fx + head_radius)
                                if y_end > y_start and x_end > x_start:
                                    blurred[y_start:y_end, x_start:x_end] = \
                                        frame[y_start:y_end, x_start:x_end]

                # 将领操人区域从模糊图恢复为原图
                lead_area = cv2.bitwise_and(frame, frame, mask=lead_mask)
                blurred_masked = cv2.bitwise_and(blurred, blurred, mask=~lead_mask)
                frame = cv2.add(blurred_masked, lead_area)
            else:
                # 无关键点：全模糊
                frame = blurred

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 300 == 0:
                pct = frame_idx / max_frames * 100
                print(f"    进度: {pct:.0f}% ({frame_idx}/{max_frames})")

        cap.release()
        writer.release()

        ctx.set("faceblur_path", str(temp_path))
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

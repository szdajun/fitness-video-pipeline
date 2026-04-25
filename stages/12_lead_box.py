"""阶段12: 领操人高亮框

给领操人物绘制一个醒目的高亮边框（矩形框），方便观众快速识别领操人。
支持多人体场景，在不同人之间切换时平滑过渡。
"""

import cv2
import numpy as np
import json
from pathlib import Path

from lib.utils import path_exists, create_writer


class LeadBoxStage:
    def run(self, ctx):
        # 增量跳过
        if ctx.get("leadbox_path") and path_exists(ctx.get("leadbox_path")):
            print("    已存在，跳过")
            return

        # 找输入视频
        input_path = (ctx.get("count_path") or
                     ctx.get("skeleton_path") or
                     ctx.get("color_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     ctx.get("stabilized_path") or
                     str(ctx.input_path))
        if not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("leadbox_path", None)
            return

        # 加载关键点
        keypoints_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        if not keypoints_path.exists():
            print("    跳过: 无关键点数据")
            ctx.set("leadbox_path", None)
            return

        with open(keypoints_path, encoding="utf-8") as f:
            raw = json.load(f)
        keypoints = raw.get("keypoints", raw)

        # 复用骨架阶段的领操人追踪结果
        lead_tid = ctx.get("lead_tid")
        lead_cx = ctx.get("lead_cx")
        if lead_tid is None:
            tracks = self._track_people(keypoints)
            if not tracks:
                print("    跳过: 无法追踪人员")
                ctx.set("leadbox_path", None)
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

        print(f"    领操人: tid={lead_tid}, x_center={lead_cx:.3f}")
        print(f"    高亮框: {orig_w}x{orig_h}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_leadbox.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        # 平滑：用 EMA 平滑头顶坐标，避免抖动
        alpha = 0.85  # 平滑系数
        prev_head = None

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 获取当前帧关键点
            frame_kps = keypoints.get(str(frame_idx))
            lead_head = None

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
                        nose = kps_arr[0]
                        if nose[2] > 0.3:
                            head_x, head_y = nose[0], nose[1]
                        else:
                            head_x = (kps_arr[1][0] + kps_arr[2][0]) / 2
                            head_y = min(kps_arr[1][1], kps_arr[2][1])

                        hx = int(head_x * orig_w)
                        hy = int(head_y * orig_h)
                        lead_head = (hx, hy)
                        break

            # EMA 平滑 + 画皇冠（即使短暂丢失也保持上一帧位置）
            if lead_head or prev_head:
                if lead_head and prev_head:
                    hx = int(alpha * lead_head[0] + (1 - alpha) * prev_head[0])
                    hy = int(alpha * lead_head[1] + (1 - alpha) * prev_head[1])
                elif lead_head:
                    hx, hy = lead_head
                else:
                    hx, hy = prev_head

                prev_head = (hx, hy)
                self._draw_crown(frame, hx, hy)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        ctx.set("leadbox_path", str(temp_path))
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

    def _draw_crown(self, frame, hx, hy):
        """在头顶位置画皇冠

        皇冠大小根据与画面宽的比例自动调整。
        """
        # 皇冠尺寸：画面宽的 3.5%
        crown_w = max(30, int(frame.shape[1] * 0.035))
        crown_h = int(crown_w * 0.65)

        # 皇冠颜色：金色
        gold = (0, 180, 255)
        dark_gold = (0, 100, 200)
        white = (255, 255, 255)

        # 皇冠左下角（戴在头上，y 在头顶上方一点）
        base_x = hx - crown_w // 2
        base_y = hy - crown_h - 8  # 稍微高出头顶

        # 皇冠三个尖的 x 坐标
        t1_x = base_x + int(crown_w * 0.15)
        t2_x = base_x + int(crown_w * 0.5)
        t3_x = base_x + int(crown_w * 0.85)
        t_y = base_y

        # 皇冠主体（凹陷的底边）
        #   /‾\   ← 三个尖
        #  /___\
        pts = np.array([
            [base_x, base_y + crown_h],        # 左下
            [t1_x, t_y],                        # 左尖
            [base_x + int(crown_w * 0.3), base_y + int(crown_h * 0.5)],  # 左谷
            [t2_x, t_y],                        # 中尖（最高）
            [base_x + int(crown_w * 0.7), base_y + int(crown_h * 0.5)],  # 右谷
            [t3_x, t_y],                        # 右尖
            [base_x + crown_w, base_y + crown_h],  # 右下
        ], dtype=np.int32)

        # 填充皇冠主体
        cv2.fillPoly(frame, [pts], gold, lineType=cv2.LINE_AA)
        # 描边
        cv2.polylines(frame, [pts], isClosed=True, color=dark_gold, thickness=2, lineType=cv2.LINE_AA)

        # 皇冠底部横线
        cv2.line(frame, (base_x, base_y + crown_h),
                (base_x + crown_w, base_y + crown_h), dark_gold, 2, cv2.LINE_AA)

        # 每个尖顶画圆珠（白点）
        for tx in [t1_x, t2_x, t3_x]:
            cv2.circle(frame, (tx, t_y + 2), max(2, crown_w // 14), white, -1, cv2.LINE_AA)

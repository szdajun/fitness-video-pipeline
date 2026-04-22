"""阶段11: 人数动态显示

在画面角落实时显示当前检测到的人数。
使用领操人追踪数据，标注"领操人+跟操人数"。
"""

import cv2
import numpy as np
import json
from pathlib import Path

from lib.utils import create_writer, draw_chinese_text


class PersonCountStage:
    def run(self, ctx):
        # 增量跳过
        if ctx.get("count_path") and Path(ctx.get("count_path")).exists():
            print("    已存在，跳过")
            return

        # 找输入视频
        input_path = (ctx.get("leadbox_path") or
                     ctx.get("skeleton_path") or
                     ctx.get("color_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     ctx.get("stabilized_path") or
                     str(ctx.input_path))
        if not Path(input_path).exists():
            print("    跳过: 无输入视频")
            ctx.set("count_path", None)
            return

        # 加载关键点
        keypoints_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        if not keypoints_path.exists():
            print("    跳过: 无关键点数据")
            ctx.set("count_path", None)
            return

        with open(keypoints_path, encoding="utf-8") as f:
            raw = json.load(f)
        keypoints = raw.get("keypoints", raw)

        # 复用骨架阶段的领操人追踪结果，避免跳变
        lead_tid = ctx.get("lead_tid")
        lead_cx = ctx.get("lead_cx")
        if lead_tid is None:
            tracks = self._track_people(keypoints)
            if not tracks:
                print("    跳过: 无法追踪人员")
                ctx.set("count_path", None)
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
        print(f"    人数显示: {orig_w}x{orig_h}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_count.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        # 预计算每帧人数（用于平滑）
        frame_counts = {}
        for fi_str, frame_data in sorted(keypoints.items(), key=lambda x: int(x[0])):
            fi = int(fi_str)
            if frame_data:
                frame_counts[fi] = len(frame_data)
            else:
                frame_counts[fi] = 0

        # 显示位置
        alpha = 0.55  # 透明度

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 计算人数（当前帧 + 平滑）
            count = frame_counts.get(frame_idx, 0)
            # 平滑：取前后各3帧的均值
            neighbors = [frame_counts.get(frame_idx + d, count) for d in range(-3, 4)]
            avg_count = int(round(sum(neighbors) / len(neighbors)))

            # 人数 = 跟操人数（总人数 - 1）
            follower_count = max(0, avg_count - 1)
            total_count = avg_count

            # 构建文字
            if follower_count > 0:
                text = f"领操+跟操: {total_count}人"
                sub_text = f"跟操 {follower_count}人"
            else:
                text = f"领操: 1人"
                sub_text = None

            # 文字渲染（使用 PIL 支持中文）
            try:
                from PIL import Image, ImageDraw, ImageFont
                box_padding = 12
                font_path = None
                for fp in ["C:/Windows/Fonts/msyh.ttc",
                           "C:/Windows/Fonts/simhei.ttf",
                           "C:/Windows/Fonts/simsun.ttc"]:
                    if Path(fp).exists():
                        font_path = fp
                        break

                if font_path:
                    pil_font = ImageFont.truetype(font_path, 22, encoding="utf-8")
                    pil_font_small = ImageFont.truetype(font_path, 16, encoding="utf-8")

                    # 计算文字尺寸
                    bbox = pil_font.getbbox(text)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]

                    x0 = orig_w - tw - box_padding * 2 - 10
                    y0 = box_padding * 2 + th + 6

                    if sub_text:
                        bbox2 = pil_font_small.getbbox(sub_text)
                        tw2 = bbox2[2] - bbox2[0]
                        th2 = bbox2[3] - bbox2[1]
                        box_h = th + th2 + 16
                    else:
                        box_h = th + 10

                    # 转为 PIL Image
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)

                    # 半透明黑色背景
                    draw.rectangle([x0, 0, x0 + tw + box_padding * 2, y0 + box_h - 4],
                                 fill=(0, 0, 0, int(255 * alpha)))

                    # 主文字
                    draw.text((x0 + box_padding, y0 - th - 2), text,
                             font=pil_font, fill=(255, 255, 255))

                    # 副文字
                    if sub_text:
                        draw.text((x0 + box_padding, y0 + 4), sub_text,
                                 font=pil_font_small, fill=(180, 180, 180))

                    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    raise ImportError("no font")
            except Exception:
                # 回退：英文渲染（数字不会乱码）
                cv2.putText(frame, text, (10, 30),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        ctx.set("count_path", str(temp_path))
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

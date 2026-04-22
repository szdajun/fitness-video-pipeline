"""阶段16: 同步率计分

比较每个跟操人员与领操人的动作姿态相似度，实时显示"同步率"百分比。
相似度基于关键点位置归一化距离：跟操人越接近领操人姿态，分数越高。
用于评判跟操质量，增加互动性和趣味性。
"""

import cv2
import numpy as np
import json
from pathlib import Path

from lib.utils import create_writer


class SyncScoreStage:
    def run(self, ctx):
        # 增量跳过
        if ctx.get("sync_path") and Path(ctx.get("sync_path")).exists():
            print("    已存在，跳过")
            return

        # 找输入视频
        input_path = (ctx.get("heatmap_path") or
                     ctx.get("faceblur_path") or
                     ctx.get("ghost_path") or
                     ctx.get("leadbox_path") or
                     ctx.get("count_path") or
                     ctx.get("skeleton_path") or
                     ctx.get("color_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     ctx.get("stabilized_path") or
                     str(ctx.input_path))
        if not Path(input_path).exists():
            print("    跳过: 无输入视频")
            ctx.set("sync_path", None)
            return

        # 加载关键点
        keypoints_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        if not keypoints_path.exists():
            print("    跳过: 无关键点数据")
            ctx.set("sync_path", None)
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
                ctx.set("sync_path", None)
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

        # 同步率参数
        sync_threshold = 0.12   # 归一化距离阈值以下认为"同步"
        similarity_alpha = 0.7   # EMA 平滑系数

        print(f"    领操人: tid={lead_tid}")
        print(f"    同步率: {orig_w}x{orig_h}, threshold={sync_threshold}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_sync.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        # 关键点索引（COCO 17，去掉面部容易差异的点）
        BODY_KPS = list(range(5, 17))  # 肩膀到脚踝

        prev_sync_rate = 0.0
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_kps = keypoints.get(str(frame_idx))

            if frame_kps:
                lead_kps = None
                others_kps = []

                for pi, person_kps in enumerate(frame_kps):
                    kps_arr = np.array(person_kps)
                    vis = kps_arr[:, 2] > 0.5
                    if vis.sum() < 6:
                        continue

                    shoulders_cx = (kps_arr[5][0] + kps_arr[6][0]) / 2
                    hips_cx = (kps_arr[11][0] + kps_arr[12][0]) / 2
                    cx = (shoulders_cx + hips_cx) / 2

                    if abs(cx - lead_cx) < 0.15:
                        lead_kps = kps_arr
                    else:
                        others_kps.append(kps_arr)

                sync_count = 0
                total_count = len(others_kps)

                if lead_kps is not None:
                    for other_kps in others_kps:
                        dist = self._pose_distance(lead_kps, other_kps, BODY_KPS)
                        if dist < sync_threshold:
                            sync_count += 1

                if total_count > 0:
                    raw_sync_rate = sync_count / total_count * 100
                else:
                    raw_sync_rate = 100.0

                # EMA 平滑
                sync_rate = similarity_alpha * prev_sync_rate + (1 - similarity_alpha) * raw_sync_rate
                prev_sync_rate = sync_rate
            else:
                sync_rate = prev_sync_rate * 0.9  # 无数据时缓慢下降

            # 绘制同步率标签
            sync_int = int(round(sync_rate))

            # 颜色：绿色(>70%)→黄色(40-70%)→红色(<40%)
            if sync_int >= 70:
                color = (0, 255, 100)
            elif sync_int >= 40:
                color = (0, 200, 255)
            else:
                color = (0, 80, 255)

            label = f"同步率 {sync_int}%"

            # 副标题
            sub = f"同步 {sync_count}/{total_count} 人" if (frame_kps and total_count > 0) else None

            # PIL 中文渲染
            try:
                from PIL import Image, ImageDraw, ImageFont
                font_path = None
                for fp in ["C:/Windows/Fonts/msyh.ttc",
                           "C:/Windows/Fonts/simhei.ttf",
                           "C:/Windows/Fonts/simsun.ttc"]:
                    if Path(fp).exists():
                        font_path = fp
                        break

                if font_path:
                    pil_font = ImageFont.truetype(font_path, 28, encoding="utf-8")
                    pil_font_small = ImageFont.truetype(font_path, 18, encoding="utf-8")

                    bbox = pil_font.getbbox(label)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]

                    pad = 10
                    bx = 15
                    by = orig_h - th - pad * 3 - 10
                    box_h = th + 10
                    if sub:
                        bbox2 = pil_font_small.getbbox(sub)
                        box_h += (bbox2[3] - bbox2[1]) + 8

                    # 转为 PIL
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)

                    # 半透明背景
                    draw.rectangle([bx - pad, by - pad, bx + tw + pad * 2, by + box_h + pad],
                                 fill=(0, 0, 0, 150))

                    # 主文字
                    draw.text((bx, by), label, font=pil_font,
                             fill=(color[2], color[1], color[0]))

                    # 副文字
                    if sub:
                        bbox2 = pil_font_small.getbbox(sub)
                        th2 = bbox2[3] - bbox2[1]
                        draw.text((bx, by + th + 6), sub, font=pil_font_small,
                                 fill=(180, 180, 180))

                    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    raise ImportError("no font")
            except Exception:
                # 回退
                cv2.putText(frame, label, (15, orig_h - 40),
                           cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        ctx.set("sync_path", str(temp_path))
        print(f"    输出: {temp_path.name}")

    def _pose_distance(self, kps_a, kps_b, kp_indices):
        """计算两个人姿态关键点之间的归一化均方根距离"""
        dists = []
        for ki in kp_indices:
            if kps_a[ki][2] > 0.3 and kps_b[ki][2] > 0.3:
                dx = kps_a[ki][0] - kps_b[ki][0]
                dy = kps_a[ki][1] - kps_b[ki][1]
                dists.append(np.sqrt(dx * dx + dy * dy))

        if not dists:
            return 1.0

        return np.mean(dists)

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

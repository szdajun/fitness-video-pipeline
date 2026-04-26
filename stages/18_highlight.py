"""阶段18: 亮点片段提取 (改进版)

自动从长视频中提取最精彩的片段，生成一个精华预告片。
多维评分：
  - 运动强度（关键点速度）
  - 节拍密度（音频节拍）
  - 姿态完整度（可见关键点数量）
  - 居中性（人物在画面中心程度）
输出一个 15-30 秒的精华合集。
"""

import cv2
import numpy as np
import json
from pathlib import Path

from lib.utils import path_exists, create_writer


class HighlightStage:
    def run(self, ctx):
        # full_video 模式下跳过精华片段选取
        if ctx.config.get("full_video", False):
            print("    跳过: full_video 模式")
            ctx.set("highlight_path", None)
            return

        if ctx.get("highlight_path") and path_exists(ctx.get("highlight_path")):
            print("    已存在，跳过")
            return

        # 找输入视频（优先用 beat_flash 输出，否则用 color 输出）
        input_path = (ctx.get("beatflash_path") or
                     ctx.get("color_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     str(ctx.input_path))
        # Windows pathlib bug: Path.exists() 返回 False 但 cv2.VideoCapture 能打开
        if not cv2.VideoCapture(input_path).isOpened():
            print("    跳过: 无输入视频")
            ctx.set("highlight_path", None)
            return

        # 加载关键点
        keypoints_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        if not keypoints_path.exists():
            print("    跳过: 无关键点数据")
            ctx.set("highlight_path", None)
            return

        with open(keypoints_path, encoding="utf-8") as f:
            raw = json.load(f)
        keypoints = raw.get("keypoints", raw)

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        total_frames = video_info["frames"]
        max_frames = video_info.get("process_frames", total_frames)

        # 从输入视频读取实际分辨率
        cap_res = cv2.VideoCapture(input_path)
        orig_w = int(cap_res.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_res.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_res.release()

        # 节拍帧（优先用 beat_flash 结果，否则自己检测）
        beat_frames = ctx.get("beat_frames")
        if not beat_frames:
            audio_path = ctx.get("audio_path")
            if not audio_path or not path_exists(audio_path):
                extracted_audio = ctx.output_dir / f"{ctx.input_path.stem}_audio_temp.wav"
                ffmpeg = "C:/Users/18091/ffmpeg/ffmpeg.exe"
                import subprocess
                subprocess.run([ffmpeg, "-y", "-i", str(ctx.input_path),
                              "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
                              str(extracted_audio)], capture_output=True, errors="replace")
                if extracted_audio.exists():
                    audio_path = str(extracted_audio)

            if audio_path:
                beat_frames = self._detect_beats(audio_path, fps, max_frames)
            if not beat_frames:
                beat_frames = []

        beat_set = set(beat_frames) if beat_frames else set()

        # 计算每帧的多维评分
        segment_seconds = 5       # 每段 5 秒
        segment_frames = int(segment_seconds * fps)
        num_segments = max_frames // segment_frames

        # 领操人中心位置（用于居中性评分）
        lead_cx = ctx.get("lead_cx", 0.5)  # 默认中心

        segment_scores = []
        prev_kps = None

        for si in range(num_segments):
            start_f = si * segment_frames
            end_f = min(start_f + segment_frames, max_frames)

            motion_sum = 0
            motion_peak = 0
            beat_count = 0
            frame_count = 0
            pose_complete_sum = 0  # 姿态完整度
            center_score_sum = 0   # 居中性

            for fi in range(start_f, end_f):
                frame_kps = keypoints.get(str(fi))
                if frame_kps:
                    beat_count += 1 if fi in beat_set else 0

                    # 取第一人作为分析对象
                    person_kps = frame_kps[0]
                    kps_arr = np.array(person_kps)

                    # 姿态完整度：可见关键点数量 (COCO 17点)
                    vis_count = (kps_arr[:, 2] > 0.3).sum()
                    pose_complete_sum += vis_count / 17.0

                    # 居中性：肩膀中心 x 距离 0.5 的偏差
                    if len(kps_arr) >= 12:
                        shoulder_cx = (kps_arr[5][0] + kps_arr[6][0]) / 2
                        center_dev = abs(shoulder_cx - 0.5)  # 0=中心, 0.5=边缘
                        center_score_sum += 1.0 - min(center_dev * 2, 1.0)  # 归一化到 0-1

                    if prev_kps:
                        # 计算与前一帧的运动差异
                        for pi, prev_person_kps in enumerate(prev_kps):
                            if pi >= len(frame_kps):
                                continue
                            prev_arr = np.array(prev_person_kps)
                            curr_arr = np.array(frame_kps[pi])
                            vis = (prev_arr[:, 2] > 0.3) & (curr_arr[:, 2] > 0.3)
                            if vis.sum() >= 6:
                                dx = curr_arr[vis, 0] - prev_arr[vis, 0]
                                dy = curr_arr[vis, 1] - prev_arr[vis, 1]
                                motion = np.mean(np.sqrt(dx*dx + dy*dy))
                                motion_sum += motion
                                motion_peak = max(motion_peak, motion)
                    prev_kps = frame_kps
                frame_count += 1

            # 多维综合分数
            avg_motion = motion_sum / max(frame_count, 1)
            beat_density = beat_count / max(segment_frames, 1)
            pose_complete = pose_complete_sum / max(frame_count, 1)
            center_score = center_score_sum / max(frame_count, 1)

            # 权重配置
            w_motion = 50
            w_beat = 150
            w_pose = 30
            w_center = 20

            # 峰值运动加成（峰值高说明动作有力）
            motion_bonus = motion_peak * 30

            # 居中性惩罚：太靠边的片段降分
            center_penalty = max(0, (0.6 - center_score) * 50) if center_score < 0.6 else 0

            score = (
                avg_motion * w_motion +
                beat_density * w_beat +
                pose_complete * w_pose +
                center_score * w_center +
                motion_bonus -
                center_penalty
            )

            segment_scores.append({
                'idx': si,
                'score': score,
                'start_f': start_f,
                'end_f': end_f,
                'motion': avg_motion,
                'beat_density': beat_density,
                'pose_complete': pose_complete,
                'center_score': center_score,
            })

        if not segment_scores:
            print("    跳过: 无法计算片段分数")
            ctx.set("highlight_path", None)
            return

        # 多样性约束：贪心选取，避免重复相似片段
        MIN_GAP_SECONDS = 3  # 最小间隔3秒
        min_gap_frames = int(MIN_GAP_SECONDS * fps)

        target_duration = 30  # 目标总时长（秒）
        max_segments = max(2, min(num_segments, int(target_duration / segment_seconds)))

        sorted_scores = sorted(segment_scores, key=lambda x: x['score'], reverse=True)
        selected = []

        for seg in sorted_scores:
            if len(selected) >= max_segments:
                break

            # 检查与已选片段的时间重叠/过近
            too_close = False
            for sel in selected:
                gap = abs(seg['start_f'] - sel['end_f']) if seg['start_f'] >= sel['end_f'] else abs(sel['start_f'] - seg['end_f'])
                if gap < min_gap_frames:
                    too_close = True
                    break

            if not too_close:
                selected.append(seg)

        selected.sort(key=lambda x: x['start_f'])  # 按时间排序，保证连贯性

        highlight_sec = len(selected) * segment_seconds
        print(f"    亮点片段: 选取 {len(selected)} 段，共 {highlight_sec:.0f} 秒")
        if selected:
            print(f"    评分: motion={max(s['motion'] for s in selected):.3f}, "
                  f"beat={max(s['beat_density'] for s in selected):.2f}, "
                  f"pose={max(s['pose_complete'] for s in selected):.2f}, "
                  f"center={max(s['center_score'] for s in selected):.2f}")

        # 读取视频并截取亮点段
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {ctx.input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_highlight.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        for seg in selected:
            start_f = seg['start_f']
            end_f = seg['end_f']
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            for fi in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret:
                    break
                # 左上角画一个黄色星形图标
                cx, cy = 32, 22
                r = 14
                pts = []
                for k in range(10):
                    angle = k * 36 - 90
                    rad = angle * np.pi / 180
                    radius = r if k % 2 == 0 else r * 0.4
                    pts.append((cx + radius * np.cos(rad), cy + radius * np.sin(rad)))
                pts = np.array(pts, dtype=np.int32)
                cv2.fillPoly(frame, [pts], (0, 200, 255), lineType=cv2.LINE_AA)
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 100), thickness=1, lineType=cv2.LINE_AA)
                writer.write(frame)

        cap.release()
        writer.release()

        ctx.set("highlight_path", str(temp_path))
        ctx.set("highlight_duration", highlight_sec)
        print(f"    输出: {temp_path.name} ({highlight_sec:.0f}s)")

    def _detect_beats(self, audio_path: str, fps: float, max_frames: int):
        """使用 librosa 检测音频节拍"""
        try:
            import librosa
        except ImportError:
            return None
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='frames')
        except Exception:
            return None
        result = []
        for f in beats:
            fn = int(round(f))
            if 0 <= fn < max_frames:
                result.append(fn)
        result = sorted(set(result))
        if len(result) > max_frames * 0.5:
            result = result[::2]
        return result
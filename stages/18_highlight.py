"""阶段18: 亮点片段提取

自动从长视频中提取最精彩的片段，生成一个精华预告片。
评分因素：运动幅度（关键点速度）+ 节拍强度。
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

        # 计算每帧的运动强度（关键点位移）
        segment_seconds = 5       # 每段 5 秒
        segment_frames = int(segment_seconds * fps)
        num_segments = max_frames // segment_frames

        segment_scores = []
        prev_kps = None

        for si in range(num_segments):
            start_f = si * segment_frames
            end_f = min(start_f + segment_frames, max_frames)

            motion_sum = 0
            beat_count = 0
            frame_count = 0

            for fi in range(start_f, end_f):
                frame_kps = keypoints.get(str(fi))
                if frame_kps:
                    beat_count += 1 if fi in beat_set else 0
                    if prev_kps:
                        # 计算与前一帧的运动差异
                        for pi, person_kps in enumerate(frame_kps):
                            if pi >= len(prev_kps):
                                continue
                            prev_arr = np.array(prev_kps[pi])
                            curr_arr = np.array(person_kps)
                            vis = (prev_arr[:, 2] > 0.3) & (curr_arr[:, 2] > 0.3)
                            if vis.sum() >= 6:
                                dx = curr_arr[vis, 0] - prev_arr[vis, 0]
                                dy = curr_arr[vis, 1] - prev_arr[vis, 1]
                                motion_sum += np.mean(np.sqrt(dx*dx + dy*dy))
                    prev_kps = frame_kps
                frame_count += 1

            # 综合分数：运动强度 + 节拍密度
            avg_motion = motion_sum / max(frame_count, 1)
            beat_density = beat_count / max(segment_frames, 1)
            score = avg_motion * 50 + beat_density * 200
            segment_scores.append((si, score, start_f, end_f))

        if not segment_scores:
            print("    跳过: 无法计算片段分数")
            ctx.set("highlight_path", None)
            return

        # 选取分数最高的片段
        target_duration = 30  # 目标总时长（秒）
        target_segments = max(2, min(num_segments, int(target_duration / segment_seconds)))
        top_segments = sorted(segment_scores, key=lambda x: x[1], reverse=True)[:target_segments]
        top_segments.sort(key=lambda x: x[2])  # 按时间排序，保证连贯性

        # 计算输出时长
        total_highlight_frames = sum(s[1] for s in top_segments) if top_segments else 0
        highlight_sec = len(top_segments) * segment_seconds
        print(f"    亮点片段: 选取 {len(top_segments)} 段，共 {highlight_sec:.0f} 秒")

        # 读取视频并截取亮点段
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_highlight.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        for si, score, start_f, end_f in top_segments:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            for fi in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret:
                    break
                # 左上角画一个黄色星形图标，不用文字
                cx, cy = 32, 22
                r = 14
                # 五角星顶点
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

    def _compute_motion(self, kps_a, kps_b):
        """计算两个关键点帧之间的平均运动量"""
        if kps_a is None or kps_b is None:
            return 0.0
        a = np.array(kps_a)
        b = np.array(kps_b)
        vis = (a[:, 2] > 0.3) & (b[:, 2] > 0.3)
        if vis.sum() < 4:
            return 0.0
        dx = b[vis, 0] - a[vis, 0]
        dy = b[vis, 1] - a[vis, 1]
        return float(np.mean(np.sqrt(dx*dx + dy*dy)))

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

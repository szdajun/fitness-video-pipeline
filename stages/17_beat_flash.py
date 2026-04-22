"""阶段17: 节拍同步闪烁

在音乐重拍瞬间叠加白色闪烁或微缩放效果，增强视频节奏感和能量感。
使用 librosa 分析音频节奏，自动检测节拍时刻。
"""

import cv2
import numpy as np
import json
from pathlib import Path

from lib.utils import create_writer


class BeatFlashStage:
    def run(self, ctx):
        # 增量跳过
        if ctx.get("beatflash_path") and Path(ctx.get("beatflash_path")).exists():
            print("    已存在，跳过")
            return

        # 找输入视频
        input_path = (ctx.get("ken_burns_path") or
                     ctx.get("color_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     ctx.get("stabilized_path") or
                     str(ctx.input_path))
        if not Path(input_path).exists():
            print("    跳过: 无输入视频")
            ctx.set("beatflash_path", None)
            return

        # 找音频文件
        audio_path = ctx.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            # 从输入视频提取音频
            extracted_audio = ctx.output_dir / f"{ctx.input_path.stem}_audio_temp.wav"
            ffmpeg = "C:/Users/18091/ffmpeg/ffmpeg.exe"
            import subprocess
            result = subprocess.run(
                [ffmpeg, "-y", "-i", str(ctx.input_path),
                 "-vn", "-acodec", "pcm_s16le",
                 "-ar", "22050", "-ac", "1",
                 str(extracted_audio)],
                capture_output=True, errors="replace"
            )
            if extracted_audio.exists():
                audio_path = str(extracted_audio)
            else:
                print(f"    跳过: 无法提取音频")
                ctx.set("beatflash_path", None)
                return

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        # 从输入视频读取实际分辨率（处理后可能已变）
        cap_check = cv2.VideoCapture(input_path)
        if not cap_check.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")
        orig_w = int(cap_check.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_check.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_check.release()

        # 检测节拍
        beat_frames = self._detect_beats(str(audio_path), fps, max_frames)
        if beat_frames is None:
            print("    跳过: 节拍检测失败（需要 librosa）")
            ctx.set("beatflash_path", None)
            return

        print(f"    检测到 {len(beat_frames)} 个节拍")
        print(f"    闪烁: {orig_w}x{orig_h}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")

        temp_path = ctx.output_dir / f"{ctx.input_path.stem}_beatflash.mp4"
        writer = create_writer(str(temp_path), fps, orig_w, orig_h)

        # 闪烁参数
        flash_duration = 6   # 闪烁持续帧数（延长让过渡更平滑）
        flash_alpha = 0.40  # 亮度提升系数
        zoom_factor = 1.025  # 节拍时放大倍数（缩小一点，更平滑）
        zoom_smoothing = 0.75  # 缩放平滑系数（越大越平滑）
        beat_set = set(beat_frames)  # 快速查找

        frame_idx = 0
        target_zoom = 1.0
        current_zoom = 1.0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 判断是否为节拍帧或节拍后几帧
            flash_strength = 0
            for f in range(frame_idx - flash_duration + 1, frame_idx + 1):
                if f in beat_set:
                    age = frame_idx - f
                    flash_strength = max(flash_strength, flash_alpha * (1.0 - age / flash_duration))
                    break

            # 目标缩放值：节拍时为 zoom_factor，否则为 1.0
            target_zoom = zoom_factor if flash_strength > 0 else 1.0

            # 平滑过渡到目标缩放值
            current_zoom = zoom_smoothing * current_zoom + (1 - zoom_smoothing) * target_zoom

            # 节拍时提亮画面
            if flash_strength > 0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + flash_strength), 0, 255).astype(np.uint8)
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 平滑缩放
            if current_zoom > 1.005:
                zoom_w = int(orig_w * current_zoom)
                zoom_h = int(orig_h * current_zoom)
                zoomed = cv2.resize(frame, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
                crop_x = (zoom_w - orig_w) // 2
                crop_y = (zoom_h - orig_h) // 2
                frame = zoomed[crop_y:crop_y + orig_h, crop_x:crop_x + orig_w]

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        ctx.set("beatflash_path", str(temp_path))
        ctx.set("beat_frames", beat_frames)
        print(f"    输出: {temp_path.name}")

    def _detect_beats(self, audio_path: str, fps: float, max_frames: int):
        """使用 librosa 检测音频节拍，返回节拍对应的视频帧号列表"""
        try:
            import librosa
        except ImportError:
            return None

        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=None, mono=True)
        except Exception as e:
            print(f"    librosa 加载音频失败: {e}")
            return None

        # 节奏检测
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='frames')
        except Exception:
            return None

        beat_frames = []
        for f in beats:
            frame_num = int(round(f))
            if 0 <= frame_num < max_frames:
                beat_frames.append(frame_num)

        # 去重排序
        beat_frames = sorted(set(beat_frames))

        # 每隔一个节拍过滤（健身音乐通常 120BPM，每秒2拍太多）
        # 保留强拍（每2个节拍取1个）
        if len(beat_frames) > max_frames * 0.5:
            beat_frames = beat_frames[::2]

        return beat_frames

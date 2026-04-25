"""阶段20: 片头片尾自动生成

片头：从处理后的视频中截取运动强度最高的3-5秒片段，
      叠加频道名称 + 标题文字渐入动画。

片尾：截取原视频最后3-5秒，叠加关注引导 + 音频淡出。

位置：在 energy_bar 之后、export 之前运行。
"""

import cv2
from lib.utils import path_exists
import numpy as np
import subprocess
import shutil
import re
import ctypes
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _get_short_path(p):
    buf_size = ctypes.windll.kernel32.GetShortPathNameW(str(p), None, 0)
    if buf_size == 0:
        return str(p)
    buf = ctypes.create_unicode_buffer(buf_size)
    ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, buf_size)
    return buf.value


def _write_video(frames, output_path, fps):
    """将帧列表通过 PNG+FFmpeg 写入视频（解决 create_writer 中文路径问题）"""
    # Use output directory for temp PNGs to avoid short-path issues
    out_dir = Path(output_path).parent
    tmpdir = out_dir / f"_intro_tmp_{Path(output_path).stem}"
    tmpdir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(tmpdir / f"f_{i:06d}.png"), frame)

    tmp_out_path = out_dir / f"_intro_out_{Path(output_path).stem}.mp4"

    ffmpeg_bin = Path("C:/Users/18091/ffmpeg/ffmpeg.exe")
    if ffmpeg_bin.exists():
        ffmpeg_bin = str(ffmpeg_bin)
    else:
        ffmpeg_bin = shutil.which("ffmpeg") or str(ffmpeg_bin)
    input_pattern = str(tmpdir / "f_%06d.png").replace("\\", "/")
    output_str = str(tmp_out_path).replace("\\", "/")
    cmd = [ffmpeg_bin, "-y", "-v", "warning",
           "-framerate", str(fps),
           "-i", input_pattern,
           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-an",
           output_str]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"    FFmpeg 失败 (return={r.returncode})")
        print(f"    前3帧mean: {[float(frames[i].mean()) for i in range(min(3, len(frames)))]}")
        print(f"    FFmpeg stderr: {r.stderr[-500:]}")
        # Save frames to output dir for debugging
        for i in range(min(3, len(frames))):
            cv2.imwrite(str(out_dir / f"_debug_frame_{i:06d}.png"), frames[i])
        # DON'T clean up tmpdir on failure - leave for debugging
        print(f"    Temp PNGs left in: {tmpdir}")
        raise RuntimeError(f"FFmpeg error: {r.stderr[-300:]}")

    shutil.move(str(tmp_out_path), str(output_path))
    shutil.rmtree(tmpdir, ignore_errors=True)
    # Clean up any leftover temp files
    for f in out_dir.glob(f"_intro_tmp_*"):
        shutil.rmtree(f, ignore_errors=True)
    for f in out_dir.glob("_intro_out_*"):
        f.unlink(missing_ok=True)


# 尝试加载支持中文的字体
def _get_font(size):
    """加载支持中文的字体"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
        "C:/Windows/Fonts/arial.ttf",       # fallback
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


class IntroOutroStage:
    def run(self, ctx):
        stages_cfg = ctx.config.get("stages", {})
        if not stages_cfg.get("intro_outro", False):
            print("    跳过: intro_outro 未启用")
            return

        # 片头和片尾都用主内容视频（不用能量条视频，能量条底部有黑背景）
        video_path = (ctx.get("color_path") or
                     ctx.get("ken_burns_path") or
                     ctx.get("warped_path") or
                     ctx.get("h2v_path") or
                     str(ctx.input_path))
        outro_video_path = video_path
        if not video_path or not cv2.VideoCapture(video_path).isOpened():
            print("    跳过: 无处理后的视频")
            return

        cfg = ctx.config.get("intro_outro", {})
        intro_duration = cfg.get("intro_duration", 4.0)
        outro_duration = cfg.get("outro_duration", 5.0)
        channel_name = cfg.get("channel_name", "胭脂虎健身团")
        cta_text = cfg.get("cta_text", "关注不迷路")
        audio_fade_out = cfg.get("audio_fade_out", 3.0)

        video_info = ctx.get("video_info")
        fps = video_info["fps"]

        stem = Path(ctx.input_path).stem
        # 提取领操人名字（从 ctx 或从文件名）
        lead_name = ctx.get("lead_name")
        if not lead_name:
            lead_name = re.sub(r'[\d_]+.*$', '', stem)
        location = cfg.get("location", "西安时代广场")
        date_str = cfg.get("date") or "2026-04-20"

        print(f"    片头片尾生成: {channel_name} | 带操人:{lead_name} | {location}/{date_str}")
        print(f"    频道: {channel_name}, 片头:{intro_duration}s, 片尾:{outro_duration}s, 音频淡出:{audio_fade_out}s")

        intro_path = self._create_intro(
            video_path, ctx.output_dir / f"{stem}_intro.mp4",
            lead_name, channel_name, location, date_str, intro_duration, fps, cfg
        )

        outro_path = self._create_outro(
            outro_video_path, ctx.output_dir / f"{stem}_outro.mp4",
            outro_duration, audio_fade_out, fps, cfg
        )

        ctx.set("intro_path", str(intro_path))
        ctx.set("outro_path", str(outro_path))
        print(f"    输出: 片头={intro_path.name}, 片尾={outro_path.name}")

    def _extract_title(self, stem: str) -> str:
        stem = re.sub(r'^[\d]+', '', stem)
        stem = re.sub(r'[_\-\s]', '', stem)
        return stem[:6] if stem else stem

    def _create_intro(self, video_path: str, output_path: Path,
                       lead_name: str, channel: str, location: str,
                       date_str: str, duration: float,
                       fps: float, cfg: dict) -> Path:
        """生成片头：运动强度最高的片段 + 中文文字动画"""
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_time = total_frames / actual_fps
        cap.release()

        if total_time <= duration:
            # 视频比片头短，直接用 FFmpeg 复制全部
            ffmpeg_bin = Path("C:/Users/18091/ffmpeg/ffmpeg.exe")
            if ffmpeg_bin.exists():
                ffmpeg_bin = str(ffmpeg_bin)
            else:
                ffmpeg_bin = shutil.which("ffmpeg") or str(ffmpeg_bin)
            out_short = _get_short_path(str(output_path))
            inp_short = _get_short_path(str(video_path))
            cmd = [ffmpeg_bin, "-y", "-i", inp_short,
                   "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                   "-pix_fmt", "yuv420p", "-an", out_short]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                raise RuntimeError(f"FFmpeg copy failed: {r.stderr[-200:]}")
            return output_path

        # 找运动强度最高的片段
        cap = cv2.VideoCapture(video_path)
        frame_scores = []
        prev_gray = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            else:
                diff = 0
            frame_scores.append(diff)
            prev_gray = gray
        cap.release()

        frame_scores = np.array(frame_scores)
        window = int(duration * actual_fps)
        best_score, best_start = -1, 0
        for i in range(len(frame_scores) - window):
            score = np.mean(frame_scores[i:i + window])
            if score > best_score:
                best_score = score
                best_start = i

        # 渲染片头：先收集所有帧
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_start)

        fade_in_frames = int(0.5 * actual_fps)
        frames_to_write = []

        while len(frames_to_write) < window:
            ret, frame = cap.read()
            if not ret:
                break

            # 叠加中文文字（用 PIL）
            frame = self._draw_intro_text_pil(frame, lead_name, channel, location, date_str, len(frames_to_write) / window)

            # 帧淡入（淡入整个合成画面，包括文字）
            # 避免从完全黑色开始（会导致x264编码失败）
            if len(frames_to_write) < fade_in_frames:
                alpha = max(0.1, len(frames_to_write) / fade_in_frames)  # 最小0.1，避免完全黑帧
                overlay = np.zeros_like(frame)
                frame = (frame * alpha + overlay * (1 - alpha)).astype(np.uint8)

            frames_to_write.append(frame)

        cap.release()
        print(f"    片头渲染完成: {len(frames_to_write)} 帧")
        _write_video(frames_to_write, str(output_path), actual_fps)
        return output_path

    def _draw_intro_text_pil(self, frame, lead_name: str, channel: str, location: str, date_str: str, progress: float):
        """用 PIL 绘制片头中文文字 - 3行格式:
        第1行(顶部): channel 频道名
        第2行(中部): 带操人：领操人名字
        第3行(底部): 地点/日期"""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size

        # 底部不透明黑底：从 y=70% 开始覆盖到画面底部
        overlay_top = int(h * 0.70)
        draw.rectangle([(0, overlay_top), (w, h)], fill=(0, 0, 0))

        # 第1行：频道名称（黑色区域中间，较大白色）
        font_lg = _get_font(int(h * 0.065))
        bbox = draw.textbbox((0, 0), channel, font=font_lg)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = (w - tw) // 2
        cy = int(overlay_top * 0.18)  # 上方黑色区域上部
        draw.text((cx, cy), channel, font=font_lg, fill=(255, 255, 255))

        # 第2行：带操人：xxx（视频区域中部，中等黄色）
        font_md = _get_font(int(h * 0.08))
        lead_text = f"带操人：{lead_name}"
        bbox = draw.textbbox((0, 0), lead_text, font=font_md)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = (w - tw) // 2
        cy = int(h * 0.40) - th // 2  # 视频区域中部
        draw.text((cx, cy), lead_text, font=font_md, fill=(255, 220, 50))

        # 第3行：地点/日期（底部黑色区域中间，白色）
        font_sm = _get_font(int(h * 0.045))
        date_text = f"{location}/{date_str}"
        bbox = draw.textbbox((0, 0), date_text, font=font_sm)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        cx = (w - tw) // 2
        cy = overlay_top + int((h - overlay_top - th) / 2) - int(h * 0.09)  # 下面黑色区域中间偏上2行
        draw.text((cx, cy), date_text, font=font_sm, fill=(255, 255, 255))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _create_outro(self, video_path: str, output_path: Path,
                       duration: float,
                       audio_fade_out: float, fps: float, cfg: dict) -> Path:
        """生成片尾：末尾片段 + CTA 文字 + 视频淡出"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        start_frame = max(0, int(total_frames - duration * actual_fps))
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames_to_write = []
        max_frames = int(duration * actual_fps)
        fade_out_frames = int(0.5 * actual_fps)  # 最后0.5秒淡出

        while len(frames_to_write) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 视频淡出：最后 fade_out_frames 帧渐变到纯黑
            frame_idx = len(frames_to_write)
            if frame_idx >= max_frames - fade_out_frames:
                alpha = 1.0 - (frame_idx - (max_frames - fade_out_frames)) / fade_out_frames
                alpha = max(0.0, alpha)
                overlay = np.full_like(frame, 0)
                frame = (frame * alpha + overlay * (1 - alpha)).astype(np.uint8)

            # 叠加 CTA 文字
            frame = self._draw_outro_text_pil(frame)
            frames_to_write.append(frame)

        cap.release()

        # 单独保存音频淡出信息到文件（export 阶段读取）
        audio_fade_file = output_path.with_suffix('.fade')
        with open(audio_fade_file, 'w') as f:
            f.write(str(audio_fade_out))

        print(f"    片尾渲染完成: {len(frames_to_write)} 帧")
        _write_video(frames_to_write, str(output_path), actual_fps)
        return output_path

    def _draw_outro_text_pil(self, frame, progress: float = 0.0):
        """用 PIL 绘制片尾 CTA 中文文字 - 直接叠加在视频上
        4行：打工牛马 / 健身达人 / 关注不迷路 / 点击关注"""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size

        # 4行文字配置
        lines = [
            ("打工牛马", int(h * 0.065), (255, 220, 50)),
            ("健身达人", int(h * 0.065), (255, 220, 50)),
            ("关注不迷路", int(h * 0.060), (255, 255, 255)),
            ("点击关注", int(h * 0.050), (200, 200, 200)),
        ]

        # 计算行间距和总高度，居中于视频下半部分
        line_spacing = int(h * 0.035)
        total_height = sum(lines[i][1] for i in range(len(lines))) + line_spacing * (len(lines) - 1)
        y = int(h * 0.36)  # 视频中间偏上

        for text, font_size, color in lines:
            font = _get_font(font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            cx = (w - tw) // 2
            draw.text((cx, y), text, font=font, fill=color)
            y += th + line_spacing

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

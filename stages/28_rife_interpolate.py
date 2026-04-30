"""阶段28: RIFE 帧插值 (30→60fps)

将 30fps 视频插值到 60fps，运动更丝滑。
需要 PyTorch，默认关闭。
"""

import os, cv2, subprocess, shutil, tempfile, ctypes
from pathlib import Path
from lib.utils import path_exists
from lib.rife_interpolate import RIFEInterpolator


class RIFEInterpolateStage:
    def run(self, ctx):
        if ctx.get("rife_path") and path_exists(ctx.get("rife_path")):
            print("    已存在，跳过")
            return

        cfg = ctx.config.get("rife", {})
        if not cfg.get("enabled", False):
            print("    跳过: RIFE 未启用")
            ctx.set("rife_path", self._input_path(ctx))
            return

        input_path = self._input_path(ctx)
        if not input_path or not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("rife_path", None)
            return

        video_info = ctx.get("video_info")
        src_fps = video_info["fps"]
        target_fps = cfg.get("target_fps", 60)
        max_frames = video_info.get("process_frames", video_info["frames"])

        if target_fps <= src_fps:
            print(f"    跳过: 目标帧率 {target_fps} <= 源帧率 {src_fps}")
            ctx.set("rife_path", input_path)
            return

        interpolator = RIFEInterpolator(
            gpu=cfg.get("gpu", True),
            half=cfg.get("half_precision", True),
        )
        if not interpolator.is_available():
            print("    跳过: PyTorch 未安装")
            ctx.set("rife_path", input_path)
            return

        print(f"    RIFE: {src_fps}→{target_fps}fps, gpu={cfg.get('gpu', True)}")

        tmpdir = Path(tempfile.mkdtemp(prefix="rife_"))
        short_tmp = self._to_short(str(tmpdir))

        total_out = interpolator.interpolate_video(
            input_path, str(tmpdir), src_fps, target_fps, max_frames)
        print(f"    插值完成: {total_out} 帧")

        out_path = ctx.output_dir / f"{Path(input_path).stem}_rife.mp4"
        short_out = self._to_short(str(out_path))
        ffmpeg_bin = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        cmd = [
            ffmpeg_bin, "-y", "-framerate", str(target_fps),
            "-i", f"{short_tmp}/f_%06d.png",
            "-c:v", "libx264", "-preset", "fast", "-crf", "1",
            "-pix_fmt", "yuv444p", "-an", short_out,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"    FFmpeg 错误: {result.stderr[-300:]}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            ctx.set("rife_path", input_path)
            return

        shutil.rmtree(tmpdir, ignore_errors=True)

        if cv2.VideoCapture(str(out_path)).isOpened():
            ctx.set("rife_path", str(out_path))
            print(f"    输出: {out_path.name} ({total_out} 帧, {target_fps}fps)")
        else:
            ctx.set("rife_path", input_path)
            print(f"    警告: RIFE 输出无效，回退输入")

    def _input_path(self, ctx):
        return (ctx.get("face_beautify2_path") or
                ctx.get("face_beautify_path") or
                ctx.get("energybar_path") or
                ctx.get("beatflash_path") or
                ctx.get("highlight_path") or
                ctx.get("color_path") or
                ctx.get("ken_burns_path") or
                str(ctx.input_path))

    def _to_short(self, path_str):
        buf_size = ctypes.windll.kernel32.GetShortPathNameW(str(path_str), None, 0)
        if buf_size == 0:
            return str(path_str)
        buf = ctypes.create_unicode_buffer(buf_size)
        ctypes.windll.kernel32.GetShortPathNameW(str(path_str), buf, buf_size)
        return buf.value

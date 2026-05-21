"""阶段30: 背景音乐

将 BGM 混入视频，节拍闪烁+能量条已提供视觉卡点。
"""

import subprocess, shutil
from pathlib import Path
from lib.utils import path_exists


class BGMBeatStage:
    def run(self, ctx):
        if ctx.get("bgm_path") and path_exists(ctx.get("bgm_path")):
            print("    已存在，跳过")
            return

        cfg = ctx.config.get("bgm_beat", {})
        bgm_file = cfg.get("bgm_file", "")
        if not bgm_file or not Path(bgm_file).exists():
            print("    跳过: 未设置 BGM 文件")
            return

        video_info = ctx.get("video_info")
        fps = video_info.get("fps", 30)
        total_frames = video_info.get("process_frames", video_info["frames"])
        duration = total_frames / fps if fps > 0 else 60

        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        bg_volume = cfg.get("bg_volume", 0.5)

        input_path = (ctx.get("pip_path") or
                     ctx.get("burst_path") or
                     ctx.get("mascot_path") or
                     ctx.get("watermark_path") or
                     ctx.get("energybar_path") or
                     ctx.get("beatflash_path") or
                     str(ctx.input_path))

        mixed_path = ctx.output_dir / f"{Path(input_path).stem}_withbgm.mp4"

        # 检查原视频是否有音轨
        probe = subprocess.run(
            [ffmpeg, "-i", input_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        has_audio = "Audio" in probe.stderr

        print(f"    BGM混入: bg_vol={bg_volume}")

        if has_audio:
            cmd = [
                ffmpeg, "-y",
                "-i", input_path,
                "-i", bgm_file,
                "-filter_complex",
                f"[0:a]volume=1.0[a0];[1:a]volume={bg_volume}[bg];[a0][bg]amix=inputs=2:duration=first:dropout_transition=0.5[a]",
                "-map", "0:v", "-map", "[a]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(duration),
                str(mixed_path)
            ]
        else:
            cmd = [
                ffmpeg, "-y",
                "-i", input_path,
                "-i", bgm_file,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(duration),
                str(mixed_path)
            ]

        r = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)

        if r.returncode != 0:
            print(f"    BGM混入失败: {r.stderr[-200:]}")
            ctx.set("bgm_path", input_path)
            return

        ctx.set("bgm_path", str(mixed_path))
        print(f"    输出: {mixed_path.name}")

"""阶段36: 秦腔冷开场

黑屏 + 逐句歌词 + 秦腔音频，一步生成完整片段，然后拼接正片。
"""

import subprocess, shutil, json, tempfile, ctypes, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lib.utils import path_exists


class QinColdOpenStage:
    def run(self, ctx):
        if ctx.get("coldopen_path") and path_exists(ctx.get("coldopen_path")):
            print("    已存在，跳过")
            return

        cfg = ctx.config.get("qin_cold_open", {})
        if not cfg.get("enabled", False):
            return

        audio_file = cfg.get("audio_file", "")
        if not audio_file or not Path(audio_file).exists():
            print("    跳过: 未设置秦腔开场音频")
            return

        input_path = (ctx.get("bgm_path") or
                     ctx.get("pip_path") or
                     ctx.get("filmlook_path") or
                     ctx.get("mascot_path") or
                     ctx.get("energybar_path") or
                     str(ctx.input_path))
        if not input_path or not path_exists(input_path):
            return

        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        ffprobe = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")

        # 获取秦腔音频时长
        import json as _json
        probe = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "json", audio_file],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        dur = 5.0
        try:
            dur = float(_json.loads(probe.stdout)["format"]["duration"])
        except Exception:
            pass

        # 歌词与时间轴
        lines = ["天～黑～咧～！", "下～班～咧～！", "吃～过～咧～！",
                 "乡党们～！", "锻～炼～咧！！！"]
        n = len(lines)
        line_dur = dur / n

        # 中文字体
        font = None
        for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, 80)
                break
        if not font:
            print("    跳过: 无中文字体")
            ctx.set("coldopen_path", input_path)
            return

        fps = 30
        total_frames = int(dur * fps)
        tmpdir = Path(tempfile.mkdtemp(prefix="qin_"))

        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint

        def to_short(p):
            buf_size = GetShortPathNameW(str(p), None, 0)
            if buf_size == 0:
                return str(p)
            buf = ctypes.create_unicode_buffer(buf_size)
            GetShortPathNameW(str(p), buf, buf_size)
            return buf.value

        tmpdir_short = to_short(str(tmpdir))

        print(f"    秦腔冷开场: {dur:.1f}s, {n}句歌词, {total_frames}帧")

        # 逐帧渲染黑屏+歌词
        import cv2
        for fi in range(total_frames):
            t = fi / fps
            line_idx = int(t / line_dur)
            text = lines[min(line_idx, n - 1)]

            img = Image.new("RGB", (1920, 1080), (0, 0, 0))
            draw = ImageDraw.Draw(img)

            # 淡入淡出
            line_t = (t % line_dur) / line_dur
            alpha = 255
            if line_t < 0.1:
                alpha = int(255 * line_t / 0.1)
            elif line_t > 0.85:
                alpha = int(255 * (1.0 - line_t) / 0.15)

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx, ty = (1920 - tw) // 2, (1080 - th) // 2
            draw.text((tx, ty), text, font=font, fill=(alpha, alpha, alpha))

            frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{tmpdir_short}/f_{fi:06d}.png", frame_bgr)

        # Step 1: 一步生成冷开场片段（黑屏帧 + 秦腔音频）
        cold_seg = ctx.output_dir / "_cold_open_seg.mp4"
        cmd1 = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", f"{tmpdir_short}/f_%06d.png",
            "-i", audio_file,
            "-c:v", "libx264", "-preset", "fast", "-crf", "1",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(dur),
            str(cold_seg)
        ]
        r1 = subprocess.run(cmd1, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60)

        shutil.rmtree(tmpdir, ignore_errors=True)

        if r1.returncode != 0:
            print(f"    冷开场片段失败: {r1.stderr[-300:]}")
            ctx.set("coldopen_path", input_path)
            return

        # 验证冷开场时长
        vprobe = subprocess.run(
            [ffprobe, "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(cold_seg)],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        cold_dur = float(vprobe.stdout.strip()) if vprobe.stdout.strip() else 0
        print(f"    冷开场验证: {cold_dur:.1f}s")

        # 不自行拼接，由 export 阶段统一拼接片头+正片+片尾
        ctx.set("coldopen_path", str(cold_seg))
        print(f"    输出: {cold_seg.name} (交由 export 拼接)")

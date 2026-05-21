"""阶段32: 变速效果

节拍点慢动作 (0.6x)，非节拍正常速度。
"""

import subprocess, shutil, json
from pathlib import Path
from lib.utils import path_exists


class SpeedRampStage:
    def run(self, ctx):
        if ctx.get("speedramp_path") and path_exists(ctx.get("speedramp_path")):
            print("    已存在，跳过")
            return

        cfg = ctx.config.get("speed_ramp", {})
        if not cfg.get("enabled", False):
            return

        input_path = (ctx.get("pip_path") or
                     ctx.get("mascot_path") or
                     ctx.get("watermark_path") or
                     ctx.get("energybar_path") or
                     str(ctx.input_path))
        if not input_path or not path_exists(input_path):
            return

        video_info = ctx.get("video_info")
        fps = video_info.get("fps", 30)
        total_frames = video_info.get("process_frames", video_info["frames"])
        duration = total_frames / fps if fps > 0 else 60

        beat_frames = sorted(set(ctx.get("beat_frames", [])))
        if not beat_frames:
            print("    跳过: 无节拍数据")
            ctx.set("speedramp_path", input_path)
            return

        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"

        slow_speed = cfg.get("slow_speed", 0.6)
        slow_window = cfg.get("slow_window", 0.15)  # 节拍前后慢放秒数

        # 构建节拍慢放段
        slow_segments = []
        for bf in beat_frames:
            bt = bf / fps
            s = max(0, bt - slow_window)
            e = min(duration, bt + slow_window)
            if not slow_segments or s > slow_segments[-1][1] + 0.1:
                slow_segments.append((s, e))
            else:
                slow_segments[-1] = (slow_segments[-1][0], e)

        if not slow_segments:
            ctx.set("speedramp_path", input_path)
            return

        slow_count = len(slow_segments)
        total_slow = sum(e - s for s, e in slow_segments)
        print(f"    变速: {slow_count}段慢动作, 共{total_slow:.1f}s (节拍驱动)")

        # 拆分 → 分别变速 → 拼接
        tmpdir = Path(str(ctx.output_dir)) / "_speed_tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        concat_file = tmpdir / "concat.txt"

        segs = []
        t = 0
        for s, e in slow_segments:
            if t < s:
                segs.append((t, s, 1.0))
            segs.append((s, e, slow_speed))
            t = e
        if t < duration:
            segs.append((t, duration, 1.0))

        part_idx = 0
        with open(concat_file, "w") as cf:
            for seg_start, seg_end, speed in segs:
                seg_dur = seg_end - seg_start
                if seg_dur < 0.05:
                    continue
                part_path = tmpdir / f"part_{part_idx:03d}.mp4"

                if abs(speed - 1.0) < 0.01:
                    cmd = [ffmpeg, "-y", "-ss", f"{seg_start:.3f}",
                           "-i", input_path, "-t", f"{seg_dur:.3f}",
                           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                           "-c:a", "aac", "-b:a", "128k",
                           str(part_path)]
                else:
                    pts_factor = 1.0 / speed
                    cmd = [ffmpeg, "-y", "-ss", f"{seg_start:.3f}",
                           "-i", input_path, "-t", f"{seg_dur:.3f}",
                           "-filter_complex",
                           f"[0:v]setpts={pts_factor:.3f}*PTS[v];"
                           f"[0:a]atempo={speed:.2f}[a]",
                           "-map", "[v]", "-map", "[a]",
                           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                           "-c:a", "aac", "-b:a", "128k",
                           str(part_path)]

                r = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=60)
                if r.returncode == 0 and Path(part_path).exists():
                    cf.write(f"file '{part_path.resolve().as_posix()}'\n")
                    part_idx += 1

        if part_idx == 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            ctx.set("speedramp_path", input_path)
            return

        out_path = ctx.output_dir / f"{Path(input_path).stem}_speedramp.mp4"
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0",
               "-i", str(concat_file), "-c", "copy", str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=120)
        shutil.rmtree(tmpdir, ignore_errors=True)

        if r.returncode != 0:
            print(f"    变速拼接失败: {r.stderr[-200:]}")
            ctx.set("speedramp_path", input_path)
            return

        ctx.set("speedramp_path", str(out_path))
        print(f"    输出: {out_path.name}")

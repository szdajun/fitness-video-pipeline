"""阶段35: 爆燃文字 — 动作峰值时弹出大字"""
import cv2, numpy as np, random, os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import tempfile, subprocess, shutil, ctypes, json
from lib.utils import path_exists

BURST_TEXTS = ["燃!", "炸!", "暴汗!", "高强度!", "拼了!", "冲!", "起飞!", "炸裂!",
               "燃爆!", "火力全开!", "燃脂巅峰!", "爆发!", "猛!", "太猛了!"]


class IntensityBurstStage:
    def run(self, ctx):
        if ctx.get("burst_path") and path_exists(ctx.get("burst_path")):
            print("    已存在，跳过")
            return

        cfg = ctx.config.get("intensity_burst", {"enabled": True})
        if not cfg.get("enabled", True):
            return

        input_path = (ctx.get("danmaku_path") or
                     ctx.get("energybar_path") or
                     ctx.get("highlight_path") or
                     ctx.get("beatflash_path") or
                     str(ctx.input_path))
        if not input_path or not path_exists(input_path):
            return

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Load keypoints for intensity calculation
        kp_file = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        beat_frames = set(ctx.get("beat_frames", []))

        # Calculate per-frame motion intensity from keypoints
        intensity = {}
        if kp_file.exists():
            with open(kp_file, encoding="utf-8") as f:
                raw = json.load(f)
            keypoints = raw.get("keypoints", raw)
            prev = None
            for fi_str in sorted(keypoints.keys(), key=int):
                fi = int(fi_str)
                frame_kps = keypoints[fi_str]
                motion = 0
                if frame_kps and prev:
                    for pk, ck in zip(prev, frame_kps):
                        pa = np.array(pk)
                        ca = np.array(ck)
                        vis = (pa[:, 2] > 0.3) & (ca[:, 2] > 0.3)
                        if vis.sum() >= 6:
                            motion += np.mean(np.sqrt((ca[vis, 0] - pa[vis, 0])**2 +
                                            (ca[vis, 1] - pa[vis, 1])**2))
                intensity[fi] = motion
                prev = frame_kps

        if not intensity:
            ctx.set("burst_path", input_path)
            return

        # Normalize intensity to 0-1
        vals = list(intensity.values())
        i_min, i_max = np.percentile(vals, 20), np.percentile(vals, 95)
        if i_max - i_min < 0.001:
            i_max = i_min + 1

        # Select burst moments: high intensity + beat sync
        burst_moments = []
        peak_threshold = 0.7
        for fi in range(0, max_frames, int(fps * 1.5)):  # every ~1.5 seconds check
            frame_intensity = intensity.get(fi, 0)
            norm_i = (frame_intensity - i_min) / (i_max - i_min)
            if norm_i > peak_threshold and (fi in beat_frames or random.random() < 0.5):
                burst_moments.append({
                    "frame": fi,
                    "intensity": norm_i,
                    "text": random.choice(BURST_TEXTS),
                })

        if not burst_moments:
            ctx.set("burst_path", input_path)
            return

        print(f"    爆燃文字: {len(burst_moments)} 处峰值")

        # Chinese font
        font_paths = ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"]
        pil_font = None
        for fp in font_paths:
            if os.path.exists(fp):
                pil_font = ImageFont.truetype(fp, int(orig_h * 0.12))
                break
        if not pil_font:
            pil_font = ImageFont.load_default()

        out_path = ctx.output_dir / f"{Path(input_path).stem}_burst.mp4"
        tmpdir = Path(tempfile.mkdtemp(prefix="burst_"))
        ffmpeg_bin = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        burst_lookup = {b["frame"]: b for b in burst_moments}
        burst_duration = int(fps * 0.6)  # 0.6 seconds

        cap = cv2.VideoCapture(input_path)
        fi = 0
        while fi < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Check active bursts
            for bf in range(fi - burst_duration, fi + 1):
                if bf in burst_lookup:
                    burst = burst_lookup[bf]
                    age = fi - bf
                    if 0 <= age < burst_duration:
                        progress = age / burst_duration
                        # Scale up then down
                        if progress < 0.15:
                            scale = progress / 0.15
                        elif progress > 0.75:
                            scale = (1.0 - progress) / 0.25
                        else:
                            scale = 1.0
                        alpha = min(1.0, scale * 1.5)

                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
                        draw = ImageDraw.Draw(overlay)

                        text = burst["text"]
                        bbox = pil_font.getbbox(text)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]

                        cx = orig_w // 2 + random.randint(-40, 40)
                        cy = orig_h // 3 + random.randint(-30, 30)

                        # Shadow + main text with scale
                        for ox, oy, color in [(3, 3, (0, 0, 0, int(180 * alpha))),
                                                (0, 0, (*random.choice([(255, 60, 30), (255, 200, 20), (255, 255, 255)]), int(255 * alpha)))]:
                            draw.text((cx - tw // 2 + ox, cy - th // 2 + oy), text,
                                     font=pil_font, fill=color)

                        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
                        frame = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                        break

            cv2.imwrite(str(tmpdir / f"f_{fi:06d}.png"), frame)
            fi += 1
            if fi % 500 == 0:
                print(f"    进度: {fi}/{max_frames}")

        cap.release()
        print(f"    写入完成: {fi} 帧, FFmpeg 编码...")

        r = subprocess.run([
            ffmpeg_bin, "-y", "-framerate", str(fps),
            "-i", str(tmpdir / "f_%06d.png"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "1",
            "-pix_fmt", "yuv420p", "-an", str(out_path),
        ], capture_output=True, text=True, encoding="utf-8", errors="replace")
        shutil.rmtree(tmpdir, ignore_errors=True)

        if r.returncode != 0:
            print(f"    FFmpeg 失败: {r.stderr[-200:]}")
            ctx.set("burst_path", input_path)
            return

        ctx.set("burst_path", str(out_path))
        print(f"    输出: {out_path.name}")

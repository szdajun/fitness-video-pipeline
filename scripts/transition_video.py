"""通用转场视频生成器

用法:
    # 生成所有类型转场
    python transition_video.py --auto

    # 在两个视频之间应用转场并合并
    python transition_video.py --apply "视频1.mp4" "视频2.mp4" "输出.mp4" --type fade
"""

import argparse, subprocess, shutil, uuid
import numpy as np, cv2
from pathlib import Path

FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"
OUTPUT_DIR = Path("F:/wkspace/fitness-video-pipeline/output/transitions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def make_avi(tmp_dir, name, width, height, fps, duration, generator_fn):
    path = tmp_dir / name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    frames = int(duration * fps)
    for i in range(frames):
        t = i / frames
        frame = generator_fn(t, width, height)
        writer.write(frame)
    writer.release()
    return path


def encode_h265(avi_path, output_path, crf=20):
    r = subprocess.run([
        FFMPEG, "-y", "-i", str(avi_path),
        "-c:v", "libx265", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-an", str(output_path)
    ], capture_output=True, text=True, errors="replace")
    return r.returncode == 0


def generate_fade(duration=1.5, fps=30, width=404, height=720, output=None):
    output = output or OUTPUT_DIR / "转场_fade.mp4"
    uid = uuid.uuid4().hex[:8]
    tmp = Path(f"F:/wkspace/fitness-video-pipeline/output/tmp_fade_{uid}")
    tmp.mkdir(exist_ok=True)
    def gen(t, w, h):
        alpha = 0.5 * (1 - np.cos(t * np.pi))
        v = int(80 * alpha)
        return np.full((h, w, 3), (v, v, v), dtype=np.uint8)
    avi = make_avi(tmp, "fade.avi", width, height, fps, duration, gen)
    ok = encode_h265(avi, output)
    shutil.rmtree(tmp)
    return output if ok else None


def generate_pulse(duration=2.0, fps=30, width=404, height=720, output=None):
    output = output or OUTPUT_DIR / "转场_pulse.mp4"
    uid = uuid.uuid4().hex[:8]
    tmp = Path(f"F:/wkspace/fitness-video-pipeline/output/tmp_pulse_{uid}")
    tmp.mkdir(exist_ok=True)
    def gen(t, w, h):
        alpha = 0.5 * (1 - np.cos(2 * np.pi * t))
        pulse = 0.15 * np.sin(4 * np.pi * t) * alpha
        total = np.clip(alpha + pulse, 0, 1)
        frame = np.full((h, w, 3), 50, dtype=np.float32)
        frame[:, :, 1] = int(140 * total)
        frame[:, :, 2] = int(255 * total)
        return frame.astype(np.uint8)
    avi = make_avi(tmp, "pulse.avi", width, height, fps, duration, gen)
    ok = encode_h265(avi, output)
    shutil.rmtree(tmp)
    return output if ok else None


def generate_flash(duration=1.0, fps=30, width=404, height=720, output=None):
    output = output or OUTPUT_DIR / "转场_flash.mp4"
    uid = uuid.uuid4().hex[:8]
    tmp = Path(f"F:/wkspace/fitness-video-pipeline/output/tmp_flash_{uid}")
    tmp.mkdir(exist_ok=True)
    def gen(t, w, h):
        if t < 0.2: alpha = t / 0.2
        elif t < 0.4: alpha = 1.0
        elif t < 0.7: alpha = 1.0 - (t - 0.4) / 0.3
        else: alpha = 0.0
        v = int(255 * alpha)
        return np.full((h, w, 3), (v, v, v), dtype=np.uint8)
    avi = make_avi(tmp, "flash.avi", width, height, fps, duration, gen)
    ok = encode_h265(avi, output)
    shutil.rmtree(tmp)
    return output if ok else None


def generate_wipe(duration=1.5, fps=30, width=404, height=720, output=None):
    output = output or OUTPUT_DIR / "转场_wipe.mp4"
    uid = uuid.uuid4().hex[:8]
    tmp = Path(f"F:/wkspace/fitness-video-pipeline/output/tmp_wipe_{uid}")
    tmp.mkdir(exist_ok=True)
    def gen(t, w, h):
        frame = np.zeros((h, w, 3), dtype=np.float32)
        center = 0.1 + 0.8 * t
        band_w = 0.10
        x = np.arange(w) / w
        band = np.exp(-((x - center) ** 2) / (2 * band_w ** 2))
        frame[:, :, 2] = band * 255 * 0.85
        frame[:, :, 1] = band * 160 * 0.85
        frame[:, :, 0] = band * 50 * 0.85
        return np.clip(frame, 0, 255).astype(np.uint8)
    avi = make_avi(tmp, "wipe.avi", width, height, fps, duration, gen)
    ok = encode_h265(avi, output)
    shutil.rmtree(tmp)
    return output if ok else None


def apply_transition(video1, video2, transition_type="fade", output=None):
    output = output or Path("F:/wkspace/fitness-video-pipeline/output/合并_带转场.mp4")
    trans_file = OUTPUT_DIR / f"转场_{transition_type}.mp4"
    if not trans_file.exists():
        gen_funcs = {"fade": generate_fade, "pulse": generate_pulse,
                     "flash": generate_flash, "wipe": generate_wipe}
        print(f"生成转场: {trans_file.name}")
        gen_funcs[transition_type](output=trans_file)
    uid = uuid.uuid4().hex[:8]
    tmp = Path(f"F:/wkspace/fitness-video-pipeline/output/tmp_merge_{uid}")
    tmp.mkdir(parents=True, exist_ok=True)
    v1, v2, tr = tmp / "v1.mp4", tmp / "v2.mp4", tmp / "trans.mp4"
    shutil.copy2(str(video1), str(v1))
    shutil.copy2(str(video2), str(v2))
    shutil.copy2(str(trans_file), str(tr))
    list_file = tmp / "list.txt"
    with open(list_file, "w") as f:
        f.write(f"file '{v1}'\nfile '{tr}'\nfile '{v2}'\n")
    out_raw = tmp / "raw.mp4"
    r = subprocess.run([
        FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c", "copy", str(out_raw)
    ], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"  concat失败: {r.stderr[-200:]}")
        shutil.rmtree(tmp)
        return None
    r = subprocess.run([
        FFMPEG, "-y", "-i", str(out_raw),
        "-c:v", "libx265", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest", str(output)
    ], capture_output=True, text=True, errors="replace")
    shutil.rmtree(tmp)
    return output if r.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description="通用转场视频生成器")
    parser.add_argument("--type", "-t", default="fade",
                        choices=["fade", "pulse", "flash", "wipe"])
    parser.add_argument("--duration", "-d", type=float, default=1.5)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--width", type=int, default=404)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--apply", nargs=3, metavar=("视频1", "视频2", "输出"))
    args = parser.parse_args()
    if args.apply:
        video1, video2, out = args.apply
        result = apply_transition(Path(video1), Path(video2), args.type, Path(out))
        if result:
            print(f"完成: {result}")
        return
    if args.auto:
        for t in ["fade", "pulse", "flash", "wipe"]:
            print(f"生成 {t}...", end=" ", flush=True)
            gen_funcs = {"fade": generate_fade, "pulse": generate_pulse,
                         "flash": generate_flash, "wipe": generate_wipe}
            out = gen_funcs[t]()
            if out:
                print(f"-> {out} ({out.stat().st_size//1024}KB)")
            else:
                print("失败")
        return
    gen_funcs = {"fade": generate_fade, "pulse": generate_pulse,
                 "flash": generate_flash, "wipe": generate_wipe}
    out = gen_funcs[args.type](duration=args.duration, width=args.width,
                               height=args.height, output=args.output)
    if out:
        print(f"完成: {out} ({out.stat().st_size//1024}KB)")


if __name__ == "__main__":
    main()
"""横版16x9磨皮处理 - 独立运行"""
import cv2, numpy as np, subprocess, shutil, os, tempfile
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
TEMP = os.environ.get("TEMP", "F:/wkspace/temp")
os.makedirs(TEMP, exist_ok=True)
OUT = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-22")

def to_short(p):
    import ctypes
    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    GetShortPathNameW.restype = ctypes.c_uint
    buf_size = GetShortPathNameW(str(p), None, 0)
    if buf_size == 0:
        return str(p)
    buf = ctypes.create_unicode_buffer(buf_size)
    GetShortPathNameW(str(p), buf, buf_size)
    return buf.value

def apply_skin_smooth(frame, strength=0.5, d=7, sigmaColor=15, sigmaSpace=15):
    if strength <= 0:
        return frame
    smoothed = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
    return cv2.addWeighted(frame, 1 - strength, smoothed, strength, 0)


def skin_smooth_video(input_path, output_path, strength=0.3, d=7, sigmaColor=15, sigmaSpace=15):
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"磨皮: {input_path.name} ({fw}x{fh}, {total}帧)")
    print(f"参数: strength={strength}, d={d}, sigmaColor={sigmaColor}, sigmaSpace={sigmaSpace}")

    tmpdir = Path(tempfile.mkdtemp(dir=TEMP, prefix="smooth_"))
    tmp_short = to_short(str(tmpdir))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = apply_skin_smooth(frame, strength, d, sigmaColor, sigmaSpace)
        cv2.imwrite(f"{tmp_short}/f_{frame_idx:06d}.png", processed)

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{total}")

    cap.release()

    r = subprocess.run([
        FFMPEG, "-y", "-framerate", str(fps),
        "-i", f"{tmp_short}/f_%06d.png",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        to_short(str(output_path))
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")

    shutil.rmtree(tmpdir, ignore_errors=True)

    if r.returncode != 0:
        print(f"编码失败: {r.stderr[-200:]}")
        return False
    else:
        size = os.path.getsize(output_path) / 1024 / 1024
        print(f"完成: {output_path.name} ({size:.1f}MB)")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="横版视频磨皮")
    parser.add_argument("--input", "-i", default="_h_rescaled.mp4", help="输入文件")
    parser.add_argument("--output", "-o", default="_h_smooth.mp4", help="输出文件")
    parser.add_argument("--strength", "-s", type=float, default=0.3, help="磨皮强度 0-1")
    parser.add_argument("--d", type=int, default=7, help="双边滤波直径")
    parser.add_argument("--sigmaColor", type=int, default=15, help="颜色标准差")
    parser.add_argument("--sigmaSpace", type=int, default=15, help="空间标准差")
    args = parser.parse_args()

    input_path = OUT / args.input
    output_path = OUT / args.output

    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
    else:
        skin_smooth_video(input_path, output_path,
                        strength=args.strength,
                        d=args.d,
                        sigmaColor=args.sigmaColor,
                        sigmaSpace=args.sigmaSpace)
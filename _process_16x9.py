"""横版16x9主视频处理：Ken Burns + 能量条 + 节拍闪光"""
import cv2, numpy as np, subprocess, shutil, os, tempfile, json, math, argparse
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
TEMP = os.environ.get("TEMP", "F:/wkspace/temp")
os.makedirs(TEMP, exist_ok=True)

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

def process_main(source, kp_file, out_path, fps=30.0):
    """处理主视频：缩放到1920x1080 + Ken Burns + 能量条 + 节拍闪光"""
    W, H = 1920, 1080

    # Ken Burns
    KB_CYCLE = 8.0
    KB_CLOSE_ZOOM = 1.2
    KB_PAN_AMP = 10

    # 能量条
    bar_w = 40
    bar_margin_right = 30
    bar_margin_bottom = 60
    bar_height = 350
    bar_x = W - bar_margin_right - bar_w
    bar_bottom = H - bar_margin_bottom
    bar_top = bar_bottom - bar_height

    EB_SMOOTH = 0.50
    EB_MIN_FILL = 0.05
    EB_MAX = 4.0

    # 加载关键点
    kp_data = {}
    if kp_file and Path(kp_file).exists():
        with open(kp_file, encoding="utf-8") as f:
            kp_data = json.load(f)

    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"源: {sw}x{sh}, {total}帧 @ {fps}fps")

    motion_smooth = 0.0
    beat_brightness = 0.0
    prev_kps = None
    prev_raw = 0.0
    frame_idx = 0

    tmpdir = Path(tempfile.mkdtemp(dir=TEMP, prefix="eb_"))
    tmp_short = to_short(str(tmpdir))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 缩放到1920x1080
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        t = frame_idx / fps

        # Ken Burns
        scene = 0.5 + 0.5 * math.sin(2 * math.pi * t / KB_CYCLE)
        zoom = 1.0 + (KB_CLOSE_ZOOM - 1.0) * scene
        pan_x = scene * KB_PAN_AMP * math.sin(2.5 * math.pi * t / KB_CYCLE)
        pan_y = scene * KB_PAN_AMP * 0.6 * math.sin(2.0 * math.pi * t / KB_CYCLE)

        scaled_w = int(W * zoom)
        scaled_h = int(H * zoom)
        scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        cx = (scaled_w - W) // 2 + int(pan_x)
        cy = (scaled_h - H) // 2 + int(pan_y)
        frame_kb = scaled[cy:cy+H, cx:cx+W]
        if frame_kb.shape[0] != H or frame_kb.shape[1] != W:
            frame_kb = cv2.resize(scaled, (W, H))

        # 运动量
        raw_motion = 0.0
        fd = kp_data.get(str(frame_idx))
        if fd and fd[0]:
            kps_arr = np.array(fd[0])
            conf = kps_arr[:, 2]
            kps = kps_arr[:, :2]
            if prev_kps is not None:
                diff = np.abs(kps - prev_kps)
                raw_motion = float(np.mean(diff[conf > 0.3]))
            prev_kps = kps.copy()
        else:
            if frame_idx > 0:
                cap2 = cv2.VideoCapture(str(source))
                cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                _, prev_frame = cap2.read()
                cap2.release()
                if prev_frame is not None:
                    prev_frame = cv2.resize(prev_frame, (W, H), interpolation=cv2.INTER_LINEAR)
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.cvtColor(frame_kb, cv2.COLOR_BGR2GRAY)
                    raw_motion = float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))))

        # 节拍闪光 - 按亮度归一化，避免亮场景过曝
        delta = abs(raw_motion - prev_raw)
        beat_brightness = max(delta * 2.0, beat_brightness * 0.85)
        beat_brightness = min(beat_brightness, 2.0)  # 限制最大值
        prev_raw = raw_motion
        avg_brightness = frame_kb.mean() / 255.0  # 0-1，越亮越降低加成
        beat_boost = 0.15 * beat_brightness / (avg_brightness + 0.5)
        brightness = 1.0 + beat_boost
        frame_kb = np.clip(frame_kb * brightness, 0, 255).astype(np.uint8)

        # 能量条
        motion_smooth = EB_SMOOTH * motion_smooth + (1 - EB_SMOOTH) * raw_motion
        fill_ratio = max(min(motion_smooth / EB_MAX, 1.0), EB_MIN_FILL)
        bar_fill_h = int(bar_height * fill_ratio)
        for y in range(bar_fill_h):
            alpha = y / bar_fill_h
            r = 255
            g = int(200 * alpha)
            b = 0
            cv2.line(frame_kb, (bar_x, bar_bottom - y), (bar_x + bar_w, bar_bottom - y), (b, g, r), 1)
        cv2.rectangle(frame_kb, (bar_x, bar_top), (bar_x + bar_w, bar_bottom), (100, 100, 100), 1)

        cv2.imwrite(f"{tmp_short}/f_{frame_idx:06d}.png", frame_kb)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{total}")

    cap.release()

    r = subprocess.run([
        FFMPEG, "-y", "-framerate", str(fps),
        "-i", f"{tmp_short}/f_%06d.png",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        to_short(str(out_path))
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")

    shutil.rmtree(tmpdir, ignore_errors=True)

    if r.returncode == 0:
        size = os.path.getsize(out_path) / 1024 / 1024
        print(f"完成: {out_path.name} ({size:.1f}MB)")
    else:
        print(f"失败: {r.stderr[-200:]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", "-s", required=True)
    p.add_argument("--kp", "-k", default=None)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--fps", "-f", type=float, default=30.0)
    args = p.parse_args()

    process_main(Path(args.source), args.kp, Path(args.output), args.fps)
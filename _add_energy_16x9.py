"""给横版主体添加能量条"""
import cv2, numpy as np, subprocess, shutil, os, tempfile, json
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

def add_energy_bars():
    # 加载关键点计算领操人运动
    kp_file = OUT / "丽丽3_cropped_keypoints.json"
    with open(kp_file, encoding="utf-8") as f:
        kp_data = json.load(f)

    fps = 30.0
    frames = 2015
    W, H = 1920, 1080

    # 能量条参数 (横版：放在右下角)
    bar_w = 40
    bar_margin_right = 30
    bar_margin_bottom = 60
    bar_height = 350
    bar_x = W - bar_margin_right - bar_w
    bar_bottom = H - bar_margin_bottom
    bar_top = bar_bottom - bar_height

    EB_SMOOTH = 0.70
    EB_MIN_FILL = 0.05
    EB_MAX = 0.3

    cap = cv2.VideoCapture(str(OUT / "_h_rescaled.mp4"))
    motion_smooth = 0.0
    prev_kps = None
    frame_idx = 0

    tmpdir = Path(tempfile.mkdtemp(dir=TEMP, prefix="eb_"))
    tmp_short = to_short(str(tmpdir))

    print(f"能量条: bar=({bar_x},{bar_top})-({bar_x+bar_w},{bar_bottom})")

    while frame_idx < frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 从关键点计算运动
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
            # 用帧差分
            if frame_idx > 0:
                cap2 = cv2.VideoCapture(str(OUT / "_h_rescaled.mp4"))
                cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
                _, prev_frame = cap2.read()
                cap2.release()
                if prev_frame is not None:
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    raw_motion = float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))))

        motion_smooth = EB_SMOOTH * motion_smooth + (1 - EB_SMOOTH) * raw_motion
        fill_ratio = max(min(motion_smooth / EB_MAX, 1.0), EB_MIN_FILL)

        # 绘制能量条 (渐变红→黄，高能量更醒目)
        bar_fill_h = int(bar_height * fill_ratio)
        for y in range(bar_fill_h):
            alpha = y / bar_fill_h
            r = 255
            g = int(200 * alpha)
            b = 0
            cv2.line(frame, (bar_x, bar_bottom - y), (bar_x + bar_w, bar_bottom - y), (b, g, r), 1)
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bottom), (100, 100, 100), 1)

        cv2.imwrite(f"{tmp_short}/f_{frame_idx:06d}.png", frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{frames}")

    cap.release()

    out_path = OUT / "_h_main_new.mp4"
    r = subprocess.run([
        FFMPEG, "-y", "-framerate", str(fps),
        "-i", f"{tmp_short}/f_%06d.png",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        to_short(str(out_path))
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")

    if r.returncode != 0:
        print(f"编码失败: {r.stderr[-200:]}")
    else:
        print(f"完成: {out_path.name}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path

if __name__ == "__main__":
    add_energy_bars()

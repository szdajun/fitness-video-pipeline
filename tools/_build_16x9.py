"""
横版16:9视频处理脚本
直接从原始视频生成，不经过竖版转换，从根本上解决地面鼓起问题。
"""
import cv2, subprocess, shutil, numpy as np, sys, json, os, tempfile
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

def build_horizontal():
    orig_path = "C:/Users/18091/Desktop/短视频素材/丽丽3.mp4"
    output_dir = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-22")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取原始视频信息
    cap = cv2.VideoCapture(orig_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"原始: {orig_w}x{orig_h}, {fps}fps, {frames}帧")

    # 确定16:9 crop参数（居中裁切）
    # 原始1280x720, 目标1920x1080
    # 从720高度中裁出1080宽*720/1920=405高度区域是不可能的
    # 所以需要scale后再crop
    # 1280x720 -> scale to 1920:1080 -> 1920x1080 (刚好)
    # 或者: crop中心区域再scale
    # 原始视频是16:9(1280x720)，直接scale到1920x1080即可填满

    out_w, out_h = 1920, 1080

    # === 1. 从原始视频scale到1920x1080 ===
    scaled_path = output_dir / "_h_scaled.mp4"
    r = subprocess.run([
        FFMPEG, "-y", "-i", orig_path,
        "-vf", f"scale={out_w}:{out_h}:flags=bilinear",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an", str(scaled_path)
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"scale失败: {r.stderr[-200:]}")
        return
    print(f"缩放完成: {scaled_path.name}")

    # === 2. 添加能量条 ===
    print("生成能量条...")
    bar_w = 32
    bar_margin_right = 20
    bar_margin_bottom = 60
    bar_height = 300
    bar_x = out_w - bar_margin_right - bar_w
    bar_top = out_h - bar_margin_bottom - bar_height
    bar_bottom = out_h - bar_margin_bottom

    EB_SMOOTH = 0.85
    EB_MIN_FILL = 0.15
    EB_MAX = 10.0

    tmpdir = Path(tempfile.mkdtemp(dir=TEMP, prefix="eb_"))
    tmpdir_short = to_short(str(tmpdir))

    cap_in = cv2.VideoCapture(str(scaled_path))
    motion_smooth = 0.0
    prev_frame = None
    frame_idx = 0

    while frame_idx < frames:
        ret, frame = cap_in.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算运动量
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            motion = float(np.mean(diff))
        else:
            motion = 0.0
        prev_frame = gray

        motion_smooth = EB_SMOOTH * motion_smooth + (1 - EB_SMOOTH) * motion
        raw_ratio = min(motion_smooth / EB_MAX, 1.0)
        fill_ratio = max(raw_ratio, EB_MIN_FILL)

        # 绘制能量条
        bar_frame_h = int(bar_height * fill_ratio)
        bar_color = (0, 200, 255)  # 蓝绿色
        # 渐变填充
        for y in range(bar_frame_h):
            alpha = y / bar_frame_h
            r_col = int(0 * (1 - alpha) + 0 * alpha)
            g_col = int(180 * (1 - alpha) + 220 * alpha)
            b_col = int(255 * (1 - alpha) + 100 * alpha)
            cv2.line(frame, (bar_x, bar_bottom - y), (bar_x + bar_w, bar_bottom - y), (b_col, g_col, r_col), 1)
        # 边框
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + bar_w, bar_bottom), (200, 200, 200), 1)

        cv2.imwrite(f"{tmpdir_short}/f_{frame_idx:06d}.png", frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  {frame_idx}/{frames}")

    cap_in.release()

    eb_out = output_dir / "_h_energybar.mp4"
    r = subprocess.run([
        FFMPEG, "-y", "-framerate", str(fps),
        "-i", f"{tmpdir_short}/f_%06d.png",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", str(eb_out)
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"能量条编码失败: {r.stderr[-200:]}")
    else:
        print(f"能量条完成: {eb_out.name}")
    shutil.rmtree(tmpdir, ignore_errors=True)

    # === 3. 添加片头片尾 ===
    intro_out = output_dir / "_h_intro.mp4"
    outro_out = output_dir / "_h_outro.mp4"

    # 片头
    intro_seconds = 4.0
    intro_frames_n = int(intro_seconds * fps)
    intro_tmpdir = Path(tempfile.mkdtemp(dir=TEMP, prefix="intro_"))
    intro_tmpdir_short = to_short(str(intro_tmpdir))

    # 片头背景：黑色渐变
    for i in range(intro_frames_n):
        t = i / intro_frames_n
        frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        # 淡入
        alpha = min(t * 2, 1.0)
        # 频道名
        font = cv2.FONT_HERSHEY_SIMPLEX
        text1 = "胭脂虎健身团"
        text2 = "带操人:丽丽"
        text3 = "西安时代广场"
        cv2.putText(frame, text1, (out_w//2 - 200, out_h//2 - 60), font, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, text2, (out_w//2 - 150, out_h//2 + 20), font, 1.2, (200, 200, 100), 2)
        cv2.putText(frame, text3, (out_w//2 - 150, out_h//2 + 80), font, 1.0, (150, 150, 150), 1)
        # 淡入效果
        frame = (frame * alpha).astype(np.uint8)
        cv2.imwrite(f"{intro_tmpdir_short}/f_{i:04d}.png", frame)

    r = subprocess.run([
        FFMPEG, "-y", "-framerate", str(fps),
        "-i", f"{intro_tmpdir_short}/f_%04d.png",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", str(intro_out)
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode == 0:
        print(f"片头完成: {intro_out.name}")
    else:
        print(f"片头失败: {r.stderr[-200:]}")
    shutil.rmtree(intro_tmpdir, ignore_errors=True)

    # 片尾
    outro_seconds = 2.5
    outro_frames_n = int(outro_seconds * fps)
    outro_tmpdir = Path(tempfile.mkdtemp(dir=TEMP, prefix="outro_"))
    outro_tmpdir_short = to_short(str(outro_tmpdir))

    for i in range(outro_frames_n):
        t = i / outro_frames_n
        frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        # 淡入
        alpha = min(t * 2, 1.0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "胭脂虎健身团", (out_w//2 - 200, out_h//2 - 40), font, 1.5, (255, 255, 255), 2)
        cv2.putText(frame, "关注不迷路", (out_w//2 - 150, out_h//2 + 40), font, 1.2, (255, 255, 50), 2)
        frame = (frame * alpha).astype(np.uint8)
        cv2.imwrite(f"{outro_tmpdir_short}/f_{i:04d}.png", frame)

    r = subprocess.run([
        FFMPEG, "-y", "-framerate", str(fps),
        "-i", f"{outro_tmpdir_short}/f_%04d.png",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", str(outro_out)
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode == 0:
        print(f"片尾完成: {outro_out.name}")
    else:
        print(f"片尾失败: {r.stderr[-200:]}")
    shutil.rmtree(outro_tmpdir, ignore_errors=True)

    # === 4. 合并片头+主体+片尾 ===
    print("合并视频...")
    list_path = output_dir / "_h_concat.txt"
    with open(list_path, "w") as f:
        f.write(f"file '{intro_out}'\n")
        f.write(f"file '{eb_out}'\n")
        f.write(f"file '{outro_out}'\n")

    final_path = output_dir / "丽丽3_16x9.mp4"
    r = subprocess.run([
        FFMPEG, "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy", "-an", str(output_dir / "_h_combined.mp4")
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"concat失败: {r.stderr[-200:]}")
        return

    # === 5. 添加音频 ===
    audio_path = output_dir / "丽丽3_audio.aac"
    r = subprocess.run([
        FFMPEG, "-y",
        "-i", str(output_dir / "_h_combined.mp4"),
        "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "copy",
        "-shortest",
        str(final_path)
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"音频合并失败: {r.stderr[-200:]}")
        # 复制无音频版本
        shutil.copy(output_dir / "_h_combined.mp4", final_path)
    else:
        print(f"最终输出: {final_path.name}")

    # 清理中间文件
    for f in [scaled_path, eb_out, intro_out, outro_out, list_path, output_dir / "_h_combined.mp4"]:
        if f.exists():
            f.unlink()

    print("完成!")

if __name__ == "__main__":
    build_horizontal()

"""横版 1920x1080 专用片头片尾生成"""
import cv2, numpy as np, subprocess, shutil, os, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FFMPEG = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
TEMP = os.environ.get("TEMP", "F:/wkspace/temp")
os.makedirs(TEMP, exist_ok=True)

OUT = Path("F:/wkspace/fitness-video-pipeline/output/2026-04-22")

# 固定配置
W, H = 1920, 1080
FPS = 30.0

def get_font(size):
    """加载中文字体"""
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc", # 宋体
    ]
    for f in font_candidates:
        if os.path.exists(f):
            try:
                return ImageFont.truetype(f, size)
            except:
                pass
    return ImageFont.load_default()

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

def make_intro():
    """横版片头：真实视频前4秒背景 + 三行文字叠加"""
    print("生成横版片头...")
    intro_frames = int(4.0 * FPS)

    cap = cv2.VideoCapture(str(OUT / "_h_rescaled.mp4"))
    prev = None
    bg_frames = []

    for i in range(intro_frames):
        ret, frame = cap.read()
        if not ret:
            break
        bg_frames.append(frame)
        prev = frame
    cap.release()

    tmpdir = Path(tempfile.mkdtemp(dir=TEMP, prefix="intro_"))
    tmp_short = to_short(str(tmpdir))

    for i, frame in enumerate(bg_frames):
        t = i / intro_frames
        alpha = min(t * 1.5, 1.0)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        W_img, H_img = pil_img.size

        overlay_top = int(H_img * 0.70)
        draw.rectangle([(0, overlay_top), (W_img, H_img)], fill=(0, 0, 0))

        font_lg = get_font(int(H_img * 0.07))
        channel = "胭脂虎健身团"
        bbox = draw.textbbox((0, 0), channel, font=font_lg)
        tw = bbox[2] - bbox[0]
        cy1 = int(overlay_top * 0.30)
        draw.text(((W_img - tw) // 2, cy1), channel, font=font_lg, fill=(255, 255, 255))

        font_md = get_font(int(H_img * 0.09))
        lead_text = "带操人：丽丽"
        bbox = draw.textbbox((0, 0), lead_text, font=font_md)
        tw = bbox[2] - bbox[0]
        cy2 = int(H_img * 0.45)
        draw.text(((W_img - tw) // 2, cy2), lead_text, font=font_md, fill=(255, 220, 50))

        font_sm = get_font(int(H_img * 0.05))
        date_text = "西安时代广场 / 2026-04-20"
        bbox = draw.textbbox((0, 0), date_text, font=font_sm)
        tw = bbox[2] - bbox[0]
        cy3 = overlay_top + int((H_img - overlay_top) * 0.3)
        draw.text(((W_img - tw) // 2, cy3), date_text, font=font_sm, fill=(200, 200, 200))

        frame_out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        frame_out = (frame_out * alpha).astype(np.uint8)
        cv2.imwrite(f"{tmp_short}/f_{i:04d}.png", frame_out)

    out_path = OUT / "_h_intro_new.mp4"
    r = subprocess.run([
        FFMPEG, "-y", "-framerate", str(FPS),
        "-i", f"{tmp_short}/f_%04d.png",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        to_short(str(out_path))
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"片头失败: {r.stderr[-200:]}")
    else:
        print(f"片头完成: {out_path.name}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path

def make_outro():
    """横版片尾：黑底 + CTA + 淡出"""
    print("生成横版片尾...")
    outro_frames = int(2.5 * FPS)

    frames = []
    for i in range(outro_frames):
        t = i / outro_frames
        alpha = 1.0 - min(t * 0.5, 0.7)  # 淡出到30%黑

        frame = np.zeros((H, W, 3), dtype=np.uint8)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 第1行：打工牛马，健身达人
        font_lg = get_font(int(H * 0.08))
        channel = "打工牛马，健身达人"
        bbox = draw.textbbox((0, 0), channel, font=font_lg)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) // 2, int(H * 0.30)), channel, font=font_lg, fill=(255, 255, 255))

        # 第2行：点击关注，一起蜕变
        font_md = get_font(int(H * 0.10))
        cta = "点击关注，一起蜕变"
        bbox = draw.textbbox((0, 0), cta, font=font_md)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) // 2, int(H * 0.50)), cta, font=font_md, fill=(255, 255, 50))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        frame = (frame * alpha).astype(np.uint8)
        frames.append(frame)

    tmpdir = Path(tempfile.mkdtemp(dir=TEMP, prefix="outro_"))
    tmp_short = to_short(str(tmpdir))
    for i, f in enumerate(frames):
        cv2.imwrite(f"{tmp_short}/f_{i:04d}.png", f)

    out_path = OUT / "_h_outro_new.mp4"
    r = subprocess.run([
        FFMPEG, "-y", "-framerate", str(FPS),
        "-i", f"{tmp_short}/f_%04d.png",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an",
        to_short(str(out_path))
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"片尾失败: {r.stderr[-200:]}")
    else:
        print(f"片尾完成: {out_path.name}")
    shutil.rmtree(tmpdir, ignore_errors=True)
    return out_path

if __name__ == "__main__":
    intro = make_intro()
    outro = make_outro()
    print("完成!")

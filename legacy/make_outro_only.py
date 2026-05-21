"""Standalone: 直接从 energybar.mp4 生成片尾（新设计：大衬底）"""
import sys
sys.path.insert(0, '.')

import cv2, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from lib.utils import create_writer

def _get_font(size):
    font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttc", "C:/Windows/Fonts/simsun.ttc"]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except:
            pass
    return ImageFont.load_default()

def _draw_outro_text_pil(frame, cta_text: str):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    # 文字区域衬底：覆盖画面下半部分（y=38%到100%）
    substrate_top = int(h * 0.38)
    draw.rectangle([(0, substrate_top), (w, h)], fill=(10, 10, 10))

    # CTA 文字（居中，大字，亮黄色）
    font_lg = _get_font(int(h * 0.12))
    bbox = draw.textbbox((0, 0), cta_text, font=font_lg)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (w - tw) // 2
    ty = substrate_top + int((h - substrate_top - th) * 0.25)
    draw.text((cx, ty), cta_text, font=font_lg, fill=(255, 255, 50))

    # 小字"点击关注"（白色）
    font_sm = _get_font(int(h * 0.07))
    sub = "点击关注"
    bbox = draw.textbbox((0, 0), sub, font=font_sm)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (w - tw) // 2
    ty2 = substrate_top + int((h - substrate_top - th) * 0.65)
    draw.text((cx, ty2), sub, font=font_sm, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def create_outro(video_path, output_path, cta_text, duration=5.0, fps=30.0):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    start_frame = max(0, int(total_frames - duration * actual_fps))
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = create_writer(str(output_path), actual_fps, width, height)
    frame_count = 0
    max_frames = int(duration * actual_fps)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _draw_outro_text_pil(frame, cta_text)
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"片尾已生成: {output_path} ({frame_count}帧)")

video = "output/2026-04-14/丽丽1_energybar.mp4"
out = "output/2026-04-14/丽丽1_outro_v2.mp4"
create_outro(video, out, "关注不迷路", duration=5.0, fps=30.0)
print("Done!")
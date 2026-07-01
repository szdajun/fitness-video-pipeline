"""tools/render_cta_overlay.py — PIL 渲染 CTA 文字到 RGBA PNG

ffmpeg 8.1 drawtext 解析 UTF-8 静默失败, 用 PIL 画中文 + ffmpeg overlay 叠加避开.

输出 PNG: 1080x1920 透明背景, CTA 三行 + 红分割线
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


# 字体路径 (沿用 legacy)
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

# 中文/英文 fallback (PIL 支持多字节字)
_FONT_CANDIDATES = [FONT_BOLD, FONT, r"C:\Windows\Fonts\simhei.ttf"]


def _load_font(size: int):
    for fp in _FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


# CTA 三行内容 (钉死 - 跟之前 shorts_legacy_filters 一致)
CTA_LINES = (
    "点赞 LIKE & SUBSCRIBE 关注",          # 点赞 LIKE & SUBSCRIBE 关注
    "完整版 Full Workout on Channel",             # 完整版 Full Workout on Channel
    "新视频 New Videos Daily",                     # 新视频 New Videos Daily
)


def render_cta_png(
    output_path: str,
    canvas_w: int = 1080,
    canvas_h: int = 1920,
    lines: tuple = CTA_LINES,
):
    """渲染 CTA 文字到透明背景 PNG (1080x1920).

    Args:
        output_path: 输出的 PNG 路径
        canvas_w/h: 9:16 画布尺寸, 默认 1080x1920 (匹配抖音/Shorts)
        lines: CTA 3 行文字 (yellow + white + gray)

    Returns:
        output_path
    """
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 红分割线
    sep_text = "─" * 10  # ──...
    sep_font = _load_font(40)
    sep_color = (220, 30, 30, 255)
    try:
        sw = draw.textlength(sep_text, font=sep_font)
    except AttributeError:
        sw = sep_font.getlength(sep_text)
    draw.text(((canvas_w - sw) / 2, canvas_h * 0.74), sep_text,
              font=sep_font, fill=sep_color)

    # 3 行 CTA
    font_sizes = (60, 48, 32)
    colors = ((255, 220, 0, 255), (255, 255, 255, 255), (180, 180, 180, 255))
    y_starts = (canvas_h * 0.78, canvas_h * 0.84, canvas_h * 0.90)

    for text, size, color, y in zip(lines, font_sizes, colors, y_starts):
        font = _load_font(size)
        try:
            tw = draw.textlength(text, font=font)
        except AttributeError:
            tw = font.getlength(text)
        # 中文字体描边 (让文字在视频上更清晰)
        draw.text(((canvas_w - tw) / 2, y), text, font=font, fill=color,
                  stroke_width=2, stroke_fill=(0, 0, 0, 200))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "tools/_test_cta.png"
    p = render_cta_png(out)
    print(f"Rendered: {p}")

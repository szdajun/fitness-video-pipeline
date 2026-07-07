"""stages/render_short_overlay.py — PIL 渲染诗词+CTA 到 RGBA PNG (1080x1920)

规避 ffmpeg 8.1 drawtext 含 UTF-8 中文静默失败 bug (CLAUDE.md 已知问题).

用法:
    render_opening(out_png, coach="艳青")     # 片头诗词 PNG
    render_cta(out_png)                       # 片尾 CTA PNG (SUBSCRIBE 等英文)
"""
import os
import sys
from pathlib import Path

# 让 render_cta_overlay 引用 PIL 不被混
from PIL import Image, ImageDraw, ImageFont

# 字体路径 (CLAUDE.md 钉死的中文字体)
FONT_REG = r"C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = r"C:/Windows/Fonts/msyhbd.ttc"
# 2026-07-07: msyhbd 无 🔥(U+1F525) 字形 → 渲染成方框(tofu). 用 Segoe UI Emoji 渲染 emoji.
FONT_EMOJI = r"C:/Windows/Fonts/seguiemj.ttf"

# 导入 coach_profiles 拿诗词和英文标题
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.coach_profiles import (
    get_shorts_poem, get_shorts_en, DEFAULT_SHORTS_POEM, DEFAULT_SHORTS_EN,
)


def _load_font(path: str, size: int):
    if os.path.exists(path):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_opening(out_png: str, coach: str = "", size=(1080, 1920), duration: float = 30.0):
    """片头: 英文标题 + 副标题 + 中文诗词 (渐显渐隐 ~3.5s)

    渐显渐隐用 PIL 在每行底部画 RGBA 蒙版列不行 (PNG 是静态的),
    实际 ffmpeg overlay 用 enable='gte(t,0)*lte(t,3.5)' 控制整 PNG 显隐.
    """
    W, H = size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    en = get_shorts_en(coach) or DEFAULT_SHORTS_EN
    poem = get_shorts_poem(coach) or DEFAULT_SHORTS_POEM

    title = en.get("title", DEFAULT_SHORTS_EN["title"])
    subtitle = en.get("subtitle", DEFAULT_SHORTS_EN["subtitle"])

    title_font = _load_font(FONT_BOLD, 56)
    sub_font = _load_font(FONT_REG, 40)
    poem_font = _load_font(FONT_REG, 44)

    def draw_centered(text, font, y_frac, fill, stroke_w=3, stroke=(0, 0, 0, 220)):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x = (W - tw) // 2
        y = int(H * y_frac)
        # 描边 (黑色外发光) 提高可读性
        if stroke_w > 0:
            for dx in range(-stroke_w, stroke_w + 1):
                for dy in range(-stroke_w, stroke_w + 1):
                    if dx * dx + dy * dy <= stroke_w * stroke_w:
                        draw.text((x + dx, y + dy), text, font=font, fill=stroke)
        draw.text((x, y), text, font=font, fill=fill)

    # 顶部半透明黑底 (诗词区可读)
    bg_top = int(H * 0.0)
    bg_bot = int(H * 0.42)
    overlay_bg = Image.new("RGBA", (W, bg_bot - bg_top), (0, 0, 0, 140))
    img.paste(overlay_bg, (0, bg_top), overlay_bg)
    draw = ImageDraw.Draw(img)

    # Title (黄色)
    draw_centered(title, title_font, 0.02, fill=(255, 220, 0, 255), stroke_w=3)
    # Subtitle (白色)
    draw_centered(subtitle, sub_font, 0.085, fill=(255, 255, 255, 255), stroke_w=2)

    # 诗词 (黄色多行)
    poem_lines = [l.strip() for l in poem.strip().split("\n") if l.strip()]
    for i, line in enumerate(poem_lines):
        y_frac = 0.17 + i * 0.055
        if y_frac > 0.4:
            break
        draw_centered(line, poem_font, y_frac, fill=(255, 220, 0, 255), stroke_w=2)

    img.save(out_png, "PNG")
    print(f"  [opening] PIL 渲染: {os.path.basename(out_png)} ({W}x{H})")
    return out_png


def render_cta(out_png: str, size=(1080, 1920)):
    """片尾 CTA: 红分割线 + 黄大字 + 副标 + 灰色 @ 账号"""
    W, H = size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(FONT_BOLD, 70)
    sub_font = _load_font(FONT_REG, 44)
    handle_font = _load_font(FONT_REG, 32)
    line_font = _load_font(FONT_REG, 36)

    def draw_centered(text, font, y_frac, fill, stroke_w=3, stroke=(0, 0, 0, 240)):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x = (W - tw) // 2
        y = int(H * y_frac)
        if stroke_w > 0:
            for dx in range(-stroke_w, stroke_w + 1):
                for dy in range(-stroke_w, stroke_w + 1):
                    if dx * dx + dy * dy <= stroke_w * stroke_w:
                        draw.text((x + dx, y + dy), text, font=font, fill=stroke)
        draw.text((x, y), text, font=font, fill=fill)

    # 底部半透明黑底
    bg_top = int(H * 0.62)
    overlay_bg = Image.new("RGBA", (W, H - bg_top), (0, 0, 0, 170))
    img.paste(overlay_bg, (0, bg_top), overlay_bg)
    draw = ImageDraw.Draw(img)

    # 红分割线
    line_y = int(H * 0.66)
    draw.line([(int(W*0.1), line_y), (int(W*0.9), line_y)], fill=(220, 30, 30, 255), width=6)

    # CTA 主体 (2026-06-29: 中英双语, 中文为主 — 用户主要中文)
    draw_centered("关注订阅", title_font, 0.69,
                  fill=(255, 220, 0, 255), stroke_w=4)
    draw_centered("点赞 · 分享 · 收藏", sub_font, 0.77,
                  fill=(255, 255, 255, 255), stroke_w=2)
    draw_centered("SUBSCRIBE for more fitness", sub_font, 0.83,
                  fill=(255, 220, 0, 255), stroke_w=2)
    draw_centered("@xiliuying_fit  细柳营健身", handle_font, 0.90,
                  fill=(190, 190, 190, 255), stroke_w=1)

    img.save(out_png, "PNG")
    print(f"  [cta] PIL 渲染: {os.path.basename(out_png)} ({W}x{H})")
    return out_png


def render_preview(out_png: str, size=(1080, 1920), duration: float = 4.0):
    """高燃预警 PNG (hook 段全程常驻, 不做渐显渐隐).

    设计 (2026-07-07 高燃预览开场):
      - 主标题 "🔥 高燃预警" 居中偏上, 橙红色 (警示色, 与 opening 黄/cta 黄区分)
      - 副标 "先睹为快" 主标下方, 黄色
      - 中部半透明黑底 提高可读性, 不挡领操人上半身动作
      - 字号比 cta (70) 更大 (110), 突出 "燃"
      - hook 与教练无关 (全教练统一), 不调 coach_profiles / get_shorts_poem
    """
    W, H = size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    title_font = _load_font(FONT_BOLD, 110)
    emoji_font = _load_font(FONT_EMOJI, 104)  # 🔥 用 Segoe UI Emoji (msyhbd 无此字形→方框)
    sub_font = _load_font(FONT_REG, 48)

    def draw_centered(text, font, y_frac, fill, stroke_w=5, stroke=(0, 0, 0, 240)):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x = (W - tw) // 2
        y = int(H * y_frac)
        if stroke_w > 0:
            for dx in range(-stroke_w, stroke_w + 1):
                for dy in range(-stroke_w, stroke_w + 1):
                    if dx * dx + dy * dy <= stroke_w * stroke_w:
                        draw.text((x + dx, y + dy), text, font=font, fill=stroke)
        draw.text((x, y), text, font=font, fill=fill)

    def draw_emoji_cjk_centered(emoji, cjk, emoji_f, cjk_f, y_frac, fill,
                                stroke_w=5, stroke=(0, 0, 0, 240), gap=14):
        """🔥(emoji 字体) + CJK(粗体) 居中拼接. emoji 不加描边 (避免糊掉 seguiemj 彩色字形)."""
        eb = draw.textbbox((0, 0), emoji, font=emoji_f)
        cb = draw.textbbox((0, 0), cjk, font=cjk_f)
        ew = eb[2] - eb[0]
        cw = cb[2] - cb[0]
        total = ew + gap + cw
        x0 = (W - total) // 2
        y = int(H * y_frac)
        # emoji (无描边; 彩色字形由 seguiemj COLR/CPAL 表提供, fill 仅对单色字形生效)
        draw.text((x0, y), emoji, font=emoji_f, fill=fill)
        xc = x0 + ew + gap
        if stroke_w > 0:
            for dx in range(-stroke_w, stroke_w + 1):
                for dy in range(-stroke_w, stroke_w + 1):
                    if dx * dx + dy * dy <= stroke_w * stroke_w:
                        draw.text((xc + dx, y + dy), cjk, font=cjk_f, fill=stroke)
        draw.text((xc, y), cjk, font=cjk_f, fill=fill)

    # 中部半透明黑底 (不挡领操人上半身, 字幕区可读)
    bg_top = int(H * 0.38)
    bg_bot = int(H * 0.56)
    overlay_bg = Image.new("RGBA", (W, bg_bot - bg_top), (0, 0, 0, 130))
    img.paste(overlay_bg, (0, bg_top), overlay_bg)
    draw = ImageDraw.Draw(img)

    # 主标题: 🔥(emoji 字体) + 高燃预警(粗体), 橙红 居中拼接.
    # 2026-07-07: 🔥 用 seguiemj 单独渲染 (msyhbd 无此字形→方框); emoji 不加描边.
    draw_emoji_cjk_centered("🔥", "高燃预警", emoji_font, title_font, 0.41,
                            fill=(255, 80, 30, 255), stroke_w=5)
    # 副标: 黄
    draw_centered("先睹为快", sub_font, 0.51,
                  fill=(255, 220, 0, 255), stroke_w=3)

    img.save(out_png, "PNG")
    print(f"  [preview] PIL 渲染: {os.path.basename(out_png)} ({W}x{H})")
    return out_png


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "_test.png"
    coach = sys.argv[2] if len(sys.argv) > 2 else "艳青"
    if "cta" in out.lower():
        render_cta(out)
    elif "preview" in out.lower() or "hook" in out.lower():
        render_preview(out)
    else:
        render_opening(out, coach=coach)

"""汉印水印叠加 — 优先用 AI 生成的 PNG, 代码版 PIL 兜底

实现要点:
- 优先加载 tools/seal_ai.png (AI 生成的方形汉印, 红边+美女头+老虎+繁体字)
- 找不到时用 PIL 绘制圆形篆体朱文红章 (代码版)
- alpha-blend 到原帧
- 支持 top-left/top-right/bottom-left/bottom-right 四个位置
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# AI 版汉印 (优先)
_SEAL_AI_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "seal_ai.png")

# 字体路径 (代码版兜底)
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simfang.ttf",
    r"C:\Windows\Fonts\simkai.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for fp in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _render_ai_seal(seal_path: str, size: int, alpha: float) -> Image.Image:
    """加载 AI 生成的汉印 PNG (RGBA)"""
    seal = Image.open(seal_path).convert("RGBA")
    seal = seal.resize((size, size), Image.LANCZOS)
    # 应用 alpha
    if alpha < 1.0:
        r, g, b, a = seal.split()
        a = a.point(lambda v: int(v * alpha))
        seal = Image.merge("RGBA", (r, g, b, a))
    # 微微旋转 (-3°~+3°)
    seal = seal.rotate(np.random.RandomState(123).uniform(-3, 3),
                       resample=Image.BICUBIC, expand=False)
    return seal


def _render_code_seal(text: str, size: int, alpha: float) -> Image.Image:
    """代码版圆形篆体红章 (AI 版不可用时的兜底)"""
    seal_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(seal_img)
    seal_red = (180, 30, 30, int(255 * alpha))

    # 外圆 + 内圆 (双圈)
    border_w = max(3, size // 40)
    draw.ellipse([(0, 0), (size - 1, size - 1)], outline=seal_red, width=border_w)
    inner_pad = max(4, size // 20)
    draw.ellipse([(inner_pad, inner_pad),
                 (size - 1 - inner_pad, size - 1 - inner_pad)],
                outline=seal_red, width=max(1, border_w - 1))

    # 文字
    n_chars = len(text)
    if n_chars <= 2:
        font_size = int(size * 0.45)
        font = _load_font(font_size)
        total_w = sum(draw.textlength(c, font=font) for c in text) + (n_chars - 1) * 4
        x = (size - total_w) / 2
        y = (size - font_size) / 2
        for c in text:
            cw = draw.textlength(c, font=font)
            draw.text((x, y), c, font=font, fill=seal_red)
            x += cw + 4
    else:
        chars = list(text)[:4]
        while len(chars) < 4:
            chars.append(" ")
        font_size = int(size * 0.30)
        font = _load_font(font_size)
        cell = size // 2
        offsets = [(0, 0), (cell, 0), (0, cell), (cell, cell)]
        for c, (ox, oy) in zip(chars, offsets):
            if c == " ":
                continue
            cw = draw.textlength(c, font=font)
            cx = ox + (cell - cw) / 2
            cy = oy + (cell - font_size) / 2
            draw.text((cx, cy), c, font=font, fill=seal_red)

    # 斑驳纹理
    seal_arr = np.array(seal_img)
    noise = np.random.RandomState(42).randint(0, 30, seal_arr.shape[:2], dtype=np.uint8)
    red_mask = (seal_arr[:, :, 0] > 50) & (seal_arr[:, :, 3] > 0)
    for c in range(3):
        ch = seal_arr[:, :, c]
        ch[red_mask] = np.clip(ch[red_mask].astype(int) - noise[red_mask].astype(int), 0, 255).astype(np.uint8)
    seal_img = Image.fromarray(seal_arr)
    seal_img = seal_img.rotate(np.random.RandomState(123).uniform(-3, 3),
                               resample=Image.BICUBIC, expand=False)
    return seal_img


def overlay_seal(frame: np.ndarray, text: str = "", pos: str = "top-left",
                 size: int = 130, margin: int = 30, alpha: float = 0.70,
                 **kwargs) -> np.ndarray:
    """叠加汉印到帧上.

    Args:
        frame: BGR ndarray
        text: 印面文字 (代码版兜底用, AI 版忽略)
        pos: 'top-left' / 'top-right' / 'bottom-left' / 'bottom-right'
        size: 印章直径
        margin: 边距
        alpha: 透明度 0~1
    """
    if not text:
        text = "胭脂虎"

    h, w = frame.shape[:2]

    # 优先用 AI 生成的 PNG
    if os.path.exists(_SEAL_AI_PATH):
        seal_img = _render_ai_seal(_SEAL_AI_PATH, size, alpha)
    else:
        seal_img = _render_code_seal(text, size, alpha)

    # 计算位置
    if pos == "top-left":
        x0, y0 = margin, margin
    elif pos == "top-right":
        x0, y0 = w - size - margin, margin
    elif pos == "bottom-left":
        x0, y0 = margin, h - size - margin
    elif pos == "bottom-right":
        x0, y0 = w - size - margin, h - size - margin
    else:
        x0, y0 = margin, margin

    # alpha-blend 到原帧
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_pil.paste(seal_img, (x0, y0), seal_img)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
"""把 ComfyUI 出的主体图 + PIL 合成红边 + 繁体「胭脂虎」, 输出最终方形汉印 PNG

输入: tools/_seal_candidates/seal_yanhu_*.png
处理:
  1. 方形 1024x1024, 外圈红色边框 (12px)
  2. 下 1/3 留白压「胭脂虎」三个繁体字, 朱红, 微软正黑体/楷体
  3. 加轻微斑驳纹理 (仿印章不规则感)
  4. 微微旋转 (-3°~+3°) 模拟手盖
输出: tools/seal_ai.png (替换原版, lib/seal.py 自动加载)

用法:
    python tools/compose_seal.py tools/_seal_candidates/seal_yanhu_00_seed42.png
    python tools/compose_seal.py tools/_seal_candidates/seal_yanhu_00_seed42.png --border red
    python tools/compose_seal.py --batch  # 批量处理目录下所有候选, 选最大一张打 *_final.png
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 字体候选 (繁體中文, 优先系统已有)
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",      # 微软正黑体 (繁体)
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑 (简体)
    r"C:\Windows\Fonts\msjhbd.ttc",    # 微软正黑体粗体
    r"C:\Windows\Fonts\simfang.ttf",   # 仿宋
    r"C:\Windows\Fonts\simkai.ttf",    # 楷体
    r"C:\Windows\Fonts\kaiu.ttf",      # 标楷体
]

# 繁體「胭脂虎」
TEXT_TRADITIONAL = "胭脂虎"

# 印面主色 (朱红)
SEAL_RED = (200, 30, 30)
BORDER_COLOR = (200, 30, 30)


def _load_font(size: int):
    for fp in FONT_CANDIDATES:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _add_distress(img: Image.Image, intensity: int = 25) -> Image.Image:
    """仿手工盖印的不规则破损"""
    arr = np.array(img).copy()
    if arr.shape[2] < 4:
        return img
    alpha = arr[:, :, 3]
    # 在 alpha>0 的区域随机挖洞
    mask = alpha > 30
    noise = np.random.RandomState(7).randint(0, 100, alpha.shape)
    erode = mask & (noise < intensity)
    arr[erode, 3] = (arr[erode, 3].astype(int) * 0.3).astype(np.uint8)
    return Image.fromarray(arr)


def compose(src_path: str, dst_path: str,
            text=TEXT_TRADITIONAL,
            border_px: int = 14,
            size: int = 1024,
            rotate_deg=None,
            distress: bool = True) -> bool:
    """合成方形红边汉印"""
    try:
        main = Image.open(src_path).convert("RGBA")
    except Exception as e:
        print(f"[compose] 打不开 {src_path}: {e}", file=sys.stderr)
        return False

    # 缩放到方形画布
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    main = main.resize((size, size), Image.LANCZOS)
    canvas.paste(main, (0, 0), main)

    draw = ImageDraw.Draw(canvas)

    # 1) 红色方形外边框
    draw.rectangle(
        [(border_px // 2, border_px // 2), (size - border_px // 2 - 1, size - border_px // 2 - 1)],
        outline=BORDER_COLOR + (255,), width=border_px,
    )
    # 内框 (双线)
    inner_pad = border_px + 6
    draw.rectangle(
        [(inner_pad, inner_pad), (size - inner_pad - 1, size - inner_pad - 1)],
        outline=BORDER_COLOR + (180,), width=max(1, border_px // 3),
    )

    # 2) 简体下 1/3 区域半透明白带 + 繁体字
    text_band_h = size // 3
    text_y0 = size - text_band_h
    band = Image.new("RGBA", (size, text_band_h), (255, 255, 255, 220))
    canvas.alpha_composite(band, (0, text_y0))

    # 字体撑满 80% 高度
    font_size = int(text_band_h * 0.78)
    font = _load_font(font_size)
    # 三个字均分宽度, 每格中心对位
    cell_w = size / 3
    for i, ch in enumerate(text[:3]):
        cw = draw.textlength(ch, font=font)
        # 垂直基线略偏上, 视觉居中
        bbox = font.getbbox(ch)
        ch_h = bbox[3] - bbox[1]
        cx = int(i * cell_w + (cell_w - cw) / 2)
        cy = int(text_y0 + (text_band_h - ch_h) / 2 - bbox[1])
        draw.text((cx, cy), ch, font=font, fill=SEAL_RED + (255,))

    # 3) 斑驳
    if distress:
        canvas = _add_distress(canvas, intensity=22)

    # 4) 微微旋转
    if rotate_deg is None:
        rotate_deg = float(np.random.RandomState(123).uniform(-3.0, 3.0))
    canvas = canvas.rotate(rotate_deg, resample=Image.BICUBIC, expand=False)

    canvas.save(dst_path, "PNG")
    print(f"[compose] -> {dst_path}  text='{text}'  rotate={rotate_deg:.1f}°")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=None, help="输入 PNG; 不传则取 _seal_candidates 第一张")
    ap.add_argument("--dst", default=None, help="输出 PNG; 不传则覆盖 tools/seal_ai.png")
    ap.add_argument("--border", type=int, default=14, help="边框粗细 px")
    ap.add_argument("--text", default=TEXT_TRADITIONAL, help="印面繁体字")
    ap.add_argument("--no-distress", action="store_true")
    ap.add_argument("--batch", action="store_true", help="处理 _seal_candidates 下所有候选, 输出 _final.png")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cand_dir = os.path.join(here, "_seal_candidates")
    default_dst = os.path.join(here, "seal_ai.png")

    if args.batch:
        files = sorted(glob.glob(os.path.join(cand_dir, "*.png")))
        if not files:
            print(f"[compose] 候选目录为空: {cand_dir}", file=sys.stderr)
            sys.exit(1)
        ok = 0
        for f in files:
            out = os.path.splitext(f)[0] + "_final.png"
            if compose(f, out, text=args.text, border_px=args.border, distress=not args.no_distress):
                ok += 1
        print(f"[compose] 批量完成, {ok}/{len(files)} 张 -> {cand_dir}")
        return

    src = args.src
    if not src:
        cands = sorted(glob.glob(os.path.join(cand_dir, "*.png")))
        # 排除已生成的 *_final.png
        cands = [c for c in cands if not c.endswith("_final.png")]
        if not cands:
            print(f"[compose] 未指定 src, 且 {cand_dir} 无候选", file=sys.stderr)
            print("  先跑: python tools/gen_seal.py", file=sys.stderr)
            sys.exit(1)
        src = cands[0]
        print(f"[compose] 自动取第一张: {src}")

    dst = args.dst or default_dst
    ok = compose(src, dst, text=args.text, border_px=args.border, distress=not args.no_distress)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""把 seal_ai.png 套到测试帧上, 输出预览图验收

不依赖 ffmpeg / 真实视频 — 用一张纯色或合成的 1080x1920 抖音帧当背景,
四个角落各盖一次, 输出 4 张 PNG (top-left / top-right / bottom-left / bottom-right).

用法:
    python tools/preview_seal.py                # 用纯白背景
    python tools/preview_seal.py bg.png         # 用真实视频帧
    python tools/preview_seal.py --text 细柳营   # 改 fallback 文字 (代码版兜底用)
"""
import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

# 让 lib/seal.py 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.seal import overlay_seal  # noqa: E402


def make_canvas(src, size=(1080, 1920)):
    if src and os.path.exists(src):
        img = cv2.imread(src)
        return cv2.resize(img, size)
    # 默认: 浅灰渐变, 模拟健身视频常见底色
    h, w = size[1], size[0]
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        v = int(60 + 80 * (y / h))
        bg[y, :] = (v, v, v)
    # 加点文字方块当 dummy
    cv2.putText(bg, "DUMMY FRAME 1080x1920", (80, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
    return bg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bg", nargs="?", default=None, help="背景图 PNG/JPG; 不传=合成灰底")
    ap.add_argument("--text", default="", help="印面文字 (AI 版忽略, 走 fallback 才生效)")
    ap.add_argument("--size", type=int, default=180, help="印面直径")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "_seal_preview"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bg = make_canvas(args.bg)
    for pos in ("top-left", "top-right", "bottom-left", "bottom-right"):
        out = overlay_seal(bg.copy(), text=args.text, pos=pos,
                           size=args.size, margin=30, alpha=0.85)
        dst = os.path.join(args.out_dir, f"seal_preview_{pos}.png")
        cv2.imwrite(dst, out)
        print(f"[preview] {pos:14s} -> {dst}")

    print(f"[preview] 完成, 4 张预览 -> {args.out_dir}")


if __name__ == "__main__":
    main()

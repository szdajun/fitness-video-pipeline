"""竖屏画中画小窗 (shorts pip) 守门测试.

2026-07-07: 竖屏 9:16 从 16:9 裁切丢左右画面, 加 16:9 全景小窗补场景.
compute_pip_rect 算右上避开领操人的 (x,y,w,h); make_vertical 加 pip_src/pip_enabled.
默认 pip_src=None → 不加小窗, 行为与旧版一致 (现有 test_shorts_stage.py 17 测试不受影响).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from stages.short_vertical import compute_pip_rect


def _make_person(cx=0.5, cy=0.5, scale=0.15, conf=0.95):
    """造 33 点假人 (归一化). 上半身 (头/肩/肘/腕/髋) 围绕 (cx, cy) 上方, 供 pip 避让测试."""
    kps = [[cx, cy, conf]] * 33
    for i in range(7):                       # 头 0-6
        kps[i] = [cx + (i - 3) * 0.02, cy - scale * 0.6, conf]
    kps[11] = [cx - scale * 0.3, cy - scale * 0.3, conf]  # 左肩
    kps[12] = [cx + scale * 0.3, cy - scale * 0.3, conf]  # 右肩
    kps[13] = [cx - scale * 0.4, cy - scale * 0.1, conf]  # 左肘
    kps[14] = [cx + scale * 0.4, cy - scale * 0.1, conf]  # 右肘
    kps[15] = [cx - scale * 0.5, cy + scale * 0.1, conf]  # 左腕
    kps[16] = [cx + scale * 0.5, cy + scale * 0.1, conf]  # 右腕
    kps[23] = [cx - scale * 0.25, cy + scale * 0.2, conf]  # 左髋
    kps[24] = [cx + scale * 0.25, cy + scale * 0.2, conf]  # 右髋
    for i in range(25, 33):                  # 腿膝脚 下方
        kps[i] = [cx + (0.1 if i % 2 else -0.1), cy + scale * 0.8, conf]
    return kps


class TestComputePipRect:
    def test_no_kp_returns_fallback_top_right(self):
        x, y, w, h = compute_pip_rect({}, [(0, 10, 600)])
        assert x == 1080 - w - 24   # 右贴边
        assert y == 24              # 顶部 margin
        assert h == round(w * 9 / 16)

    def test_no_crop_segments_returns_fallback(self):
        kp = {0: [_make_person()]}
        x, y, w, h = compute_pip_rect(kp, [])
        assert x == 1080 - w - 24

    def test_16by9_aspect_ratio(self):
        kp = {i: [_make_person()] for i in range(30)}
        x, y, w, h = compute_pip_rect(kp, [(0, 30, 656)])
        assert h == round(w * 9 / 16)

    def test_within_vertical_bounds(self):
        kp = {i: [_make_person()] for i in range(30)}
        x, y, w, h = compute_pip_rect(kp, [(0, 30, 656)])
        assert x >= 0 and y >= 0
        assert x + w <= 1080
        assert y + h <= 1920

    def test_lead_centered_pip_top_right(self):
        """领操人居中 (cx=0.5, cy=0.5) → 小窗右上: y 最靠上 (领操人在中下, 顶部空)."""
        kp = {i: [_make_person(cx=0.5, cy=0.5, scale=0.15)] for i in range(60)}
        x, y, w, h = compute_pip_rect(kp, [(0, 60, 656)])  # crop_x=656 居中
        assert x == 1080 - w - 24      # 右贴边
        assert y == 24                 # 顶部 (领操人上半身在 y≈788-1017, 不挡顶部)
        assert w == 480 and h == 270

    def test_multi_segment_uses_each_crop_x(self):
        """多段 crop (合并视频) 每段 crop_x 不同, 不崩, 仍在竖屏内."""
        kp = {i: [_make_person(cx=0.5)] for i in range(120)}
        segs = [(0, 60, 400), (60, 120, 1100)]  # 两段不同 crop_x
        x, y, w, h = compute_pip_rect(kp, segs)
        assert x + w <= 1080 and y + h <= 1920
        assert h == round(w * 9 / 16)

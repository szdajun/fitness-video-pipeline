"""tests/test_shorts_stage.py — 39_shorts 重构后的纯算法层测试

不依赖 ffmpeg / insightface / GPU.
测试覆盖:
  1. cx 裁切算法 (前 N 帧 cx 中位数 + padding clamp + fallback 居中)
  2. profile → CTA/intro 模板映射
  3. intro_outro 时长探测

集成测试 (跑 ffmpeg 看产物) 不在本文件, 需在 docs/manual.md 写明手动跑法.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── 测试 fixture: 造 fake keypoints.json ──────────────


def _make_person(cx_norm: float, body_w: float = 0.20, n_valid: int = 10):
    """造一个 BlazePose 33kp person, 鼻子在 cx_norm, 肩宽 body_w"""
    kp = [[0.0, 0.0, 0.0]] * 33
    kp[0] = [cx_norm, 0.30, 0.9]
    kp[11] = [cx_norm - body_w / 2, 0.50, 0.9]
    kp[12] = [cx_norm + body_w / 2, 0.50, 0.9]
    # 填几个身体点, 让 find_lead_person 的 valid count >= 5
    for i, (x, y) in enumerate([
        (cx_norm - 0.02, 0.35), (cx_norm + 0.02, 0.38),
        (cx_norm - body_w / 3, 0.55), (cx_norm + body_w / 3, 0.55),
        (cx_norm - 0.05, 0.70), (cx_norm + 0.05, 0.70),
    ]):
        if n_valid > 3 + i:
            kp[13 + i] = [x, y, 0.9]
    return kp


def _make_kp_dict(frames: list) -> dict:
    """frames: [(frame_idx_str, [person, ...]), ...]"""
    return {str(i): persons for i, persons in frames}


# ── 1. cx 裁切算法测试 ──────────────────────────────


class TestCropXCxMedian:
    """验证 _compute_crop_x (待实现): 前 N 帧 cx 中位数 → crop_x"""

    def test_centered_lead_gives_mid_crop(self):
        """领操人在画面中央 → crop_x ≈ (1920-608)/2 = 656"""
        from stages.short_vertical import compute_crop_x_from_kp
        kp = _make_kp_dict([(i, [_make_person(0.50)]) for i in range(60)])
        crop_x = compute_crop_x_from_kp(kp, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        assert abs(crop_x - 656) < 5, f"领操人在中央时 crop_x 应≈656, 得 {crop_x}"

    def test_left_lead_shifts_crop_left(self):
        """领操人在左侧 → crop_x 偏左"""
        from stages.short_vertical import compute_crop_x_from_kp
        kp = _make_kp_dict([(i, [_make_person(0.30)]) for i in range(60)])
        crop_x = compute_crop_x_from_kp(kp, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        # cx=0.30 → crop_x = (0.30 - 608/1920/2) * 1920 = (0.30 - 0.158) * 1920 ≈ 272
        assert crop_x < 500, f"左侧领操人 crop_x 应<500, 得 {crop_x}"
        assert crop_x > 100, f"左侧领操人 crop_x 应>100 (padding), 得 {crop_x}"

    def test_right_lead_shifts_crop_right(self):
        from stages.short_vertical import compute_crop_x_from_kp
        kp = _make_kp_dict([(i, [_make_person(0.70)]) for i in range(60)])
        crop_x = compute_crop_x_from_kp(kp, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        # cx=0.70 → crop_x = (0.70 - 0.158) * 1920 ≈ 1040
        assert crop_x > 800, f"右侧领操人 crop_x 应>800, 得 {crop_x}"
        assert crop_x < 1300, f"右侧领操人 crop_x 应<1300 (padding), 得 {crop_x}"

    def test_padding_clamps_extreme(self):
        """领操人 cx=0.05 极端偏左 → crop_x 钳到 padding 边界 (>=30)"""
        from stages.short_vertical import compute_crop_x_from_kp
        kp = _make_kp_dict([(i, [_make_person(0.05)]) for i in range(60)])
        crop_x = compute_crop_x_from_kp(kp, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        assert crop_x >= 30, f"极端左应钳到 padding=30, 得 {crop_x}"

    def test_padding_clamps_extreme_right(self):
        from stages.short_vertical import compute_crop_x_from_kp
        kp = _make_kp_dict([(i, [_make_person(0.95)]) for i in range(60)])
        crop_x = compute_crop_x_from_kp(kp, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        # max = 1920-608-30 = 1282
        assert crop_x <= 1282, f"极端右应钳到 1920-608-30=1282, 得 {crop_x}"

    def test_median_robust_to_outliers(self):
        """前 60 帧里有几个 cx 异常, 中位数应稳定"""
        from stages.short_vertical import compute_crop_x_from_kp
        frames = []
        for i in range(57):
            frames.append((i, [_make_person(0.50)]))
        # 加 3 个异常帧
        frames.append((57, [_make_person(0.05)]))
        frames.append((58, [_make_person(0.05)]))
        frames.append((59, [_make_person(0.95)]))
        kp = _make_kp_dict(frames)
        crop_x = compute_crop_x_from_kp(kp, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        assert abs(crop_x - 656) < 20, f"中位数应抗异常, 得 {crop_x}"

    def test_only_lookhead_n_frames(self):
        """只看前 N 帧, 后面的不参与"""
        from stages.short_vertical import compute_crop_x_from_kp
        # 前 60 帧全居中
        frames = [(i, [_make_person(0.50)]) for i in range(60)]
        # 60~120 帧 cx 跑偏
        for i in range(60, 120):
            frames.append((i, [_make_person(0.10)]))
        kp = _make_kp_dict(frames)
        crop_x = compute_crop_x_from_kp(kp, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        # 只看前 60 帧, 全是 0.50 → crop_x ≈ 656
        assert abs(crop_x - 656) < 5, f"只看前 60 帧应得居中, 得 {crop_x}"

    def test_empty_kp_falls_back_to_center(self):
        """kp 为空 dict → fallback crop_x=656"""
        from stages.short_vertical import compute_crop_x_from_kp
        crop_x = compute_crop_x_from_kp({}, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        assert crop_x == 656

    def test_no_valid_lead_falls_back(self):
        """前 60 帧所有 person 都不满足 find_lead_person 门槛 → fallback 居中"""
        from stages.short_vertical import compute_crop_x_from_kp
        # 全是空 person (None 或空 list)
        kp = _make_kp_dict([(i, []) for i in range(60)])
        crop_x = compute_crop_x_from_kp(kp, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        assert crop_x == 656

    def test_multiple_persons_picks_lead(self):
        """多人时, 应选 cx 居中 + body 最大的 (复用 find_lead_person)"""
        from stages.short_vertical import compute_crop_x_from_kp
        frames = []
        for i in range(60):
            # A: 中央大 (领操人), B: 左侧小 (观众)
            frames.append((i, [
                _make_person(0.50, body_w=0.30),  # 大
                _make_person(0.20, body_w=0.10),  # 小
            ]))
        kp = _make_kp_dict(frames)
        crop_x = compute_crop_x_from_kp(kp, frame_w=1920, crop_w=608,
                                        lookhead_frames=60)
        assert abs(crop_x - 656) < 30, f"应选中央大的人, crop_x≈656, 得 {crop_x}"


# ── 2. profile 映射测试 ──────────────────────────────


class TestProfileMapping:
    """make_vertical 的 profile 参数应正确决定 CTA/intro 模板"""

    def test_yt_shorts_profile_has_cta(self):
        from stages.short_vertical import get_overlay_filters
        result = get_overlay_filters(profile="yt_shorts", coach="郭海军", duration=30)
        # CTA 应该出现 (英文 SUBSCRIBE 等关键字)
        joined = result if isinstance(result, str) else " ".join(result)
        assert "SUBSCRIBE" in joined.upper() or "CTA" in joined.upper(), \
            f"yt_shorts 应有 CTA filter, 实际: {joined[:200]}"

    def test_douyin_profile_no_cta(self):
        from stages.short_vertical import get_overlay_filters
        result = get_overlay_filters(profile="douyin", coach="郭海军", duration=None)
        joined = result if isinstance(result, str) else " ".join(result)
        assert "SUBSCRIBE" not in joined.upper(), \
            f"douyin 不应有 CTA, 实际: {joined[:200]}"

    def test_douyin_uses_chinese_intro(self):
        from stages.short_vertical import get_overlay_filters
        result = get_overlay_filters(profile="douyin", coach="郭海军", duration=None)
        joined = result if isinstance(result, str) else " ".join(result)
        # 至少有一个 drawtext, 中文 / 简体标识
        assert "drawtext" in joined

    def test_invalid_profile_raises(self):
        from stages.short_vertical import get_overlay_filters
        with pytest.raises(ValueError, match="profile"):
            get_overlay_filters(profile="weird_unknown", coach="x", duration=30)


# ── 3. intro_outro 时长探测测试 ─────────────────────


class TestIntroOutroDetection:
    """make_vertical 需要拿到 intro_outro 时长用于 -ss 跳过"""

    def test_intro_outro_from_explicit_seconds(self):
        """传 intro_seconds=4.0 时, 应返回 4.0"""
        from stages.short_vertical import resolve_intro_skip
        skip = resolve_intro_skip(intro_path=None, outro_path=None, intro_seconds=4.0)
        assert skip == 4.0

    def test_intro_outro_from_probe_path(self):
        """给定 intro_path, ffprobe 出时长"""
        from stages.short_vertical import resolve_intro_skip
        # 用 2026-06-27 的真实 intro.mp4 (3.97s)
        intro_p = ROOT / "tools/_test_data/jianling_baseline_2026-06-27/建玲1_intro.mp4"
        if intro_p.exists():
            skip = resolve_intro_skip(intro_path=str(intro_p), outro_path=None,
                                      intro_seconds=None)
            assert 3.5 <= skip <= 4.5, f"intro.mp4 应≈4s, 探测得 {skip}"

    def test_intro_outro_default_when_no_data(self):
        """都没给 → 默认 5s (宽屏常见 intro 时长)"""
        from stages.short_vertical import resolve_intro_skip
        skip = resolve_intro_skip(intro_path=None, outro_path=None, intro_seconds=None)
        assert skip == 5.0

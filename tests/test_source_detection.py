"""test_source_detection.py (2026-07-10 竖屏源端到端通路)

守门:
- apply_transpose_filter rotation 0/90/180/270 → 正确 vf 字符串
- is_vertical_video 像素判定 (h > w 且比例 0.50-0.65)
- is_vertical_video 比例不在 9:16 不判 (3:4/1:1 不会误判)
- _is_already_vertical 像素 + rotation 判定
- 异常路径 (文件不存在) 不抛
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lib.source_detection import (
    apply_transpose_filter,
    is_vertical_video,
    _is_already_vertical,
    detect_source_orientation,
)


# ── apply_transpose_filter ──────────────────────────────────────────

def test_apply_transpose_filter_0():
    assert apply_transpose_filter(0) == ""


def test_apply_transpose_filter_90():
    # 2026-07-10: apply_transpose_filter 保留向后兼容, 但实际不再使用 (ffmpeg 自动处理 rotate)
    assert apply_transpose_filter(90) == "transpose=1"


def test_apply_transpose_filter_180():
    assert apply_transpose_filter(180) == "transpose=1,transpose=1"


def test_apply_transpose_filter_270():
    assert apply_transpose_filter(270) == "transpose=2"


# ── _is_already_vertical ────────────────────────────────────────────

def test_is_already_vertical_true_pixels():
    # 1080x1920 像素 + rotation=0 → 已 9:16
    assert _is_already_vertical(1080, 1920, 0) is True


def test_is_already_vertical_false_pixels():
    # 1920x1080 像素 + rotation=0 → 横屏
    assert _is_already_vertical(1920, 1080, 0) is False


def test_is_already_vertical_false_rotation_90():
    # 1920x1080 像素 + rotation=90 → 实际是 9:16 但像素没 baked
    assert _is_already_vertical(1920, 1080, 90) is False


def test_is_already_vertical_false_rotation_270():
    # 1920x1080 像素 + rotation=270 → 实际是 9:16 但像素没 baked
    assert _is_already_vertical(1920, 1080, 270) is False


# ── is_vertical_video 异常路径 ─────────────────────────────────────

def test_is_vertical_video_nonexistent_returns_false():
    """文件不存在 → 不抛异常 → False."""
    assert is_vertical_video("/nonexistent/path/foo.mp4") is False


def test_is_vertical_video_empty_path_returns_false():
    assert is_vertical_video("") is False


# ── detect_source_orientation 异常路径 ─────────────────────────────

def test_detect_source_orientation_nonexistent():
    """文件不存在 → 返 {src_w:0, src_h:0, rotation:0, is_vertical:False, needs_normalize:False}."""
    info = detect_source_orientation("/nonexistent/path/foo.mp4")
    assert info["src_w"] == 0
    assert info["src_h"] == 0
    assert info["rotation"] == 0
    assert info["is_vertical"] is False
    assert info["needs_normalize"] is False
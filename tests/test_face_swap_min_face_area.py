"""测试 face_swap min_face_area 默认值必须 ≤ 0.001 — 防止远景领操人被过滤

固化规则:
- swap_face() 默认 min_face_area=0.001
- 原因: 健身视频全身镜头里领操人脸常只占 0.5-1%, 0.02 会全过滤
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FS = Path(__file__).parent.parent / "tools" / "face_swap.py"
src = FS.read_text(encoding="utf-8")


def test_swap_face_default_min_face_area():
    """swap_face() 默认参数 min_face_area 必须 ≤ 0.001"""
    m = re.search(r'def swap_face\([^)]*min_face_area=([\d.]+)', src, re.DOTALL)
    assert m, "找不到 swap_face 的 min_face_area 默认值"
    val = float(m.group(1))
    assert val <= 0.001, f"min_face_area 默认值 {val} 太大, 领操人会被过滤. 应 ≤ 0.001"


def test_process_video_default_min_face_area():
    """process_video() 默认参数 min_face_area 必须 ≤ 0.001"""
    m = re.search(r'def process_video\([^)]*min_face_area=([\d.]+)', src, re.DOTALL)
    assert m, "找不到 process_video 的 min_face_area 默认值"
    val = float(m.group(1))
    assert val <= 0.001, f"process_video min_face_area 默认值 {val} 太大"


def test_docstring_mentions_reason():
    """docstring 应说明为什么默认这么小 (远景领操人)"""
    assert "领操人" in src or "健身" in src, \
        "docstring 应解释默认 min_face_area 小的原因, 否则容易被改回"
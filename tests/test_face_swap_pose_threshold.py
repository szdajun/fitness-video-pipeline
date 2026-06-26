"""真跑 face_swap 函数的回归测试

之前 test_face_swap_no_gfpgan.py 全是字符串 grep, 漏掉了 NameError.
本文件 import 模块, 真调 get_lead_bbox_from_pose + find_lead_person,
固化 lead_orientation_threshold 参数链路, 防止再次 silent broken.
"""
import sys
import os
from pathlib import Path

import pytest

# 让 tools/ 可导入
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import tools.face_swap as fs  # noqa: E402


def _make_kp(nose=(0.5, 0.3, 0.9),
             ls=(0.45, 0.5, 0.9),
             rs=(0.55, 0.5, 0.9),
             n_visible=10):
    """造一个 33-关键点 person, 默认 10 个有效点 (含鼻子+双肩+其他)"""
    kp = [[0.0, 0.0, 0.0]] * 33
    kp[0] = list(nose)
    kp[11] = list(ls)
    kp[12] = list(rs)
    # 填若干身体中部点, 让 valid count >= 5 (find_lead_person 的最低门槛)
    extras = [
        (0.48, 0.35), (0.50, 0.40), (0.52, 0.42),  # 脖子/胸
        (0.46, 0.55), (0.54, 0.55),  # 髋
        (0.43, 0.65), (0.57, 0.65),  # 膝
        (0.42, 0.80), (0.58, 0.80),  # 脚踝
    ]
    for i, (x, y) in enumerate(extras[:max(0, n_visible - 3)]):
        kp[13 + i] = [x, y, 0.9]
    return kp


def test_get_lead_bbox_threshold_param_exists():
    """get_lead_bbox_from_pose 必须接受 lead_orientation_threshold 参数 (修 NameError)"""
    import inspect
    sig = inspect.signature(fs.get_lead_bbox_from_pose)
    assert "lead_orientation_threshold" in sig.parameters, \
        f"缺参数 lead_orientation_threshold (会 NameError). 当前签名: {sig}"
    # 默认值 0.10 必须保留 (CLAUDE.md 钉死阈值)
    assert sig.parameters["lead_orientation_threshold"].default == 0.10


def test_get_lead_bbox_front_orientation():
    """nose 居中 (offset < 0.03) → front, bbox 非空"""
    kp = _make_kp(nose=(0.5, 0.3, 0.9))
    bbox, ori = fs.get_lead_bbox_from_pose(kp, 1920, 1080, lead_orientation_threshold=0.10)
    assert ori == "front"
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x2 > x1 and y2 > y1
    assert x1 >= 0 and y1 >= 0 and x2 <= 1920 and y2 <= 1080


def test_get_lead_bbox_side_orientation():
    """鼻子偏移 0.05 (在 0.03~0.10 之间) → side, bbox 仍返回"""
    kp = _make_kp(nose=(0.55, 0.3, 0.9), ls=(0.45, 0.5, 0.9), rs=(0.55, 0.5, 0.9))
    bbox, ori = fs.get_lead_bbox_from_pose(kp, 1920, 1080, lead_orientation_threshold=0.10)
    assert ori == "side"
    assert bbox is not None


def test_get_lead_bbox_back_too_much_offset():
    """鼻子偏移 0.15 (>0.10) → back, bbox=None"""
    kp = _make_kp(nose=(0.65, 0.3, 0.9), ls=(0.45, 0.5, 0.9), rs=(0.55, 0.5, 0.9))
    bbox, ori = fs.get_lead_bbox_from_pose(kp, 1920, 1080, lead_orientation_threshold=0.10)
    assert ori == "back"
    assert bbox is None


def test_get_lead_bbox_back_low_nose_conf():
    """鼻子 conf < 0.3 → back (无论偏移)"""
    kp = _make_kp(nose=(0.5, 0.3, 0.1))
    bbox, ori = fs.get_lead_bbox_from_pose(kp, 1920, 1080, lead_orientation_threshold=0.10)
    assert ori == "back"
    assert bbox is None


def test_get_lead_bbox_no_shoulders_assumes_front():
    """肩膀不可见 → 默认 front, 不抛"""
    kp = _make_kp(ls=(0, 0, 0), rs=(0, 0, 0))
    bbox, ori = fs.get_lead_bbox_from_pose(kp, 1920, 1080, lead_orientation_threshold=0.10)
    assert ori == "front"
    assert bbox is not None


def test_find_lead_person_picks_center_largest():
    """find_lead_person: 选 cx 接近 0.5 + 身体最大的"""
    # A: 居中 + 大 (领操人)
    a = _make_kp(nose=(0.5, 0.3, 0.9), ls=(0.40, 0.5, 0.9), rs=(0.60, 0.5, 0.9))
    # B: 偏右 + 小 (远处观众)
    b = _make_kp(nose=(0.85, 0.3, 0.9), ls=(0.80, 0.5, 0.9), rs=(0.90, 0.5, 0.9))

    lead = fs.find_lead_person([a, b], 1920, 1080)
    assert lead is not None
    # 应选 A: cx 更居中, body 更宽
    assert lead[0][0] == pytest.approx(0.5, abs=0.01)


def test_find_lead_person_empty_returns_none():
    assert fs.find_lead_person([], 1920, 1080) is None
    assert fs.find_lead_person(None, 1920, 1080) is None


def test_process_video_signature_has_keypoint_params():
    """process_video 必须接受 keypoints_file + lead_orientation_threshold"""
    import inspect
    sig = inspect.signature(fs.process_video)
    assert "keypoints_file" in sig.parameters
    assert "lead_orientation_threshold" in sig.parameters
    assert sig.parameters["lead_orientation_threshold"].default == 0.10


def test_swap_face_signature_has_lead_bbox():
    """swap_face 必须接受 lead_bbox 参数 (pose-driven 模式)"""
    import inspect
    sig = inspect.signature(fs.swap_face)
    assert "lead_bbox" in sig.parameters
    # 默认 None 保留旧行为 (全图检测)
    assert sig.parameters["lead_bbox"].default is None
    # color_match_strength 默认 0.8 保留 (CLAUDE.md 钉死)
    assert sig.parameters["color_match_strength"].default == 0.8

"""lib/tracker.py 领操人识别 + 切换平滑的纯逻辑测试 (2026-07-02 债 C).

主管线 h2v_convert / 领操人跟踪的核心决策在 lib/tracker.py, 之前无单测.
这里测纯逻辑 (无需视频/GPU):
- identify_lead_person: 选身体面积(肩宽×身高×可见度)最大者
- LeadPersonSmoother: 连续 CONSECUTIVE(5) 帧才切换, 单帧抖动不跳

注: tracker.py 顶部 import filterpy, 故本文件用 importorskip 保护;
identify_lead_person / LeadPersonSmoother 本身只用 numpy."""
import numpy as np
import pytest

pytest.importorskip("filterpy")

from lib.tracker import identify_lead_person, LeadPersonSmoother


def _mp_person(nose, l_shoulder, r_shoulder, l_ankle, r_ankle):
    """合成 mediapipe-33 人 (identify_lead_person 读 idx 0/11/12/27/28).
    其余点 conf=1 坐标 0, 保证 visibility 高."""
    kps = np.zeros((33, 3))
    kps[:, 2] = 1.0
    kps[0, :2] = nose
    kps[11, :2] = l_shoulder
    kps[12, :2] = r_shoulder
    kps[27, :2] = l_ankle
    kps[28, :2] = r_ankle
    return kps


# ---------- identify_lead_person ----------

def test_identify_empty_returns_zero():
    assert identify_lead_person([], 100, 100) == 0


def test_identify_single_returns_zero():
    p = _mp_person([.5, .1], [.4, .2], [.6, .2], [.45, .95], [.55, .95])
    assert identify_lead_person([p], 100, 100) == 0


def test_identify_picks_largest_body():
    """两人时选肩宽×身高更大者 (离镜头最近)"""
    small = _mp_person([.5, .10], [.4, .20], [.6, .20], [.45, .95], [.55, .95])
    big = _mp_person([.5, .05], [.3, .15], [.7, .15], [.40, .98], [.60, .98])
    # small: shoulder_w=.2 body_h=.85; big: shoulder_w=.4 body_h=.93 → big
    assert identify_lead_person([small, big], 100, 100) == 1
    # 顺序无关
    assert identify_lead_person([big, small], 100, 100) == 0


def test_identify_skips_low_visibility():
    """可见点 <5 的人跳过 (visibility 过滤)"""
    good = _mp_person([.5, .10], [.4, .20], [.6, .20], [.45, .95], [.55, .95])
    # 构造一个 "看似更大" 但关键点几乎全不可见的人
    big_invis = _mp_person([.5, .05], [.3, .15], [.7, .15], [.40, .98], [.60, .98])
    big_invis[:, 2] = 0.1  # 全部 conf<0.5 → vis_mask.sum()==0 <5 → 跳过
    assert identify_lead_person([good, big_invis], 100, 100) == 0


# ---------- LeadPersonSmoother ----------

def test_smoother_needs_consecutive_frames():
    """连续 CONSECUTIVE(5) 帧同候选人才切换"""
    sm = LeadPersonSmoother()
    results = [sm.update(2) for _ in range(5)]
    assert results[:4] == [0, 0, 0, 0], "前 4 帧不应切换"
    assert results[4] == 2, "第 5 帧连续候选才切换到 2"


def test_smoother_ignores_single_frame_jitter():
    """稳态后单帧异样候选不切换"""
    sm = LeadPersonSmoother()
    for _ in range(5):
        sm.update(1)
    assert sm.update(1) == 1  # 稳态在 1
    assert sm.update(2) == 1  # 单帧 2: candidate=2 count=1, 不切
    assert sm.update(1) == 1  # 回 1: candidate 重置, 仍 1


def test_smoother_resets_candidate_on_flip():
    """候选切换会重置计数 (A B A B... 永远不达 5, 永不切)"""
    sm = LeadPersonSmoother()
    out = []
    for i in range(20):
        out.append(sm.update(3 if i % 2 == 0 else 4))
    assert all(o == 0 for o in out), "交替候选永不满 5 帧, 不应切换"

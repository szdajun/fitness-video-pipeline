"""高燃预览开场 (hook) compute_hook_window 守门测试 (2026-07-07).

不跑 ffmpeg, 纯算法层: 选全片最燃 hook_dur 秒窗, 排除首尾 10%,
单帧尖刺不污染 (滑动窗自身稀释), crop_x 钳制, skip_sec 正片相对映射.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from stages.short_vertical import compute_hook_window


def _make_person(cx=0.5, cy=0.5, scale=0.15, conf=0.95):
    """造 33 点假人 (归一化). 围绕 (cx, cy) 分布, 整体平移时所有点同 Δcx."""
    kps = [[cx, cy, conf]] * 33
    for i in range(7):                       # 头 0-6
        kps[i] = [cx + (i - 3) * 0.02, cy - scale * 0.6, conf]
    kps[11] = [cx - scale * 0.3, cy - scale * 0.3, conf]
    kps[12] = [cx + scale * 0.3, cy - scale * 0.3, conf]
    kps[13] = [cx - scale * 0.4, cy - scale * 0.1, conf]
    kps[14] = [cx + scale * 0.4, cy - scale * 0.1, conf]
    kps[15] = [cx - scale * 0.5, cy + scale * 0.1, conf]
    kps[16] = [cx + scale * 0.5, cy + scale * 0.1, conf]
    kps[23] = [cx - scale * 0.25, cy + scale * 0.2, conf]
    kps[24] = [cx + scale * 0.25, cy + scale * 0.2, conf]
    for i in range(25, 33):
        kps[i] = [cx + (0.1 if i % 2 else -0.1), cy + scale * 0.8, conf]
    return kps


def _kp_motion(n_frames, motion_by_frame, base_cx=0.5, fps=30, skip_sec=0.0):
    """造 kp_dict: motion_by_frame[wf] = 该 workout 帧的目标位移幅度.

    每帧 cx 围绕 base_cx 振荡 ±amp/2 → 相邻帧位移 ≈ amp (整体平移, 所有点同 Δcx).
    key 用源帧 (skip_sec*fps + wf), 模拟真实 keypoints JSON 的源帧索引空间.
    """
    src0 = int(round(skip_sec * fps))
    kp = {}
    sign = 1
    for wf in range(n_frames):
        src_f = src0 + wf
        amp = motion_by_frame.get(wf, 0.001)
        cx = base_cx + sign * amp / 2.0
        sign *= -1
        kp[src_f] = [_make_person(cx=cx)]
    return kp


class TestComputeHookWindow:
    def test_selects_high_activity_window(self):
        """中段持续高燃 → hook_start 落在高燃区."""
        fps, total = 10, 20
        n = fps * total
        motion = {wf: 0.04 for wf in range(n)}
        for wf in range(80, 120):          # 8-12s 持续高燃
            motion[wf] = 0.20
        kp = _kp_motion(n, motion, fps=fps)
        res = compute_hook_window(kp, [(0, 1000, 656)], fps=fps,
                                  total_dur=total, hook_dur=4.0)
        assert res is not None
        hook_start, _ = res
        assert 6.0 <= hook_start <= 12.0, f"hook_start={hook_start} 不在高燃区[8,12]"

    def test_single_frame_spike_not_selected(self):
        """持续高燃段 vs 别处单帧 spike → 选持续高燃, 不被 spike 带偏."""
        fps, total = 10, 20
        n = fps * total
        motion = {wf: 0.04 for wf in range(n)}
        for wf in range(80, 120):          # 持续高燃 8-12s
            motion[wf] = 0.20
        motion[150] = 0.60                 # 单帧 spike @ 15s (候选区内)
        kp = _kp_motion(n, motion, fps=fps)
        res = compute_hook_window(kp, [(0, 1000, 656)], fps=fps,
                                  total_dur=total, hook_dur=4.0)
        hook_start, _ = res
        assert 6.0 <= hook_start <= 12.0, f"被 spike 带偏: hook_start={hook_start}"

    def test_excludes_head_tail(self):
        """首尾各 10% 高燃 → 排除, hook_start 落中间."""
        fps, total = 10, 20
        n = fps * total
        motion = {wf: 0.04 for wf in range(n)}
        for wf in range(0, 20):            # 首 2s 高燃 (应排除)
            motion[wf] = 0.30
        for wf in range(180, 200):         # 尾 2s 高燃 (应排除)
            motion[wf] = 0.30
        kp = _kp_motion(n, motion, fps=fps)
        res = compute_hook_window(kp, [(0, 1000, 656)], fps=fps,
                                  total_dur=total, hook_dur=4.0)
        hook_start, _ = res
        # 排除首 10%(2s) 尾 10%(18s): hook_start ∈ [2, 18-4=14]
        assert 2.0 <= hook_start <= 14.0, f"hook_start={hook_start} 未排除首尾"

    def test_crop_x_clamp(self):
        """crop_x 超 [padding, frame_w-crop_w-padding] → 钳制."""
        fps, total = 10, 20
        n = fps * total
        motion = {wf: 0.04 for wf in range(n)}
        for wf in range(80, 120):
            motion[wf] = 0.20
        kp = _kp_motion(n, motion, fps=fps)
        # crop_x=5000 超 1920-608-30=1282 → 钳到 1282
        res = compute_hook_window(kp, [(0, 1000, 5000)], fps=fps,
                                  total_dur=total, hook_dur=4.0)
        _, hook_cx = res
        assert hook_cx == 1282, f"应钳到 1282, got {hook_cx}"

    def test_multi_segment_crop_x(self):
        """多段 crop, hook 落第二段 → hook_crop_x = 第二段 crop_x."""
        fps, total = 10, 20
        n = fps * total
        motion = {wf: 0.04 for wf in range(n)}
        for wf in range(120, 160):         # 12-16s 高燃 → 源帧 120-160 落第二段
            motion[wf] = 0.20
        kp = _kp_motion(n, motion, fps=fps)
        segs = [(0, 100, 400), (100, 200, 1100)]
        res = compute_hook_window(kp, segs, fps=fps, total_dur=total, hook_dur=4.0)
        _, hook_cx = res
        assert hook_cx == 1100, f"应取第二段 crop_x=1100, got {hook_cx}"

    def test_skip_sec_mapping(self):
        """skip_sec 偏移: workout 帧 wf → 源帧 skip_sec*fps+wf, 仍选正片中段高燃."""
        fps, total = 10, 20
        n = fps * total
        motion = {wf: 0.04 for wf in range(n)}
        for wf in range(80, 120):
            motion[wf] = 0.20
        skip = 4.0
        kp = _kp_motion(n, motion, fps=fps, skip_sec=skip)
        # kp 的 key 是源帧 (40+wf), 模拟 keypoints JSON 源帧索引空间
        res = compute_hook_window(kp, [(0, 1000, 656)], fps=fps,
                                  total_dur=total, hook_dur=4.0, skip_sec=skip)
        assert res is not None
        hook_start, _ = res
        assert 6.0 <= hook_start <= 12.0, f"skip 映射错: hook_start={hook_start}"

    def test_frame_aligned(self):
        """hook_start 落在整数帧 (hook_start * fps 是整数)."""
        fps, total = 30, 20
        n = fps * total
        motion = {wf: 0.04 for wf in range(n)}
        for wf in range(300, 420):         # 10-14s 高燃
            motion[wf] = 0.20
        kp = _kp_motion(n, motion, fps=fps)
        res = compute_hook_window(kp, [(0, 1000, 656)], fps=fps,
                                  total_dur=total, hook_dur=4.0)
        hook_start, _ = res
        # hook_start = best_sf/fps round 到 3 位小数 (ffmpeg -ss 精度), 允许 round 误差 (≤0.015 帧)
        assert abs(hook_start * fps - round(hook_start * fps)) < 0.05

    def test_empty_kp_returns_none(self):
        assert compute_hook_window({}, [(0, 100, 656)], fps=10,
                                   total_dur=20, hook_dur=4) is None

    def test_short_total_returns_none(self):
        # total_dur=5 < min_total_dur=10
        assert compute_hook_window({0: [_make_person()]}, [(0, 100, 656)],
                                   fps=10, total_dur=5, hook_dur=4) is None

    def test_hook_dur_too_big_returns_none(self):
        # usable = 20*0.8=16, hook_dur=17 >= 16 → None
        assert compute_hook_window({0: [_make_person()]}, [(0, 100, 656)],
                                   fps=10, total_dur=20, hook_dur=17) is None

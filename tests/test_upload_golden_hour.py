"""Shorts 自动黄金时段发布守门 (2026-07-12 用户拍板).

数据依据: 用户频道 195 视频分析
- 黄金时段 10-14 / 19-23 北京时间
- 5 月黄金期均 view 1376 (13-14) / 935 (22-23) / 862 (19-20)
- 避开 8-10 (98 view), 16-17 (199 view 样本偏少)

历史教训 (per memory yt-long-video-publish-immediately):
- 长视频 scheduled (publishAt) 在 YT 平台挂死, 必须立即发
- 短片走客户端 sleep + 立即发, 不用 publishAt, 绕开 bug

不重新跑真 upload, 纯算法层: _is_golden_hour + seconds_until_next_golden.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lib.upload_utils import (
    _is_golden_hour,
    seconds_until_next_golden,
)


def at(year, month, day, hour, minute=0):
    """构造北京时间 (UTC+8)."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone(timedelta(hours=8)))


class TestIsGoldenHour:
    def test_in_golden_window_10_to_14(self):
        assert _is_golden_hour(at(2026, 7, 12, 10, 0)) is True
        assert _is_golden_hour(at(2026, 7, 12, 11, 30)) is True
        assert _is_golden_hour(at(2026, 7, 12, 13, 59)) is True

    def test_in_golden_window_19_to_23(self):
        assert _is_golden_hour(at(2026, 7, 12, 19, 0)) is True
        assert _is_golden_hour(at(2026, 7, 12, 20, 30)) is True
        assert _is_golden_hour(at(2026, 7, 12, 22, 59)) is True

    def test_outside_golden_window(self):
        # 早晨
        assert _is_golden_hour(at(2026, 7, 12, 8, 0)) is False
        assert _is_golden_hour(at(2026, 7, 12, 9, 59)) is False
        # 下午低谷
        assert _is_golden_hour(at(2026, 7, 12, 14, 0)) is False
        assert _is_golden_hour(at(2026, 7, 12, 15, 30)) is False
        assert _is_golden_hour(at(2026, 7, 12, 16, 0)) is False
        assert _is_golden_hour(at(2026, 7, 12, 17, 30)) is False
        assert _is_golden_hour(at(2026, 7, 12, 18, 59)) is False
        # 深夜
        assert _is_golden_hour(at(2026, 7, 12, 23, 0)) is False
        assert _is_golden_hour(at(2026, 7, 12, 0, 30)) is False
        assert _is_golden_hour(at(2026, 7, 12, 5, 0)) is False

    def test_boundary_excluded(self):
        # 边界: 14:00 不在 (10-14 半开), 23:00 不在 (19-23 半开)
        assert _is_golden_hour(at(2026, 7, 12, 14, 0)) is False
        assert _is_golden_hour(at(2026, 7, 12, 23, 0)) is False
        # 19:00 在, 18:59 不在
        assert _is_golden_hour(at(2026, 7, 12, 19, 0)) is True
        assert _is_golden_hour(at(2026, 7, 12, 18, 59)) is False


class TestSecondsUntilNextGolden:
    def test_in_golden_returns_zero(self):
        # 黄金时段内 → 0
        assert seconds_until_next_golden(at(2026, 7, 12, 13, 30)) == 0
        assert seconds_until_next_golden(at(2026, 7, 12, 22, 0)) == 0

    def test_morning_before_10_waits_to_10(self):
        # 8:00 → 等 10:00 (2 小时 = 7200 秒)
        secs = seconds_until_next_golden(at(2026, 7, 12, 8, 0))
        assert secs == 2 * 3600
        # 9:30 → 等 30 分钟
        secs = seconds_until_next_golden(at(2026, 7, 12, 9, 30))
        assert secs == 30 * 60

    def test_afternoon_14_to_18_waits_to_19(self):
        # 14:00 → 等 19:00 (5 小时 = 18000 秒)
        secs = seconds_until_next_golden(at(2026, 7, 12, 14, 0))
        assert secs == 5 * 3600
        # 15:30 → 等 3.5 小时
        secs = seconds_until_next_golden(at(2026, 7, 12, 15, 30))
        assert secs == 3 * 3600 + 30 * 60
        # 18:59 → 等 1 分钟
        secs = seconds_until_next_golden(at(2026, 7, 12, 18, 59))
        assert secs == 60

    def test_evening_after_23_waits_to_next_day_10(self):
        # 23:00 → 等明天 10:00 (11 小时 = 39600 秒)
        secs = seconds_until_next_golden(at(2026, 7, 12, 23, 0))
        assert secs == 11 * 3600
        # 0:30 → 等明天 9.5 小时
        secs = seconds_until_next_golden(at(2026, 7, 12, 0, 30))
        assert secs == 9 * 3600 + 30 * 60
        # 5:00 → 等明天 5 小时
        secs = seconds_until_next_golden(at(2026, 7, 12, 5, 0))
        assert secs == 5 * 3600


class TestIntegration:
    """守门: 主管线 Shorts 默认走黄金时段等待."""

    def test_upload_pair_default_wait_for_short_golden_hour(self):
        """upload_pair 默认 wait_for_short_golden_hour=True (2026-07-12 用户拍板)."""
        import inspect
        sig = inspect.signature(__import__('lib.upload_utils', fromlist=['upload_pair']).upload_pair)
        assert sig.parameters['wait_for_short_golden_hour'].default is True

    def test_no_publishat_used_in_short_path(self):
        """守门: short 上传路径不传 publish_at (避开 YT 长视频挂死 bug)."""
        src = Path("lib/upload_utils.py").read_text(encoding="utf-8")
        # 找到 short 上传这段 — 调用 upload_video 短片分支
        # 关键: 该段不能传 publish_at=...
        short_block = src.split("if short_path and Path(short_path).exists():")[1].split("results[\"short\"]")[0]
        assert "publish_at" not in short_block, (
            "short 上传分支不能传 publish_at (per memory yt-long-video-publish-immediately, "
            "scheduled 视频在 YT 平台挂死)"
        )

    def test_long_path_also_publishes_immediately(self):
        """守门: long 上传也不传 publish_at (per CLAUDE 钉死规则)."""
        src = Path("lib/upload_utils.py").read_text(encoding="utf-8")
        # long 上传分支
        long_block = src.split("if long_path and Path(long_path).exists():")[1].split("results[\"long\"]")[0]
        assert "publish_at" not in long_block, (
            "long 上传分支不能传 publish_at (per CLAUDE 钉死, scheduled 长视频挂死)"
        )

    def test_golden_hour_does_not_use_publishat(self):
        """守门: 黄金时段函数 wait_for_golden_hour 用 sleep 不用 publishAt."""
        src = Path("lib/upload_utils.py").read_text(encoding="utf-8")
        # wait_for_golden_hour 函数体内不能引用 publish_at
        func_block = src.split("def wait_for_golden_hour(")[1].split("def ")[0]
        assert "publish_at" not in func_block, (
            "wait_for_golden_hour 应该用客户端 sleep 而非 publishAt, "
            "绕开 YT 长视频 scheduled 挂死 bug"
        )
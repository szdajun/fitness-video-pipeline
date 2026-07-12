"""Tests for golden-hour boundary + ANSI helper (sanity checks)."""
import sys
from datetime import datetime
from pathlib import Path

import pytest
from lib.upload_utils import _is_golden_hour

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _fake_dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 13, hour, minute, 0)


def test_golden_14_00_false():
    assert _is_golden_hour(_fake_dt(14, 0)) is False


def test_golden_14_59_false():
    assert _is_golden_hour(_fake_dt(14, 59)) is False


def test_golden_15_00_false():
    assert _is_golden_hour(_fake_dt(15, 0)) is False


def test_golden_13_59_true():
    assert _is_golden_hour(_fake_dt(13, 59)) is True


def test_golden_22_30_true():
    assert _is_golden_hour(_fake_dt(22, 30)) is True


def test_golden_8_30_false():
    assert _is_golden_hour(_fake_dt(8, 30)) is False


def test_golden_16_30_false():
    assert _is_golden_hour(_fake_dt(16, 30)) is False


def test_ansi_yellow_wraps_text():
    """_c('yellow', 'X') 总是包含 'X', ANSI 启用时含 \033."""
    from tools.upload_reminder import _c, _ANSI_ENABLED
    out = _c("yellow", "X")
    assert "X" in out
    if _ANSI_ENABLED:
        assert "\033" in out

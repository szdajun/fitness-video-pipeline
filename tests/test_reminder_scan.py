"""Tests for tools.upload_reminder.scan_pending_videos — scan output/, filter, enrich."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json
import os
import time
import pytest
from tools.upload_reminder import scan_pending_videos, _extract_coach


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 1024)


def test_extract_coach_splits_on_underscore():
    assert _extract_coach("艳青1_2_merged") == "艳青"


def test_extract_coach_falls_back_for_plain_stem():
    assert _extract_coach("小飞侠") == "小飞侠"


def test_scan_finds_all_three_kinds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    date_dir = tmp_path / "output" / "2026-07-13"
    _touch(date_dir / "艳青1_2_merged_full_16x9_1920x1080.mp4")
    _touch(date_dir / "艳青1_2_merged_full_16x9_1920x1080_yt_shorts.mp4")
    _touch(date_dir / "艳青1_2_merged_full_16x9_1920x1080_douyin.mp4")
    result = scan_pending_videos(output_dir="output", state={})
    assert len(result) == 3
    kinds = {r["kind"] for r in result}
    assert kinds == {"long", "short", "douyin"}


def test_scan_skips_intermediate_products(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    date_dir = tmp_path / "output" / "2026-07-13"
    _touch(date_dir / "艳青1_2_merged_full_16x9_1920x1080.mp4")
    _touch(date_dir / "艳青1_2_merged_color.mp4")
    _touch(date_dir / "艳青1_2_merged_energybar_watermark.mp4")
    _touch(date_dir / "艳青1_2_merged_faceswap_burst_danmaku.mp4")
    _touch(date_dir / "艳青1_2_merged_intro.mp4")
    _touch(date_dir / "艳青1_2_merged_outro.mp4")
    result = scan_pending_videos(output_dir="output", state={})
    assert len(result) == 1
    assert result[0]["kind"] == "long"


def test_scan_skips_uploaded_and_archived(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    date_dir = tmp_path / "output" / "2026-07-13"
    long_p = date_dir / "艳青1_2_merged_full_16x9_1920x1080.mp4"
    short_p = date_dir / "艳青1_2_merged_full_16x9_1920x1080_yt_shorts.mp4"
    douyin_p = date_dir / "艳青1_2_merged_full_16x9_1920x1080_douyin.mp4"
    _touch(long_p)
    _touch(short_p)
    _touch(douyin_p)
    state = {
        str(long_p.resolve()): {"marked_uploaded_at": "2026-07-13T10:00", "remind_count": 0, "last_reminded_at": None},
        str(douyin_p.resolve()): {"remind_count": 3, "auto_archived": True, "last_reminded_at": "2026-07-13T09:00", "marked_uploaded_at": None},
    }
    result = scan_pending_videos(output_dir="output", state=state)
    paths = [r["path"] for r in result]
    assert len(result) == 1
    assert str(short_p.resolve()) in paths


def test_scan_skips_full_16x9_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    date_dir = tmp_path / "output" / "2026-07-13"
    _touch(date_dir / "艳青1_2_merged_full_16x9_1920x1080.mp4")
    _touch(date_dir / "艳青1_2_merged_full_16x9_1920x1080_full_16x9.mp4")
    result = scan_pending_videos(output_dir="output", state={})
    assert len(result) == 1
"""Tests for lib.reminder_state — log.json read/write/mutate."""
import json
import pytest
from lib.reminder_state import (
    LOG_PATH, load, save, mark_uploaded, increment_remind,
    reset_remind, mark_all_uploaded,
)


def _isolated_log(tmp_path, monkeypatch):
    """Redirect LOG_PATH to tmp_path for test isolation."""
    test_log = tmp_path / "upload_reminder_log.json"
    monkeypatch.setattr("lib.reminder_state.LOG_PATH", str(test_log))
    return test_log


def test_load_missing_file_returns_empty(tmp_path, monkeypatch):
    _isolated_log(tmp_path, monkeypatch)
    assert load() == {}


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    _isolated_log(tmp_path, monkeypatch)
    state = {
        r"F:\fake\video1.mp4": {
            "marked_uploaded_at": "2026-07-13T10:00:00",
            "remind_count": 1,
            "last_reminded_at": "2026-07-13T10:00:00",
        }
    }
    save(state)
    assert load() == state


def test_mark_uploaded_sets_timestamp_and_resets_count(tmp_path, monkeypatch):
    _isolated_log(tmp_path, monkeypatch)
    state = {r"F:\fake\video1.mp4": {"marked_uploaded_at": None, "remind_count": 2, "last_reminded_at": "2026-07-12"}}
    new = mark_uploaded(state, r"F:\fake\video1.mp4")
    assert new[r"F:\fake\video1.mp4"]["marked_uploaded_at"] is not None
    assert new[r"F:\fake\video1.mp4"]["remind_count"] == 0


def test_increment_remind_accumulates_and_archives_at_3(tmp_path, monkeypatch):
    _isolated_log(tmp_path, monkeypatch)
    state = {}
    state = increment_remind(state, r"F:\fake\video1.mp4")
    state = increment_remind(state, r"F:\fake\video1.mp4")
    assert state[r"F:\fake\video1.mp4"]["remind_count"] == 2
    assert "auto_archived" not in state[r"F:\fake\video1.mp4"]
    state = increment_remind(state, r"F:\fake\video1.mp4")
    assert state[r"F:\fake\video1.mp4"]["remind_count"] == 3
    assert state[r"F:\fake\video1.mp4"]["auto_archived"] is True


def test_reset_remind_clears_archive(tmp_path, monkeypatch):
    _isolated_log(tmp_path, monkeypatch)
    state = {r"F:\fake\video1.mp4": {"remind_count": 3, "auto_archived": True}}
    new = reset_remind(state, r"F:\fake\video1.mp4")
    assert new[r"F:\fake\video1.mp4"]["remind_count"] == 0
    assert "auto_archived" not in new[r"F:\fake\video1.mp4"]


def test_mark_all_uploaded_marks_each(tmp_path, monkeypatch):
    _isolated_log(tmp_path, monkeypatch)
    state = {}
    new = mark_all_uploaded(state, [r"F:\fake\v1.mp4", r"F:\fake\v2.mp4"])
    assert new[r"F:\fake\v1.mp4"]["marked_uploaded_at"] is not None
    assert new[r"F:\fake\v2.mp4"]["marked_uploaded_at"] is not None


def test_corrupt_log_creates_backup_and_returns_empty(tmp_path, monkeypatch):
    test_log = _isolated_log(tmp_path, monkeypatch)
    test_log.write_text("{this is not valid json", encoding="utf-8")
    assert load() == {}
    assert (tmp_path / "upload_reminder_log.json.bak").exists()

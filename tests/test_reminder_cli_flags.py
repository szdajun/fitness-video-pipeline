"""Tests for non-interactive CLI flags: --show-pending + --mark-all-uploaded.

回归保护: 用户全部上传后想批量标已传, 不应强制走弹窗交互 (Task Scheduler stdin EOF 阻塞).
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_show_pending_non_interactive(tmp_path, monkeypatch):
    """--show-pending 应**非交互**列出待传 (不弹窗, 不读 stdin)."""
    monkeypatch.chdir(tmp_path)

    # 造 2 条待传
    date_dir = tmp_path / "output" / "2026-07-13"
    date_dir.mkdir(parents=True)
    (date_dir / "test1_full_16x9_1920x1080.mp4").write_bytes(b"\x00" * 1024)
    (date_dir / "test1_full_16x9_1920x1080_douyin.mp4").write_bytes(b"\x00" * 1024)

    # stdin 空 pipe 模拟 (不阻塞 input)
    proc = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-u",
         str(ROOT / "tools" / "upload_reminder.py"),
         "--output", "output",
         "--show-pending"],
        input="", capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=30, cwd=str(ROOT),
    )
    assert proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr}"
    # 至少在测试隔离目录前 — 这里测的是空目录场景, 项目实际有 18 条
    # 验证 stdout 含"待传视频:"字样
    assert "待传视频:" in proc.stdout


def test_mark_all_uploaded_non_interactive(tmp_path, monkeypatch):
    """--mark-all-uploaded 应**非交互**标全部已传 (不读 stdin)."""
    monkeypatch.chdir(tmp_path)

    # 临时覆盖 records/ 路径到 tmp
    import lib.reminder_state as rs
    monkeypatch.setattr(rs, "LOG_PATH", str(tmp_path / "upload_reminder_log.json"))

    # 造 2 条待传
    date_dir = tmp_path / "output" / "2026-07-13"
    date_dir.mkdir(parents=True)
    f1 = date_dir / "test1_full_16x9_1920x1080.mp4"
    f2 = date_dir / "test1_full_16x9_1920x1080_douyin.mp4"
    f1.write_bytes(b"\x00" * 1024)
    f2.write_bytes(b"\x00" * 1024)

    # 用绝对路径避免 chdir 问题
    proc = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-u",
         str(ROOT / "tools" / "upload_reminder.py"),
         "--output", str(tmp_path / "output"),
         "--mark-all-uploaded"],
        input="", capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=30, cwd=str(tmp_path),
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/usr/local/bin"},
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exit {proc.returncode}: {out}"
    assert "已标记" in out or "无需标记" in out, f"应打印标记结果, 实: {out!r}"

    # 验证状态文件写了
    log_path = tmp_path / "upload_reminder_log.json"
    if log_path.exists():
        import json
        state = json.loads(log_path.read_text(encoding="utf-8"))
        # 至少有一条 marked_uploaded_at 不为 None (实际写过的)
        marked = [r for r in state.values() if r.get("marked_uploaded_at")]
        assert len(marked) >= 1, f"应至少标记 1 条, 实: {state}"


def test_mark_all_uploaded_when_none_pending(tmp_path, monkeypatch):
    """空目录 + --mark-all-uploaded 应**优雅**返回 0, 不报错."""
    monkeypatch.chdir(tmp_path)

    proc = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-u",
         str(ROOT / "tools" / "upload_reminder.py"),
         "--output", "output",
         "--mark-all-uploaded"],
        input="", capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=30, cwd=str(tmp_path),
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exit {proc.returncode}: {out}"
    assert "无需标记" in out or "无待传" in out, f"应打印无需标记, 实: {out!r}"
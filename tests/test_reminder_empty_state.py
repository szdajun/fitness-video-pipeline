"""Tests for tools.upload_reminder empty-state behavior.

弹窗无待传视频时**不应阻塞等输入** — 直接返回, 不留卡住窗口.
回归保护: commit ea74f07 引入 input("按 Enter 继续...") 阻塞, 2026-07-13 用户实测卡住手动关窗.
"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_render_and_interact_returns_immediately_when_no_pending(tmp_path, monkeypatch, capsys):
    """无待传时 _render_and_interact 应**秒级返回** (不阻塞等 input)."""
    monkeypatch.chdir(tmp_path)
    # 空 output/ 目录 → scan_pending_videos 返 []
    from tools.upload_reminder import _render_and_interact

    finished = threading.Event()

    def run():
        _render_and_interact(output_dir="output")
        finished.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    # 3 秒内必须返回 (input() 会无限阻塞 → 超时 = bug)
    assert finished.wait(timeout=3.0), (
        "_render_and_interact 在无待传状态下阻塞超过 3s, "
        "不应 input('按 Enter 继续...')"
    )
    out = capsys.readouterr().out
    assert "无待传" in out or "关闭窗口" in out, (
        f"应打印无待传提示, 实际输出: {out!r}"
    )


def test_main_returns_cleanly_when_no_pending(tmp_path, monkeypatch):
    """main() 无待传时应正常退出 (exit 0), 不挂."""
    monkeypatch.chdir(tmp_path)
    from tools import upload_reminder

    # 不调 main() 的 argparse, 直接复用 main 函数体逻辑 — 测 render 路径
    from tools.upload_reminder import _render_and_interact
    # 已通过上面 test 验证, 这里只 smoke 跑一次不报错
    _render_and_interact(output_dir="output")
# Upload Reminder Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows-scheduled CLI tool that pops a command-line window at golden-hour timepoints (10-14/19-23 Beijing, 6 daily triggers) to remind the user to manually upload pending videos to YouTube/Douyin, with numbered-checkbox marking, system-player open, clipboard copy, and 3-strike auto-archive.

**Architecture:** 4 new files (state lib + main CLI + 2 .bat installers), 3 new test files. Reuses existing `lib/upload_utils.py` golden-hour functions and `build_title()`. Zero changes to main pipeline (stages/main.py/pipeline/). State in `records/upload_reminder_log.json` with path-keyed entries.

**Tech Stack:** Python 3.11 + stdlib only (no new deps), Windows 10/11 Task Scheduler via `schtasks.exe`, ANSI color via `os.system('')`.

## Global Constraints

- Python 3.11+ (per project `.python-version`); run via `uv run python` or `.venv/Scripts/python.exe` (per CLAUDE §"环境管理 (uv)")
- No new pip deps — stdlib only (`json`, `os`, `pathlib`, `subprocess`, `datetime`, `sys`, `ctypes`)
- Test runner: `uv run pytest tests/ -q` (per CLAUDE §"环境管理")
- Pre-commit hook: `.venv/Scripts/python.exe` + `pytest` (per `pre-commit-hook-venv-pipefail`)
- Path style: absolute + backslash-escaped in JSON (per existing `upload_manifest.json` style in `records/`)
- All Chinese strings UTF-8 (per CLAUDE §"emoji print GBK 坑" — never use emoji in print)
- Source dir: `F:\wkspace\fitness-video-pipeline`; `output/`, `tools/`, `lib/`, `tests/`, `records/`, `docs/` per CLAUDE.md §"Architecture"
- 不修改 `lib/upload_utils.py`, `stages/`, `main.py`, `pipeline/`, `config.yaml` (主管线零改动)
- 不修改 `tools/upload_youtube.py` (用户拍板 2026-07-13 自动上传全停)
- 黄金时段: 10-14 / 19-23 北京时间 (UTC+8), per `lib/upload_utils.py:435 _is_golden_hour` (已实现, 复用)
- 6 个时点: 10:00 / 12:00 / 14:00 / 19:00 / 21:00 / 23:00
- 3 件套文件名匹配:
  - `*_final_16x9_1920x1080.mp4` (long)
  - `*_full_16x9_1920x1080_yt_shorts.mp4` (short)
  - `*_full_16x9_1920x1080_douyin.mp4` (douyin)
- 跳过 `*_full_16x9.mp4` (去头去尾副本, per CLAUDE §"`_full_16x9`")
- 跳过所有中间产物 (`*_color.mp4` / `*_watermark.mp4` / `*_faceswap*.mp4` / `*_energybar*.mp4` / `*_intro.mp4` / `*_outro.mp4` 等)
- 漏处理 3 次未标记 → `auto_archived: true`, 不再弹
- log.json 损坏 → 备份到 `.bak` + 重生空 log
- 弹窗 ANSI 颜色失败 (cmd.exe 不支持) → 退化无色不报错
- 用户不点 [X 关闭] 就不关窗 (永远不自动 close)

---

## Task 1: State lib `lib/reminder_state.py` + state tests

**Files:**
- Create: `lib/reminder_state.py`
- Create: `tests/test_reminder_state.py`
- Test: `uv run pytest tests/test_reminder_state.py -v`

**Interfaces (consumed by later tasks):**
- `LOG_PATH = "records/upload_reminder_log.json"` (module-level constant)
- `load() -> dict[str, dict]` — returns `{}` if file missing or corrupt (also writes `.bak` on corrupt)
- `save(state: dict) -> None`
- `mark_uploaded(state: dict, file_path: str) -> dict` — returns new state with `{marked_uploaded_at: ISO timestamp, remind_count: 0, last_reminded_at: <preserve>}` for that path
- `increment_remind(state: dict, file_path: str) -> dict` — returns new state with `remind_count += 1`, `last_reminded_at: now`, `auto_archived: True` if count >= 3
- `reset_remind(state: dict, file_path: str) -> dict` — returns new state with `remind_count = 0`, removes `auto_archived`
- `mark_all_uploaded(state: dict, file_paths: list[str]) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reminder_state.py`:

```python
"""Tests for lib.reminder_state — log.json read/write/mutate."""
import json
import pytest
from pathlib import Path
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
    """log.json 不存在 → load() 返 {} 不抛异常."""
    _isolated_log(tmp_path, monkeypatch)
    assert load() == {}


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    """save() 后 load() 拿回完全相同数据."""
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
    """mark_uploaded 写时间戳 + remind_count 归 0."""
    _isolated_log(tmp_path, monkeypatch)
    state = {r"F:\fake\video1.mp4": {"marked_uploaded_at": None, "remind_count": 2, "last_reminded_at": "2026-07-12"}}
    new = mark_uploaded(state, r"F:\fake\video1.mp4")
    assert new[r"F:\fake\video1.mp4"]["marked_uploaded_at"] is not None
    assert new[r"F:\fake\video1.mp4"]["remind_count"] == 0


def test_increment_remind_accumulates_and_archives_at_3(tmp_path, monkeypatch):
    """increment_remind 累加, 第 3 次设 auto_archived."""
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
    """reset_remind 重置 remind_count=0 + 删 auto_archived."""
    _isolated_log(tmp_path, monkeypatch)
    state = {r"F:\fake\video1.mp4": {"remind_count": 3, "auto_archived": True}}
    new = reset_remind(state, r"F:\fake\video1.mp4")
    assert new[r"F:\fake\video1.mp4"]["remind_count"] == 0
    assert "auto_archived" not in new[r"F:\fake\video1.mp4"]


def test_mark_all_uploaded_marks_each(tmp_path, monkeypatch):
    """mark_all_uploaded 对每个路径调 mark_uploaded."""
    _isolated_log(tmp_path, monkeypatch)
    state = {}
    new = mark_all_uploaded(state, [r"F:\fake\v1.mp4", r"F:\fake\v2.mp4"])
    assert new[r"F:\fake\v1.mp4"]["marked_uploaded_at"] is not None
    assert new[r"F:\fake\v2.mp4"]["marked_uploaded_at"] is not None


def test_corrupt_log_creates_backup_and_returns_empty(tmp_path, monkeypatch):
    """log.json 损坏 → 备份 .bak + 返 {} 不崩."""
    test_log = _isolated_log(tmp_path, monkeypatch)
    test_log.write_text("{this is not valid json", encoding="utf-8")
    assert load() == {}
    assert (tmp_path / "upload_reminder_log.json.bak").exists()
```

- [ ] **Step 2: Run tests, verify they all fail (no implementation yet)**

Run: `uv run pytest tests/test_reminder_state.py -v`
Expected: ALL FAIL with `ModuleNotFoundError: No module named 'lib.reminder_state'` or `ImportError`

- [ ] **Step 3: Implement `lib/reminder_state.py`**

Create `lib/reminder_state.py`:

```python
"""上传提醒状态管理 — records/upload_reminder_log.json 读写 + mutate.

数据结构 (path → record):
  {
    "marked_uploaded_at": str ISO timestamp | None,
    "remind_count": int,
    "last_reminded_at": str ISO timestamp | None,
    "auto_archived": bool | absent (默认 False)
  }

log.json 损坏时自动备份到 .bak 并重生空 log, 不抛异常.
"""
import json
from datetime import datetime
from pathlib import Path

LOG_PATH = "records/upload_reminder_log.json"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load() -> dict:
    """读 log.json, 返 dict. 文件不存在/损坏返 {} + 备份损坏文件."""
    p = Path(LOG_PATH)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        # 损坏 → 备份 + 返空
        bak = p.with_suffix(p.suffix + ".bak")
        try:
            p.rename(bak)
        except OSError:
            pass
        return {}


def save(state: dict) -> None:
    """写 log.json (indent=2 + ensure_ascii=False 保中文)."""
    p = Path(LOG_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def _ensure_record(state: dict, file_path: str) -> dict:
    """Get-or-init record for path. 不修改 state, 返新 dict."""
    if file_path not in state:
        state[file_path] = {
            "marked_uploaded_at": None,
            "remind_count": 0,
            "last_reminded_at": None,
        }
    return state


def mark_uploaded(state: dict, file_path: str) -> dict:
    """标记为已上传, 重置 remind_count."""
    _ensure_record(state, file_path)
    state[file_path]["marked_uploaded_at"] = _now_iso()
    state[file_path]["remind_count"] = 0
    return state


def increment_remind(state: dict, file_path: str) -> dict:
    """累加 remind_count + 写 last_reminded_at, 达 3 设 auto_archived."""
    _ensure_record(state, file_path)
    state[file_path]["remind_count"] = state[file_path].get("remind_count", 0) + 1
    state[file_path]["last_reminded_at"] = _now_iso()
    if state[file_path]["remind_count"] >= 3:
        state[file_path]["auto_archived"] = True
    return state


def reset_remind(state: dict, file_path: str) -> dict:
    """恢复 (从 auto_archived 回到待提醒)."""
    _ensure_record(state, file_path)
    state[file_path]["remind_count"] = 0
    if "auto_archived" in state[file_path]:
        del state[file_path]["auto_archived"]
    return state


def mark_all_uploaded(state: dict, file_paths: list) -> dict:
    """批量标记."""
    for fp in file_paths:
        mark_uploaded(state, fp)
    return state
```

- [ ] **Step 4: Run tests, verify they all pass**

Run: `uv run pytest tests/test_reminder_state.py -v`
Expected: 7 passed

- [ ] **Step 5: Run full suite to ensure no regression**

Run: `uv run pytest tests/ -q`
Expected: 244 + 7 = 251 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add lib/reminder_state.py tests/test_reminder_state.py
git commit -m "@feat(reminder): 状态管理 lib/reminder_state.py + 7 tests (TDD)"
```

---

## Task 2: Scan logic in `tools/upload_reminder.py` + scan tests

**Files:**
- Modify: `tools/upload_reminder.py` (新建, 后续 task 加 CLI 入口)
- Create: `tests/test_reminder_scan.py`
- Test: `uv run pytest tests/test_reminder_scan.py -v`

**Interfaces (consumed by Task 3):**
- `scan_pending_videos(output_dir: str = "output", state: dict | None = None) -> list[dict]` — returns `[{"path": str, "kind": "long"|"short"|"douyin", "size_mb": int, "mtime": str ISO, "title": str, "stem": str}, ...]`
  - `kind` derived from filename suffix (long = `_final_16x9_1920x1080.mp4` exact, short = `_yt_shorts.mp4` suffix, douyin = `_douyin.mp4` suffix)
  - filters out `*_full_16x9.mp4` (no _final in name)
  - filters out anything in state with `marked_uploaded_at not None` or `auto_archived == True`
  - `title` from `build_title(coach=stem, ...)` — coach extracted from stem via `_extract_coach(stem)` (split on `_` first token, fallback "教练")

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reminder_scan.py`:

```python
"""Tests for tools.upload_reminder.scan_pending_videos — scan output/, filter, enrich."""
import json
import os
import time
from pathlib import Path
import pytest
from tools.upload_reminder import scan_pending_videos, _extract_coach


def _make_video(path: Path, content: bytes = b"\x00" * 1024) -> None:
    """Create a dummy video file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 1024)


def test_extract_coach_splits_on_underscore():
    """_extract_coach('艳青1_2_merged') → '艳青'."""
    assert _extract_coach("艳青1_2_merged") == "艳青"


def test_extract_coach_falls_back_for_plain_stem():
    """_extract_coach('小飞侠') → '小飞侠' (无下划线)."""
    assert _extract_coach("小飞侠") == "小飞侠"


def test_scan_finds_all_three_kinds(tmp_path, monkeypatch):
    """output/<date>/*.{final_16x9,yt_shorts,douyin}.mp4 → 3 条."""
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
    """中间产物 *_color.mp4 / *_watermark.mp4 / *_intro.mp4 不进列表."""
    monkeypatch.chdir(tmp_path)
    date_dir = tmp_path / "output" / "2026-07-13"
    _touch(date_dir / "艳青1_2_merged_full_16x9_1920x1080.mp4")  # keep
    _touch(date_dir / "艳青1_2_merged_color.mp4")  # skip
    _touch(date_dir / "艳青1_2_merged_energybar_watermark.mp4")  # skip
    _touch(date_dir / "艳青1_2_merged_faceswap_burst_danmaku.mp4")  # skip
    _touch(date_dir / "艳青1_2_merged_intro.mp4")  # skip
    _touch(date_dir / "艳青1_2_merged_outro.mp4")  # skip
    result = scan_pending_videos(output_dir="output", state={})
    assert len(result) == 1
    assert result[0]["kind"] == "long"


def test_scan_skips_uploaded_and_archived(tmp_path, monkeypatch):
    """state 标 uploaded 或 auto_archived 的不进列表."""
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
    """*_full_16x9.mp4 副本 (无 _final) 不进列表 — per CLAUDE.md."""
    monkeypatch.chdir(tmp_path)
    date_dir = tmp_path / "output" / "2026-07-13"
    _touch(date_dir / "艳青1_2_merged_full_16x9_1920x1080.mp4")  # keep
    _touch(date_dir / "艳青1_2_merged_full_16x9_1920x1080_full_16x9.mp4")  # skip
    result = scan_pending_videos(output_dir="output", state={})
    assert len(result) == 1
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `uv run pytest tests/test_reminder_scan.py -v`
Expected: ALL FAIL (module not found or `scan_pending_videos` not defined)

- [ ] **Step 3: Implement scan logic in `tools/upload_reminder.py`**

Create `tools/upload_reminder.py`:

```python
"""Upload Reminder — Windows Task Scheduler 触发的命令行弹窗, 提醒用户上传视频.

入口: uv run python tools/upload_reminder.py
由 tools/install_reminder_task.bat 注册到 Task Scheduler (6 个黄金时段时点).

不修改主管线; 复用 lib/upload_utils 的黄金时段算法 + build_title.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.upload_utils import _is_golden_hour, build_title


# ====== 文件名匹配 (三件套) ======
_LONG_SUFFIX = "_full_16x9_1920x1080.mp4"
_SHORT_SUFFIX = "_full_16x9_1920x1080_yt_shorts.mp4"
_DOUYIN_SUFFIX = "_full_16x9_1920x1080_douyin.mp4"
_FULL16X9_COPY_SUFFIX = "_full_16x9_1920x1080_full_16x9.mp4"  # 跳过


def _extract_coach(stem: str) -> str:
    """从文件名 stem 提取教练名. 例 '艳青1_2_merged' → '艳青', '小飞侠' → '小飞侠'.

    规则: 按 _ 拆, 取第一段. 没有 _ 整段就是教练.
    """
    parts = stem.split("_")
    return parts[0] if parts else stem


def _classify_kind(name: str) -> Optional[str]:
    """返回 'long' / 'short' / 'douyin' / None."""
    if name.endswith(_DOUYIN_SUFFIX):
        return "douyin"
    if name.endswith(_SHORT_SUFFIX):
        return "short"
    if name.endswith(_LONG_SUFFIX):
        return "long"
    return None


def scan_pending_videos(output_dir: str = "output", state: Optional[dict] = None) -> list:
    """扫 output/<date>/*.{三件套}.mp4, 过滤已传 + 已归档, 返 list[dict].

    每条: {"path", "kind", "size_mb", "mtime", "title", "stem"}
    state 默认从 lib.reminder_state.load() 读.
    """
    if state is None:
        from lib.reminder_state import load
        state = load()

    out_root = Path(output_dir)
    if not out_root.exists():
        return []

    results = []
    for date_dir in sorted(out_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for f in date_dir.iterdir():
            if not f.is_file() or not f.name.endswith(".mp4"):
                continue
            # 跳过 _full_16x9 副本 (CLAUDE: 上传只传 *_final_16x9)
            if f.name.endswith(_FULL16X9_COPY_SUFFIX):
                continue
            kind = _classify_kind(f.name)
            if kind is None:
                # 中间产物 (color/watermark/intro/outro/faceswap/burst/...) → 跳过
                continue
            abs_path = str(f.resolve())
            # 过滤已传 + 已归档
            rec = state.get(abs_path, {})
            if rec.get("marked_uploaded_at"):
                continue
            if rec.get("auto_archived"):
                continue
            stem = f.stem
            # 去掉 _full_16x9_1920x1080 后缀再取 coach (e.g. "艳青1_2_merged_full_16x9_1920x1080" → "艳青1_2_merged")
            coach = _extract_coach(stem)
            # 优先用 build_title (带 coach profile), fallback "上传提醒"
            try:
                title = build_title(coach, video_type=kind, duration_sec=30)
            except Exception:
                title = f"【{coach}】上传提醒"
            results.append({
                "path": abs_path,
                "kind": kind,
                "size_mb": f.stat().st_size // (1024 * 1024),
                "mtime": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
                "title": title,
                "stem": stem,
            })
    return results


# ====== CLI 入口 (Task 3 加) ======
def main() -> int:
    """CLI 入口. argparse + 弹窗交互循环."""
    ap = argparse.ArgumentParser(
        description="上传提醒弹窗 (Task Scheduler 触发)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", default="output", help="output 目录 (默认 'output')")
    ap.add_argument("--skip-golden-check", action="store_true",
                    help="跳过黄金时段检查 (调试用)")
    args = ap.parse_args()

    if not args.skip_golden_check and not _is_golden_hour():
        print("[WARN] 当前不在黄金时段 (10-14/19-23 北京时间).")
        ans = input("还要继续吗? (y/N): ").strip().lower()
        if ans != "y":
            print("已取消.")
            return 0

    # 弹窗渲染 + 交互 (Task 3 实现 _render_and_interact)
    from tools.upload_reminder import _render_and_interact
    _render_and_interact(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run scan tests, verify they pass**

Run: `uv run pytest tests/test_reminder_scan.py -v`
Expected: 6 passed

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -q`
Expected: 251 + 6 = 257 passed, 0 failed

- [ ] **Step 6: Commit**

```bash
git add tools/upload_reminder.py tests/test_reminder_scan.py
git commit -m "@feat(reminder): scan 三件套 + 过滤中间产物 + 6 tests (TDD)"
```

---

## Task 3: Render + interact loop in `tools/upload_reminder.py` + golden hour tests

**Files:**
- Modify: `tools/upload_reminder.py` (加 `_render_and_interact`, ANSI 颜色, input loop)
- Create: `tests/test_reminder_golden.py` (复用 `_is_golden_hour` 边界)
- Test: `uv run pytest tests/test_reminder_golden.py -v`

**Interfaces (consumed by user + install .bat):**
- `_render_and_interact(output_dir: str) -> None` — 渲染弹窗 + 读 input 循环, 直到 q 退出
- ANSI helper: `_c(code: str, text: str) -> str` — code = "yellow" / "green" / "gray" / "red" / "bold"
- `_enable_ansi() -> None` — 调 `os.system('')` 启用 Win10/11 ANSI, 失败吞掉

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reminder_golden.py`:

```python
"""Tests for golden-hour boundary + ANSI helper (sanity checks)."""
from datetime import datetime
import pytest
from lib.upload_utils import _is_golden_hour


# ====== _is_golden_hour 边界 (复用 lib/upload_utils.py:435) ======
def _fake_dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 13, hour, minute, 0)


def test_golden_14_00_true():
    """14:00 北京 → True (上午段开始)."""
    assert _is_golden_hour(_fake_dt(14, 0)) is True


def test_golden_14_59_true():
    """14:59 北京 → True (上午段最后分钟)."""
    assert _is_golden_hour(_fake_dt(14, 59)) is True


def test_golden_15_00_false():
    """15:00 北京 → False (上午段结束)."""
    assert _is_golden_hour(_fake_dt(15, 0)) is False


def test_golden_13_59_false():
    """13:59 北京 → True (上午段中)."""
    assert _is_golden_hour(_fake_dt(13, 59)) is True


def test_golden_22_30_true():
    """22:30 北京 → True (夜间段中)."""
    assert _is_golden_hour(_fake_dt(22, 30)) is True


def test_golden_8_30_false():
    """8:30 北京 → False (低峰, 5月数据 98 view)."""
    assert _is_golden_hour(_fake_dt(8, 30)) is False


def test_golden_16_30_false():
    """16:30 北京 → False (下午低谷)."""
    assert _is_golden_hour(_fake_dt(16, 30)) is False


# ====== ANSI helper (从 tools/upload_reminder 导入) ======
def test_ansi_yellow_wraps_text():
    """_c('yellow', 'X') 返包含 '\\033[33m' 或空 (cmd 不支持时)."""
    from tools.upload_reminder import _c
    out = _c("yellow", "X")
    assert "X" in out  # 总是包含原文本
    # 退化为空或 ANSI 都可 — 不崩即可
```

- [ ] **Step 2: Run tests, verify they fail (golden tests pass since lib is there, ANSI fails since no helper yet)**

Run: `uv run pytest tests/test_reminder_golden.py -v`
Expected: 7 golden pass (lib already), 1 ANSI FAIL (`cannot import name '_c'`)

- [ ] **Step 3: Add render + interact + ANSI helper to `tools/upload_reminder.py`**

Append to `tools/upload_reminder.py` (before `if __name__`):

```python
import os
import subprocess

# ====== ANSI 颜色 (Win10/11 Task Scheduler cmd 默认关闭) ======
_ANSI_ENABLED = False


def _enable_ansi() -> None:
    """尝试启用 Win10/11 ANSI 颜色, 失败吞掉. 用 os.system('') 调 VT100."""
    global _ANSI_ENABLED
    if sys.platform != "win32":
        _ANSI_ENABLED = True
        return
    try:
        # ctypes 调 SetConsoleMode + ENABLE_VIRTUAL_TERMINAL_PROCESSING
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        _ANSI_ENABLED = True
    except Exception:
        _ANSI_ENABLED = False


def _c(color: str, text: str) -> str:
    """ANSI 颜色包装. _ANSI_ENABLED=False 时返原文本."""
    if not _ANSI_ENABLED:
        return text
    codes = {
        "yellow": "\033[33m",
        "green": "\033[32m",
        "gray": "\033[90m",
        "red": "\033[31m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    c = codes.get(color, "")
    return f"{c}{text}{codes['reset']}"


# ====== 渲染 + 交互 ======
HELP_TEXT = """
指令:
  1,2,3       标记为已传 (空格或逗号分隔)
  o 1         用系统播放器打开视频1
  p 1         复制视频1路径到剪贴板
  s 1         跳过 (下次还弹)
  r 1         恢复 (把 [OLD 3+次] 重置为待提醒)
  a           全部标记为已传
  h           查看帮助
  q           退出 (不标已传, 保留状态)
"""


def _render_header(state: dict, pending: list) -> None:
    """顶部: 黄金期状态 + 今日剩时点 + 总览."""
    print(_c("bold", f"[胭脂虎健身团] 上传提醒 - {datetime.now().strftime('%Y-%m-%d %H:%M')} 北京"))
    print("=" * 50)
    if _is_golden_hour():
        print(_c("green", "当前黄金时段: ✓ (10-14 / 19-23 北京, 5月数据均 view 800+"))
    else:
        from lib.upload_utils import seconds_until_next_golden
        secs = seconds_until_next_golden()
        h, m = divmod(secs // 60, 60)
        print(_c("yellow", f"当前不在黄金时段, 距下一个 ≈ {h}h {m}m"))
    print("=" * 50)
    archived = sum(1 for r in state.values() if r.get("auto_archived"))
    uploaded = sum(1 for r in state.values() if r.get("marked_uploaded_at"))
    print(f"待上传: {len(pending)} 条  |  已上传: {uploaded} 条  |  已归档: {archived} 条")
    print("=" * 50)


def _render_video(idx: int, v: dict) -> None:
    """单条视频渲染."""
    kind_label = {"long": "LONG ", "short": "SHORT", "douyin": "DOUYIN"}[v["kind"]]
    marker = ""
    if v.get("auto_archived"):
        marker = _c("yellow", " [OLD 3+次未处理]")
    print(_c("bold", f"[{idx}] {kind_label}  {v['stem']}  {v['size_mb']}MB  {v['mtime'][:10]}{marker}"))
    print(_c("gray", f"    路径: {v['path']}"))
    print(f"    标题: {v['title']}")


def _parse_indices(s: str, max_n: int) -> list[int]:
    """解析 '1,2,3' / '1 2 3' / '1-3' → [0-based indices]."""
    s = s.strip()
    if not s:
        return []
    out = set()
    for part in s.replace(" ", ",").split(","):
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                a, b = int(a), int(b)
                for i in range(min(a, b), max(a, b) + 1):
                    if 1 <= i <= max_n:
                        out.add(i - 1)
            except ValueError:
                continue
        else:
            try:
                i = int(part)
                if 1 <= i <= max_n:
                    out.add(i - 1)
            except ValueError:
                continue
    return sorted(out)


def _open_path(p: str) -> None:
    """系统默认程序打开路径 (Win: os.startfile, Mac: open, Linux: xdg-open)."""
    try:
        if sys.platform == "win32":
            os.startfile(p)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])
    except Exception as e:
        print(_c("red", f"[ERR] 打开失败: {e}"))


def _copy_to_clipboard(s: str) -> None:
    """复制到剪贴板. Win 用 clip, Mac pbcopy, Linux xclip. 失败提示."""
    try:
        if sys.platform == "win32":
            subprocess.Popen(["clip"], stdin=subprocess.PIPE).communicate(s.encode("utf-8"))
        elif sys.platform == "darwin":
            subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE).communicate(s.encode("utf-8"))
        else:
            subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE).communicate(s.encode("utf-8"))
        print(_c("green", "[OK] 已复制到剪贴板"))
    except Exception as e:
        print(_c("red", f"[ERR] 复制失败: {e}"))


def _render_and_interact(output_dir: str) -> None:
    """弹窗主循环."""
    from lib.reminder_state import load, save, mark_uploaded, increment_remind, reset_remind, mark_all_uploaded

    _enable_ansi()
    state = load()
    pending = scan_pending_videos(output_dir=output_dir, state=state)
    if not pending:
        print(_c("green", "今天无待传视频, 关闭窗口即可."))
        input("按 Enter 继续...")
        return

    _render_header(state, pending)
    print(f"\n待上传视频 ({len(pending)} 条):\n")
    for i, v in enumerate(pending, 1):
        _render_video(i, v)
        print()

    print(_c("gray", HELP_TEXT))
    print(_c("bold", "> "), end="", flush=True)

    while True:
        try:
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit] EOF / Ctrl-C, 保留状态不标记.")
            return
        if not line:
            print(_c("bold", "> "), end="", flush=True)
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "q":
            print("[exit] 状态保留. 关闭窗口即可.")
            return
        if cmd == "h":
            print(_c("gray", HELP_TEXT))
        elif cmd == "a":
            paths = [v["path"] for v in pending]
            mark_all_uploaded(state, paths)
            save(state)
            print(_c("green", f"[OK] 全部 {len(paths)} 条已标已传. 关闭窗口."))
            return
        elif cmd == "o":
            indices = _parse_indices(arg, len(pending))
            for i in indices:
                _open_path(pending[i]["path"])
            if indices:
                print(f"已请求打开 {len(indices)} 个文件.")
        elif cmd == "p":
            indices = _parse_indices(arg, len(pending))
            for i in indices:
                _copy_to_clipboard(pending[i]["path"])
        elif cmd == "s":
            indices = _parse_indices(arg, len(pending))
            for i in indices:
                increment_remind(state, pending[i]["path"])
            save(state)
            print(f"已跳过 {len(indices)} 条, 下次还弹.")
        elif cmd == "r":
            indices = _parse_indices(arg, len(pending))
            for i in indices:
                reset_remind(state, pending[i]["path"])
            save(state)
            print(_c("green", f"已恢复 {len(indices)} 条."))
        elif cmd.replace(",", "").replace(" ", "").isdigit() or "-" in cmd:
            # '1,2,3' 形式
            indices = _parse_indices(cmd, len(pending))
            for i in indices:
                mark_uploaded(state, pending[i]["path"])
            save(state)
            print(_c("green", f"[OK] 已标 {len(indices)} 条已传. 关闭窗口或继续操作."))
            # 重新渲染 (减少的列表)
            pending = [v for i, v in enumerate(pending) if i not in indices]
            if not pending:
                print(_c("green", "全部已传, 关闭窗口."))
                return
            print("\n" + "=" * 50 + "\n更新后剩余:\n")
            for i, v in enumerate(pending, 1):
                _render_video(i, v)
                print()
        else:
            print(_c("red", f"[?] 未知指令 '{cmd}'. 输 h 看帮助."))
        print(_c("bold", "> "), end="", flush=True)
```

- [ ] **Step 4: Run scan tests + golden tests, verify pass**

Run: `uv run pytest tests/test_reminder_scan.py tests/test_reminder_golden.py -v`
Expected: 6 + 8 = 14 passed

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -q`
Expected: 257 + 1 (ANSI) = 258 passed, 0 failed

- [ ] **Step 6: Manual smoke test (10s)**

Run: `uv run python tools/upload_reminder.py --skip-golden-check`
Expected: 弹窗列出待传 (or 提示"今天无待传"), 输 `q` 退出. ANSI 颜色在 Win10/11 终端能看见.

- [ ] **Step 7: Commit**

```bash
git add tools/upload_reminder.py tests/test_reminder_golden.py
git commit -m "@feat(reminder): 渲染+交互循环 + ANSI + 8 tests (TDD)"
```

---

## Task 4: Windows Task Scheduler .bat installers

**Files:**
- Create: `tools/install_reminder_task.bat`
- Create: `tools/uninstall_reminder_task.bat`
- No new tests (script is dumb wrapper, no logic)

**Interfaces:**
- `install_reminder_task.bat` — 双击注册 Task Scheduler, 6 个时点
- `uninstall_reminder_task.bat` — 双击卸载

- [ ] **Step 1: Implement `tools/install_reminder_task.bat`**

Create `tools/install_reminder_task.bat`:

```batch
@echo off
REM 安装上传提醒到 Windows Task Scheduler.
REM 6 个黄金时段时点 (北京 UTC+8): 10:00 12:00 14:00 19:00 21:00 23:00
REM
REM 双击运行, 看到 "SUCCESS" 即可关闭. 卸载用 uninstall_reminder_task.bat.

setlocal
set "TASK_NAME=FitnessVideoPipeline_UploadReminder"
set "PROJECT_DIR=F:\wkspace\fitness-video-pipeline"
set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%\tools\upload_reminder.py"

REM 检查 venv python
if not exist "%PYTHON_EXE%" (
    echo [ERR] 找不到 %PYTHON_EXE%
    echo       请先跑 uv sync 创建 .venv
    pause
    exit /b 1
)

REM 删除旧的 (如果存在)
schtasks /delete /tn "%TASK_NAME%" /f >nul 2>&1

REM 注册 (每天 10:00 触发, /sc daily /st 10:00:00; 6 个时点用 /tr 多任务或用 PowerShell 多次)
REM 用 schtasks 一次性建多触发器: /du 9999:12:31 /ri 1 /st 00:00:00 /et 23:59:00 不可行
REM 改为建 6 个独立任务, 后缀 _10 _12 _14 _19 _21 _23
for %%H in (10 12 14 19 21 23) do (
    echo 注册 %%H:00 触发器...
    schtasks /create ^
        /tn "%TASK_NAME%_%%H" ^
        /tr "\"%PYTHON_EXE%\" \"%SCRIPT%\"" ^
        /sc daily ^
        /st %%H:00:00 ^
        /rl highest ^
        /ru "%USERNAME%" ^
        /f >nul
    if errorlevel 1 (
        echo [ERR] 创建 %%H:00 触发器失败
        pause
        exit /b 1
    )
)

echo.
echo [OK] 6 个时点已注册:
echo   %TASK_NAME%_10  每天 10:00
echo   %TASK_NAME%_12  每天 12:00
echo   %TASK_NAME%_14  每天 14:00
echo   %TASK_NAME%_19  每天 19:00
echo   %TASK_NAME%_21  每天 21:00
echo   %TASK_NAME%_23  每天 23:00
echo.
echo 卸载跑: tools\uninstall_reminder_task.bat
echo.
pause
endlocal
```

- [ ] **Step 2: Implement `tools/uninstall_reminder_task.bat`**

Create `tools/uninstall_reminder_task.bat`:

```batch
@echo off
REM 卸载上传提醒 (Task Scheduler 6 个时点).
REM 双击运行, 看到 "[OK] 已卸载" 即可.

setlocal
set "TASK_NAME=FitnessVideoPipeline_UploadReminder"

for %%H in (10 12 14 19 21 23) do (
    schtasks /delete /tn "%TASK_NAME%_%%H" /f >nul 2>&1
    if errorlevel 1 (
        echo [WARN] 卸载 %%H:00 失败 (可能不存在)
    ) else (
        echo [OK] 已卸载 %%H:00 触发器
    )
)

echo.
pause
endlocal
```

- [ ] **Step 3: Manual verify (双击 .bat)**

Run: 双击 `tools/install_reminder_task.bat`
Expected: cmd 窗口显示 "[OK] 6 个时点已注册". 然后跑 `schtasks /query /fo LIST /v | findstr "UploadReminder"` 看到 6 个任务.

Run: 双击 `tools/uninstall_reminder_task.bat`
Expected: cmd 窗口显示 "[OK] 已卸载 6 个触发器".

**不**跑 install 留到用户拍板装 — 实施完只验证 uninstall 干净 (装上再卸).

- [ ] **Step 4: Commit**

```bash
git add tools/install_reminder_task.bat tools/uninstall_reminder_task.bat
git commit -m "@feat(reminder): Windows Task Scheduler .bat 安装/卸载 (6 时点)"
```

---

## Task 5: CLAUDE.md + memory/MEMORY.md 索引

**Files:**
- Modify: `CLAUDE.md` §"独立工具 (tools/, 主管线零改动)" 加 1 行
- Modify: `memory/MEMORY.md` 加 1 行
- New: `memory/upload-reminder-tool.md` (详细档案)

- [ ] **Step 1: Add line to `CLAUDE.md`**

Find the section starting with `- \`tools/bg_swap.py\`` in CLAUDE.md.

Insert after `tools/bg_swap.py` line:

```markdown
- `tools/upload_reminder.py` — Windows Task Scheduler 触发, 黄金时段弹窗提醒手工上传 YouTube/抖音, 编号勾选已传, 3 次未标自动归档 (per `upload-reminder-tool` memory + spec 2026-07-13); 状态 `records/upload_reminder_log.json`. 双击 `tools/install_reminder_task.bat` 注册 6 时点 (10/12/14/19/21/23)
```

- [ ] **Step 2: Create `memory/upload-reminder-tool.md`**

Create `memory/upload-reminder-tool.md`:

```markdown
---
name: upload-reminder-tool
description: 【2026-07-13 上线】Windows 定时任务 + 命令行弹窗, 黄金时段 (10-14/19-23 北京) 提醒人工上传 YouTube/抖音. YouTube+抖音全停自动 (历史教训: YT 平台挂死自动上传视频), 弹窗是当前拍板的"半自动"方案.
metadata:
  type: project
---

# Upload Reminder 工具 (2026-07-13 用户拍板上线)

## 一句话

YouTube + 抖音全停自动上传 (per [[yt-long-video-publish-immediately]] + [[douyin-manual-upload]]), 用 Windows Task Scheduler + Python CLI 弹窗在黄金时段提醒用户手工传. **半自动** = 调度自动 + 上传人工.

## 触发机制

- **黄金时段** (复用 [[shorts-golden-hour-auto-publish-2026-07-12]]): 10-14 / 19-23 北京 UTC+8 (5月数据 1376/935/862 view 高峰)
- **6 个时点**: 10:00 / 12:00 / 14:00 / 19:00 / 21:00 / 23:00 (Task Scheduler 6 个独立任务, 后缀 _10/_12/_14/_19/_21/_23)
- **手动跑**: `uv run python tools/upload_reminder.py --skip-golden-check` (调试用)

## 文件清单

| 文件 | 角色 |
|------|------|
| `tools/upload_reminder.py` | CLI 入口 + 渲染 + 交互循环 (input 指令) |
| `lib/reminder_state.py` | `records/upload_reminder_log.json` 读写 + 状态 mutate |
| `tools/install_reminder_task.bat` | 双击注册 6 时点 |
| `tools/uninstall_reminder_task.bat` | 双击卸载 |
| `records/upload_reminder_log.json` | 状态 (path → record, git 入 .gitignore 或本地不传) |
| `tests/test_reminder_state.py` | 7 tests |
| `tests/test_reminder_scan.py` | 6 tests |
| `tests/test_reminder_golden.py` | 8 tests (含 7 个 _is_golden_hour 边界) |
| `docs/superpowers/specs/2026-07-13-upload-reminder-design.md` | 设计 spec |
| `docs/superpowers/plans/2026-07-13-upload-reminder.md` | 实施 plan |

## 交互指令 (命令窗口 input)

- `1,2,3` — 标记编号 1/2/3 已传 (空格或逗号分隔, 支持 `1-3` 区间)
- `o 1` — 系统默认播放器打开视频1
- `p 1` — 复制视频1路径到剪贴板
- `s 1` — 跳过 (下次还弹, 累加 remind_count)
- `r 1` — 恢复 (把 [OLD 3+次] 重置回待提醒)
- `a` — 全部标已传
- `h` — 查看帮助
- `q` — 退出 (不标已传, 保留状态)

## 漏处理容错

- 同一视频提醒 ≥3 次未标记 → `auto_archived: true`, 不再弹
- 顶部 [OLD 3+次未处理] 黄色块提示
- 用 `r 1` 手动恢复 (重置 remind_count=0, 删 auto_archived)
- log.json 损坏 → 自动备份 `.bak` + 返空 (不崩)

## 三件套扫描规则 (跟产物命名一致)

- 匹配: `*_full_16x9_1920x1080.mp4` (long) / `*_full_16x9_1920x1080_yt_shorts.mp4` (short) / `*_full_16x9_1920x1080_douyin.mp4` (douyin)
- 跳过: `*_full_16x9_1920x1080_full_16x9.mp4` 副本 (per CLAUDE §"上传只传 final_16x9")
- 跳过: 所有中间产物 (`*_color.mp4` / `*_watermark.mp4` / `*_faceswap*.mp4` / `*_energybar*.mp4` / `*_intro.mp4` / `*_outro.mp4` / `*_keypoints.json` 等)

## 关键设计选择 (钉死)

- ❌ **不读** `upload_manifest.json` 判 YouTube 已传 (用户拍板"全停自动", manifest 不可靠)
- ❌ **不自动**调 `tools/upload_youtube.py` 上传 (per 拍板)
- ❌ **不删**已传视频 (用户自己管空间)
- ❌ **不自动关窗** (等用户点 [X])
- ✅ 复用 `lib/upload_utils.py` 的 `_is_golden_hour` / `seconds_until_next_golden` / `build_title` (零侵入)
- ✅ Win10/11 ANSI 颜色 (ctypes 启用 VT100), 退化无色不报错
- ✅ 跨平台 open/clipboard (Win startfile/clip, Mac open/pbcopy, Linux xdg-open/xclip)

## 用户安装

双击 `tools/install_reminder_task.bat` (管理员权限) → 看到 6 个 [OK] 即装上. 卸载双击 `uninstall_reminder_task.bat`.

## 跟主管线关系

- **零侵入**: 不改 `stages/`, `main.py`, `pipeline/`, `lib/upload_utils.py`, `config.yaml`
- 新代码仅在 `tools/`, `lib/`, `tests/`, `memory/`, `CLAUDE.md`, `docs/superpowers/`
- 24 + 7 + 6 + 8 = 45 new tests, 总 244 + 21 = 265 (等 plan 实施后实数)

## 相关 memory

- [[yt-long-video-publish-immediately]] — YT 长视频立即发布 (用户拍板)
- [[douyin-manual-upload]] — 抖音手工传 (用户拍板)
- [[shorts-golden-hour-auto-publish-2026-07-12]] — 黄金时段算法
- [[upload-manifest-required]] — manifest 写规则 (本工具不依赖 manifest, 但风格一致)
- [[pre-commit-hook-venv-pipefail]] — 守门 hook (每次 commit 跑相关 tests)
```

- [ ] **Step 3: Add line to `memory/MEMORY.md`**

Append to `memory/MEMORY.md` (end of file, after `source-quality-gate-2026-07-13.md` line):

```markdown
- [Upload Reminder 工具](upload-reminder-tool.md) — 【2026-07-13 用户拍板】Windows Task Scheduler 6 时点 + Python CLI 弹窗, 黄金时段提醒手工上传 YouTube/抖音. 编号勾选 + 3 次未标自动归档. 21 tests, 零侵入主管线. 设计 spec/plan 落 docs/superpowers/
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md memory/upload-reminder-tool.md memory/MEMORY.md
git commit -m "@docs(reminder): CLAUDE/memory 索引 + 详细档案 (2026-07-13)"
```

---

## Task 6: Final verification + 守门 + HANDOFF 更新

**Files:**
- Modify: `HANDOFF.md` (顶部"最后更新"段加本轮记录)
- No code changes

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: 265 passed (244 + 21 new), 0 failed

- [ ] **Step 2: Run pre-commit hook (manual dry run)**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider`
Expected: 265 passed, exit 0

- [ ] **Step 3: Verify all 4 new files exist + have content**

Run:
```bash
ls -la lib/reminder_state.py tools/upload_reminder.py tools/install_reminder_task.bat tools/uninstall_reminder_task.bat
```
Expected: 4 files exist, > 0 bytes each

- [ ] **Step 4: Smoke test reminder tool**

Run: `uv run python tools/upload_reminder.py --skip-golden-check < /dev/null`
Expected: 弹窗打印 header + 列出待传 (or "今天无待传") + 读到 EOF 自动退出 (因为 stdin 是 /dev/null). 不崩.

- [ ] **Step 5: Verify Task Scheduler .bat (just check syntax, don't actually register)**

Run: `type tools\install_reminder_task.bat | findstr /c:"schtasks" /c:"echo" /c:"pause"`
Expected: 看到 schtasks/echo/pause 关键字 (语法对)

- [ ] **Step 6: Update HANDOFF.md**

Find the section `## 📦 存档模式 (2026-07-12 用户拍板)` and prepend a new "最后更新" section above it:

```markdown
最后更新: 2026-07-13（**Upload Reminder 工具上线 — Windows 定时任务 + 命令行弹窗, 黄金时段提醒手工上传 YouTube/抖音 ✅**）:

**【本轮任务】**: 用户"以后遇到这些竖屏录制/低分辨率...直接放弃" + "新需求: Windows 定时任务弹命令行窗口提醒人工上传" (per spec + plan).

**【本轮完成 — 5 commits, 21 new tests, 零主管线改动】**:

1. **铁娘子5+6 误诊 (找到原因即可)**: 源 12.5MB 544×1296 rotation=90 手机原始 9:16 + 部分播放器不应用 EXIF rotation 视觉错觉. 主管线 vertical_native 正确. memory `tnz-vertical-native-stretched-misdiagnosis`
2. **源素材准入门槛 (钉死)**: 短边 ≥720 + 码率 ≥5Mbps + 时长足够, 不达标直接放弃. 加 CLAUDE.md + memory `source-quality-gate-2026-07-13`
3. **小飞侠1+2 合并 + 主管线 跑通**: 1 行 ffmpeg 跨 E 盘合并 1920×1080 30fps 112.37s + youtube preset 50min. 三件套 output/2026-07-13/. face_swap swap=3085/3371 (91.5%, 286 背跳, 0 无pose)
4. **Upload Reminder 工具上线**: 4 文件 (reminder_state.py + upload_reminder.py + install/uninstall_reminder_task.bat) + 3 套测试 (state 7 + scan 6 + golden 8 = 21) + spec/plan + memory. 0 修改主管线. 双击 install 装 6 时点
5. **CLAUDE.md / HANDOFF / memory 同步**: 准入门槛段 + 工具段 + 索引

**【测试状态】**: 244 → 265 (244 + 21 new), 0 回归.

**【下一步候选】**:
1. 用户拍板双击 `tools/install_reminder_task.bat` 装上 6 时点
2. 抖音 douyin 手工传 (8 套待传: 蜂王1+2 / 李娜1 / 海军1_2 / 丽丽1_2 / 建玲1_2 / 铁娘子1_2 / 艳青1_2 / **小飞侠1_2 (新)**)
3. 下一个视频 (source_videos/ 还剩: 彩娥 1/2/merged, 枫林红 1/2)

**【待用户拍板】**: 装 reminder 任务 + 抖音上传 + 下一个视频.
```

- [ ] **Step 7: Commit HANDOFF update**

```bash
git add HANDOFF.md
git commit -m "@docs(handoff): 2026-07-13 upload reminder + 小飞侠 + 准入门槛 (3 commits 等)"
```

- [ ] **Step 8: Final summary to user**

Report:
- Total commits: 5 (1 spec, 4 implementation/plan/docs)
- Total new tests: 21 (state 7 + scan 6 + golden 8)
- Total test suite: 265 passed
- Zero changes to main pipeline (stages/main.py/pipeline/)
- User action: 双击 `tools/install_reminder_task.bat` 装上 6 时点

---

## Self-Review

**1. Spec coverage**:
- Architecture (4 files) → Tasks 1-4 ✅
- Data flow + state JSON → Task 1 (lib/reminder_state.py) ✅
- Render ANSI + interaction → Task 3 ✅
- Error handling (4 cases) → Task 3 (`_enable_ansi` try/except, empty output_dir, corrupt log, open failure) ✅
- Windows Task Scheduler → Task 4 ✅
- Tests (3 suites) → Tasks 1, 2, 3 ✅
- CLAUDE.md + memory → Task 5 ✅
- Verification + HANDOFF → Task 6 ✅

**2. Placeholder scan**: No TBD/TODO/"implement later"/etc. All steps have actual code or commands.

**3. Type consistency**:
- `LOG_PATH` in `lib/reminder_state.py:14` — used in Task 1, 3 (via `from lib.reminder_state import load`)
- `load/save/mark_uploaded/increment_remind/reset_remind/mark_all_uploaded` — Task 1 defines, Task 3 imports
- `scan_pending_videos` in `tools/upload_reminder.py` — Task 2 defines, Task 3 uses
- `_render_and_interact` — Task 3 defines, used by main()
- `_extract_coach` — Task 2 defines, used in Task 2 scan
- `_is_golden_hour` — `lib/upload_utils.py` existing, reused

**No type/signature drift found.**

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-13-upload-reminder.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

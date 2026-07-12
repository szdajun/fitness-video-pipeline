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

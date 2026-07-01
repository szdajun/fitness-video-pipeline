# -*- coding: utf-8 -*-
"""SessionStart hook — 会话开局自动注入当前迭代状态.

新会话/恢复会话时自动打印 HANDOFF.md (活状态) + 最近 git 提交 + 最近上传,
让新会话无缝衔接上次进度 (本项目是长期项目, 会话常因 token 耗尽重启).

取代旧 .claude/hooks/SessionStart.yaml (那只是 spec, 从未注册).
CLAUDE.md 的"会话开局协议"是兜底: 即使本 hook 没触发, 新会话也会被指示读 HANDOFF.md.
"""
import json
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 坑
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent  # 项目根

# 1. HANDOFF.md 当前迭代状态 (核心: 让新会话看到上次停在哪)
handoff = ROOT / "HANDOFF.md"
if handoff.exists():
    lines = handoff.read_text(encoding="utf-8").splitlines()
    print("=" * 64)
    print("[HANDOFF] 当前迭代状态 (HANDOFF.md):")
    print("=" * 64)
    # 当前状态区在前 60 行 (已完成/进行中/待确认)
    print("\n".join(lines[:60]))
    if len(lines) > 60:
        print(f"... (共 {len(lines)} 行, 完整内容见 HANDOFF.md)")
    print()

# 2. 最近 git 提交 (最近干了什么)
try:
    r = subprocess.run(
        ["git", "log", "--oneline", "-5"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=5,
    )
    if r.returncode == 0 and r.stdout.strip():
        print("[GIT] 最近提交:")
        for line in r.stdout.strip().splitlines():
            print(f"  {line}")
        print()
except Exception:
    pass

# 3. 最近上传 manifest (防重复上传)
manifest = ROOT / "records" / "upload_manifest.json"
if manifest.exists():
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        recent = entries[-3:]
        print(f"[YT manifest] 最近 {len(recent)} 条上传:")
        for e in recent:
            c = str(e.get("coach", e.get("title", "?")))[:24]
            print(f"  - {c} ({e.get('type', '?')}) {e.get('ytid', '?')}")
        print()
    except Exception:
        pass

print("[开局] 先读 HANDOFF.md + docs/PROJECT_DESIGN.md, 再干活. "
      "会话结束前更新 HANDOFF.md.")

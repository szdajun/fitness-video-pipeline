"""Upload Reminder — Windows Task Scheduler 触发的命令行弹窗, 提醒用户上传视频.

入口: uv run python tools/upload_reminder.py
由 tools/install_reminder_task.bat 注册到 Task Scheduler (6 个黄金时段时点).

不修改主管线; 复用 lib/upload_utils 的黄金时段算法 + build_title.
"""
import argparse
import os
import subprocess
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

    规则: 按 _ 拆, 取第一段, 再去掉尾部连续数字. 没有 _ 整段就是教练.
    """
    parts = stem.split("_")
    first = parts[0] if parts else stem
    # 去掉尾部连续数字 (例 '艳青1' → '艳青', '小飞侠' 不变)
    stripped = first.rstrip("0123456789")
    return stripped or first


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
            # 去掉 _full_16x9_1920x1080 后缀再取 coach
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


# ====== CLI 入口 (Task 3 加 _render_and_interact 完整实现) ======
def main() -> int:
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

    # 弹窗渲染 + 交互
    _render_and_interact(args.output)
    return 0


# ====== ANSI 颜色 (Win10/11 Task Scheduler cmd 默认关闭) ======
_ANSI_ENABLED = False


def _enable_ansi() -> None:
    """尝试启用 Win10/11 ANSI 颜色, 失败吞掉."""
    global _ANSI_ENABLED
    if sys.platform != "win32":
        _ANSI_ENABLED = True
        return
    try:
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
  1,2,3       标记为已传 (空格或逗号分隔, 支持 1-3 区间)
  o 1         用系统播放器打开视频1
  p 1         复制视频1路径到剪贴板
  s 1         跳过 (下次还弹)
  r 1         恢复 (把 [OLD 3+次] 重置为待提醒)
  a           全部标记为已传
  h           查看帮助
  q           退出 (不标已传, 保留状态)
"""


def _render_header(state: dict, pending: list) -> None:
    """顶部: 黄金期状态 + 总览."""
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
    kind_label = {"long": "LONG ", "short": "SHORT", "douyin": "DOUYIN"}[v["kind"]]
    marker = ""
    if v.get("auto_archived"):
        marker = _c("yellow", " [OLD 3+次未处理]")
    print(_c("bold", f"[{idx}] {kind_label}  {v['stem']}  {v['size_mb']}MB  {v['mtime'][:10]}{marker}"))
    print(_c("gray", f"    路径: {v['path']}"))
    print(f"    标题: {v['title']}")


def _parse_indices(s: str, max_n: int) -> list:
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
    """系统默认程序打开路径."""
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
    """复制到剪贴板."""
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
        try:
            input("按 Enter 继续...")
        except EOFError:
            pass
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
            indices = _parse_indices(cmd, len(pending))
            for i in indices:
                mark_uploaded(state, pending[i]["path"])
            save(state)
            print(_c("green", f"[OK] 已标 {len(indices)} 条已传. 关闭窗口或继续操作."))
            pending = [v for ii, v in enumerate(pending) if ii not in indices]
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


if __name__ == "__main__":
    sys.exit(main())
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

    # 弹窗渲染 + 交互 (Task 3 实现 _render_and_interact)
    try:
        from tools.upload_reminder import _render_and_interact
        _render_and_interact(args.output)
    except ImportError:
        # Task 3 还没实现, 退化: 打印扫描结果
        from lib.reminder_state import load
        state = load()
        pending = scan_pending_videos(output_dir=args.output, state=state)
        print(f"待传视频: {len(pending)} 条")
        for v in pending:
            print(f"  [{v['kind']:6}] {v['stem']}  {v['size_mb']}MB  {v['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""通用 YouTube 上传 CLI — 参数化, 不用每次重写上传脚本.

标准教练 (在 COACH_PROFILES) 标题/描述/标签全自动:
  python tools/upload_youtube.py --coach 建玲 \\
      --long "output/2026-06-29/建玲2_3_merged_final_16x9_1920x1080.mp4" \\
      --short "output/2026-06-29/建玲2_3_merged_final_16x9_1920x1080_yt_shorts.mp4" \\
      --privacy public --date 2026-04-20

只传主视频 / 只传 Shorts: 省略对应参数即可.

自定义标题 (学员等不在 COACH_PROFILES 的, 如灼华娘子):
  python tools/upload_youtube.py --coach 灼华娘子 --short "..._60s.mp4" \\
      --title "【灼华娘子】户外燃脂操特写跟练 | 全程暴汗打卡 | 细柳营健身 #Shorts"

注意:
  --long 必须是 *final_16x9_1920x1080.mp4 (含片头片尾), 不是 *_full_16x9.mp4 副本
  upload_video 内置 verify (防大文件 200 OK 误拿旧视频) + 自动写 manifest
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.upload_utils import (upload_video, build_title, build_description,
                              LONG_TAGS, SHORTS_TAGS)


def main():
    ap = argparse.ArgumentParser(
        description="YouTube 上传 (long/short, 参数化)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coach", required=True, help="教练名 (建玲/艳青/丽丽...)")
    ap.add_argument("--long", dest="long_path",
                    help="主视频 *final_16x9_1920x1080.mp4 (含片头片尾)")
    ap.add_argument("--short", dest="short_path", help="Shorts *_yt_shorts.mp4")
    ap.add_argument("--privacy", default="public",
                    choices=["public", "private", "unlisted"])
    ap.add_argument("--date", help="录制日期 YYYY-MM-DD (省略用今天)")
    ap.add_argument("--short-duration", type=int, default=30,
                    help="Shorts 时长秒, 写入标题 (默认 30)")
    ap.add_argument("--title", help="自定义标题 (覆盖模板, 学员等场景用)")
    args = ap.parse_args()

    if not args.long_path and not args.short_path:
        ap.error("至少指定 --long 或 --short 之一")

    date_str = args.date or ""
    res = {}

    if args.long_path:
        lp = Path(args.long_path)
        if not lp.exists():
            print(f"[ERR] long 文件不存在: {lp}", file=sys.stderr)
            sys.exit(1)
        if "_full_16x9" in lp.name and "_final" not in lp.name:
            print(f"[WARN] {lp.name} 是 full_16x9 副本(去头去尾), "
                  f"应传 *final_16x9_1920x1080.mp4", file=sys.stderr)
        title = args.title or build_title(args.coach, date_str, "long")
        desc = build_description(args.coach, date_str, video_type="long")
        print(f"[upload] long  ({lp.stat().st_size // 1048576}MB): {title}")
        res["long"] = upload_video(str(lp), title, desc, LONG_TAGS,
                                   privacy=args.privacy, coach=args.coach,
                                   video_type="long")
        print(f"  => https://www.youtube.com/watch?v={res['long']}")

    if args.short_path:
        sp = Path(args.short_path)
        if not sp.exists():
            print(f"[ERR] short 文件不存在: {sp}", file=sys.stderr)
            sys.exit(1)
        title = (args.title or build_title(args.coach, date_str, "short",
                                           duration_sec=args.short_duration))
        desc = build_description(args.coach, date_str, video_type="short")
        print(f"[upload] short ({sp.stat().st_size // 1048576}MB): {title}")
        res["short"] = upload_video(str(sp), title, desc, SHORTS_TAGS,
                                    privacy=args.privacy, coach=args.coach,
                                    video_type="short")
        print(f"  => https://www.youtube.com/watch?v={res['short']}")

    print("\n[done]")
    for k, vid in res.items():
        print(f"  {k:6s}: https://www.youtube.com/watch?v={vid}")


if __name__ == "__main__":
    main()

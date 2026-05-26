#!/usr/bin/env python3
"""发布调度 CLI — 队列管理 + 定时发布

用法:
  python publish.py add --coach 艳青 --long long.mp4 --short short.mp4 --date 2026-04-20
  python publish.py status
  python publish.py run [--dry-run] [--interval 5]
  python publish.py schedule --type long --monday 20:00 --wednesday 20:00 --saturday 14:00
  python publish.py default [--type all]
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.publisher import PublishQueue, PublishEngine, WEEKDAYS, DEFAULT_LONG_SCHEDULE, DEFAULT_SHORTS_SCHEDULE, VIDEO_TYPE_LONG, VIDEO_TYPE_SHORTS, CST
from lib.upload_utils import build_title, build_description, LONG_TAGS, SHORTS_TAGS

QUEUE_FILE = str(Path(__file__).parent / "records" / "publish_queue.json")


def _setup_logger(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_add(args):
    queue = PublishQueue(QUEUE_FILE)
    has_long = args.long and Path(args.long).exists()
    has_short = args.short and Path(args.short).exists()

    if not has_long and not has_short:
        print("错误: 没有找到有效的视频文件")
        return 1

    if has_long:
        title = build_title(args.coach, args.date, VIDEO_TYPE_LONG)
        desc = build_description(args.coach, args.date, video_type=VIDEO_TYPE_LONG)
        vid = queue.add(args.coach, VIDEO_TYPE_LONG, str(Path(args.long).resolve()),
                        title, desc, LONG_TAGS, args.date)
        print(f"  长视频加入队列 [{vid}]: {Path(args.long).name}")
        print(f"    标题: {title}")

    if has_short:
        title = build_title(args.coach, args.date, VIDEO_TYPE_SHORTS)
        desc = build_description(args.coach, args.date, video_type=VIDEO_TYPE_SHORTS)
        vid = queue.add(args.coach, VIDEO_TYPE_SHORTS, str(Path(args.short).resolve()),
                        title, desc, SHORTS_TAGS, args.date)
        print(f"  短视频加入队列 [{vid}]: {Path(args.short).name}")
        print(f"    标题: {title}")

    print(f"\n  队列状态:")
    print(queue.summary())
    return 0


def cmd_status(args):
    queue = PublishQueue(QUEUE_FILE)
    print(queue.summary())


def cmd_run(args):
    _setup_logger(args.verbose)
    queue = PublishQueue(QUEUE_FILE)
    engine = PublishEngine(queue)
    engine.run_forever(interval_minutes=args.interval, dry_run=args.dry_run)


def cmd_schedule(args):
    queue = PublishQueue(QUEUE_FILE)
    day_schedule = {}
    for dow in WEEKDAYS:
        val = getattr(args, dow, None)
        if val:
            day_schedule[dow] = val if isinstance(val, list) else [val]
    if day_schedule:
        queue.set_schedule(args.type, day_schedule)
        print(f"  已更新 [{args.type}] 发布策略")
    else:
        cur = queue.schedule.get(args.type, {})
        print(f"  当前 [{args.type}] 发布策略 (北京时间):")
        for dow in WEEKDAYS:
            slots = cur.get(dow, [])
            if slots:
                print(f"    {dow}: {', '.join(slots)}")


def cmd_default(args):
    queue = PublishQueue(QUEUE_FILE)
    if args.type in (VIDEO_TYPE_LONG, "all"):
        queue.set_schedule(VIDEO_TYPE_LONG, DEFAULT_LONG_SCHEDULE)
        print("  长视频策略已恢复默认")
    if args.type in (VIDEO_TYPE_SHORTS, "all"):
        queue.set_schedule(VIDEO_TYPE_SHORTS, DEFAULT_SHORTS_SCHEDULE)
        print("  短视频策略已恢复默认")


def cmd_upload(args):
    """一键上传 + 定时发布视频"""
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"错误: 视频不存在: {video_path}")
        return 1

    # Lazy imports (cv2/PIL 较重)
    from lib.coach_profiles import get_coach, detect_coach_from_filename
    from lib.upload_utils import upload_video
    from datetime import datetime

    # 检测教练
    coach_name = detect_coach_from_filename(video_path.stem)
    coach = get_coach(coach_name)
    nickname = coach["nickname"]
    print(f"教练: {coach_name} / {nickname}")

    # 获取 SEO 元数据
    seo_data = None
    if args.seo:
        print(f"SEO: {args.seo}")
        with open(args.seo, encoding="utf-8") as f:
            seo_data = json.load(f)
    else:
        # 自动查找: 同目录下 *_seo.json 匹配教练名
        for sj in sorted(video_path.parent.glob("*_seo.json")):
            if coach_name in sj.stem or nickname in sj.stem:
                seo_data = json.loads(sj.read_text(encoding="utf-8"))
                print(f"自动发现 SEO: {sj.name}")
                break

    if seo_data:
        title = seo_data["title"]
        description = seo_data["description"]
        tags = seo_data["tags"]
    else:
        from lib.coach_profiles import generate_title, generate_description, generate_tags
        cfg = {"channel": args.channel, "intensity": "中等强度",
               "audience": "所有水平 / 新手友好",
               "tags": ["有氧健身操", "暴汗燃脂", "瘦全身", "居家健身", "跟练"]}
        title = generate_title(coach, cfg)
        description = generate_description(coach, cfg)
        tags = generate_tags(coach, cfg)

    # 发布时间
    if args.publish_at:
        publish_dt = datetime.strptime(args.publish_at, "%Y-%m-%d %H:%M").replace(tzinfo=CST)
        privacy = "private"
    elif args.public:
        publish_dt = None
        privacy = "public"
    else:
        # 默认: 明天 20:00（如未到 20:00 则今天 20:00）
        from datetime import timedelta
        now = datetime.now(CST)
        publish_dt = now.replace(hour=20, minute=0, second=0, microsecond=0)
        if publish_dt <= now:
            publish_dt += timedelta(days=1)
        privacy = "private"

    print(f"\n标题: {title}")
    if publish_dt:
        print(f"发布: 定时 {publish_dt.strftime('%Y-%m-%d %H:%M')} 北京时间")
    else:
        print("发布: 立即公开")
    print(f"标签: {len(tags)} 个")

    if args.dry_run:
        print(f"\n[Dry Run] 预览完成，未实际上传")
        if args.shorts:
            print(f"  Shorts 标题: 【{nickname}】{nickname}暴汗燃脂30秒 #Shorts")
        return 0

    # 生成缩略图
    print(f"\n生成缩略图...")
    thumb = None
    try:
        from auto_publish import generate_thumbnail
        day = 1
        dc_path = Path(__file__).parent / "day_counter.json"
        if dc_path.exists():
            dc = json.loads(dc_path.read_text(encoding="utf-8"))
            day = dc.get(coach_name, 1)
        thumb = generate_thumbnail(str(video_path), nickname, title, day=day)
        if thumb:
            print(f"  缩略图: {thumb}")
    except Exception as e:
        print(f"  缩略图跳过: {e}")

    # 上传主视频
    print(f"\n上传主视频...")
    ytid = upload_video(str(video_path), title, description, tags,
                        privacy=privacy, channel="fitness",
                        publish_at=publish_dt.isoformat() if publish_dt else None,
                        thumbnail_path=thumb)
    print(f"  OK => https://youtube.com/watch?v={ytid}")

    # 上传短视频
    if args.shorts:
        shorts_path = Path(args.shorts)
        if shorts_path.exists():
            shorts_title = f"【{nickname}】{nickname}暴汗燃脂30秒 #Shorts"
            shorts_desc = f"完整版在频道 · {nickname}带练\n\n#Shorts #细柳营胭脂虎 #暴汗燃脂"
            print(f"\n上传短视频...")
            print(f"  标题: {shorts_title}")
            ytid2 = upload_video(str(shorts_path), shorts_title, shorts_desc,
                                 tags + ["Shorts", "YouTubeShorts"],
                                 privacy="public", channel="fitness")
            print(f"  OK => https://youtube.com/watch?v={ytid2}")
        else:
            print(f"  短视频文件不存在: {shorts_path}")

    print(f"\n完成!")
    return 0


def main():
    parser = argparse.ArgumentParser(description="发布调度工具")
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="添加视频到发布队列")
    p_add.add_argument("--coach", required=True, help="带操人名称")
    p_add.add_argument("--long", default="", help="长视频文件路径")
    p_add.add_argument("--short", default="", help="短视频文件路径")
    p_add.add_argument("--date", default="", help="录制日期 (YYYY-MM-DD)")

    # status
    sub.add_parser("status", help="查看队列状态")

    # run
    p_run = sub.add_parser("run", help="启动发布调度（常驻循环）")
    p_run.add_argument("--dry-run", action="store_true", help="仅预览，不实际发布")
    p_run.add_argument("--interval", type=int, default=5, help="检查间隔（分钟）")
    p_run.add_argument("--verbose", action="store_true", help="详细日志")

    # schedule
    p_sch = sub.add_parser("schedule", help="查看/设置发布策略")
    p_sch.add_argument("--type", choices=[VIDEO_TYPE_LONG, VIDEO_TYPE_SHORTS], default=VIDEO_TYPE_LONG)
    for dow in WEEKDAYS:
        p_sch.add_argument(f"--{dow}", nargs="*", default=None,
                          help=f"如 --{dow} 20:00 12:30")

    # default
    p_def = sub.add_parser("default", help="恢复默认发布策略")
    p_def.add_argument("--type", choices=[VIDEO_TYPE_LONG, VIDEO_TYPE_SHORTS, "all"],
                      default="all")

    # upload
    p_up = sub.add_parser("upload", help="一键上传 + 定时发布")
    p_up.add_argument("video", help="视频文件路径")
    p_up.add_argument("--shorts", default="", help="短视频文件路径")
    p_up.add_argument("--seo", default="", help="SEO JSON 文件路径（不指定则自动查找）")
    p_up.add_argument("--channel", default="细柳营健身", help="频道名称")
    pub = p_up.add_mutually_exclusive_group()
    pub.add_argument("--publish-at", default="", help="定时发布时间，如 '2026-05-27 20:00'")
    pub.add_argument("--public", action="store_true", help="立即公开")
    p_up.add_argument("--dry-run", action="store_true", help="预览打印元数据，不实际上传")

    args = parser.parse_args()

    commands = {
        "add": cmd_add,
        "status": cmd_status,
        "run": cmd_run,
        "schedule": cmd_schedule,
        "default": cmd_default,
        "upload": cmd_upload,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

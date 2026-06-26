"""Stage 39: YouTube Shorts / 抖音竖版生成

设计 (2026-06-27 重构):
  - 复用 youtube 宽屏 final_path, 不再独立跑 douyin preset
  - 单入口 stages.short_vertical.make_vertical(profile, duration)
  - 滤镜复用 shorts_legacy_filters._opening_overlay_filter / _ending_cta_filter (含诗词 + coach_profiles 映射)
  - cx 自适应裁切 (前 60 帧 cx 中位数 → 静态 crop_x)
  - -ss 跳过宽屏 intro 时间段

Config:
    stages.shorts: bool = True              # 是否启用
    stages.shorts_yt: bool = True           # 生成 YouTube Shorts (30s)
    stages.shorts_douyin: bool = True       # 生成抖音完整版
    stages.shorts_duration: int = 30        # Shorts 时长(秒)
    stages.shorts_coach: str = ""           # 教练名 (用于片头)
    stages.shorts_intro_seconds: float = None  # 显式 intro 跳过秒数
"""
import os
from pathlib import Path

sys_path = str(Path(__file__).parent.parent)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from stages.short_vertical import make_vertical


class ShortsStage:
    def run(self, ctx):
        cfg = ctx.config.get("stages", {})
        # 默认开 (2026-06-27: 抖音 + YT Shorts 一套逻辑, 跑 youtube preset 默认就出竖版)
        if not cfg.get("shorts", True):
            print("  [跳过] shorts")
            return

        # 找输入视频: 优先 final_path → burst_path → mascot_path → watermark_path
        src = (ctx.get("final_path") or ctx.get("burst_path") or
               ctx.get("mascot_path") or ctx.get("watermark_path") or
               ctx.get("energybar_path") or ctx.get("highlight_path") or
               ctx.get("beatflash_path") or ctx.get("color_path"))
        if not src or not os.path.exists(str(src)):
            print("    跳过: 无可用上游视频作为 Shorts 输入")
            ctx.set("shorts_path", None)
            ctx.set("douyin_vertical_path", None)
            return

        stem = Path(ctx.input_path).stem
        kp_file = ctx.output_dir / f"{stem}_keypoints.json"
        if not kp_file.exists():
            kp_file = ctx.output_dir / f"{stem}_cropped_keypoints.json"

        coach = cfg.get("shorts_coach", "")
        duration = int(cfg.get("shorts_duration", 30))
        intro_seconds = cfg.get("shorts_intro_seconds", None)
        audio_src = str(ctx.input_path)  # 用原片音频保音质

        # 检测宽屏 intro (用于 -ss 跳过)
        intro_p = ctx.output_dir / f"{stem}_intro.mp4"
        intro_path = str(intro_p) if intro_p.exists() else None

        # YouTube Shorts (默认开)
        if cfg.get("shorts_yt", True):
            result = make_vertical(
                src_path=str(src), output_dir=str(ctx.output_dir),
                profile="yt_shorts",
                keypoints_file=str(kp_file) if kp_file.exists() else None,
                duration=duration, coach=coach,
                audio_src=audio_src,
                intro_path=intro_path, intro_seconds=intro_seconds,
            )
            if result:
                ctx.set("shorts_path", result)

        # 抖音竖版 (默认开, 出完整版)
        if cfg.get("shorts_douyin", True):
            result = make_vertical(
                src_path=str(src), output_dir=str(ctx.output_dir),
                profile="douyin",
                keypoints_file=str(kp_file) if kp_file.exists() else None,
                duration=None, coach=coach,  # None=完整版
                audio_src=audio_src,
                intro_path=intro_path, intro_seconds=intro_seconds,
            )
            if result:
                ctx.set("douyin_vertical_path", result)

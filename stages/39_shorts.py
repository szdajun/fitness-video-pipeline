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
        if not coach:
            # 2026-06-29 BUGFIX: 未传 --shorts-coach 时 coach="" → get_coach("") 被
            # _resolve_coach_name step3 的 `"" in key` 恒真误解析成字典首个最长 key
            # (小红豆) → 所有未指定教练的 Shorts 片头诗词都串成小红豆 (建玲踩过).
            # 正解: 从输入文件名提取教练名 (建玲1.mp4 → 建玲).
            from lib.coach_profiles import detect_coach_from_filename
            coach = detect_coach_from_filename(ctx.input_path)
        duration = int(cfg.get("shorts_duration", 30))
        intro_seconds = cfg.get("shorts_intro_seconds", None)
        # 2026-06-29 BUGFIX: audio_src 必须用 src (final_path), 不是 ctx.input_path (source).
        #   视频 src = final_path 含 intro(0-4s)+workout+outro, -ss 4 跳到 workout 起点.
        #   若 audio_src=source (无 intro), -ss 4 会取到 workout t=4 → 音视频错位 4s,
        #   且 source(194s) 从 t=4 只剩 190s → step2 -shortest 把视频截到 190s (丢尾 4s).
        #   final_path 音频是 export 已响度标准化的分发级音质, 用它既对齐又保音质.
        audio_src = str(src)

        # 检测宽屏 intro (用于 -ss 跳过)
        intro_p = ctx.output_dir / f"{stem}_intro.mp4"
        intro_path = str(intro_p) if intro_p.exists() else None

        # 2026-07-07 画中画小窗: 换脸后横屏缩 16:9 全景小窗, 诗词结束后全程常驻右上.
        # 内容源降级链: face_swap_path (换脸·干净横屏) > final_path (含文字) > source.
        pip_enabled = cfg.get("shorts_pip", True)
        pip_src = None
        if pip_enabled:
            for _key in ("face_swap_path", "final_path"):
                _cand = ctx.get(_key)
                if _cand and os.path.exists(str(_cand)):
                    pip_src = str(_cand)
                    break

        # 2026-07-07 高燃预览开场: 竖版最前拼全片最燃 ~4s (静音+字幕). yt_shorts + douyin 都加
        # (2026-07-07: 用户要抖音版也有爆燃预警). 拉升前 3 秒完播率 → 平台推荐权重.
        # 音频 anullsrc 真静音+concat 零错位 (⚠ 不用 adelay — 前导静音被 AAC gapless 当
        # encoder_delay 解码丢弃=错位, memory adelay-silence-gapless-strip).
        hook_enabled = cfg.get("shorts_hook", True)
        hook_dur = float(cfg.get("shorts_hook_dur", 4.0))

        # YouTube Shorts (默认开)
        if cfg.get("shorts_yt", True):
            result = make_vertical(
                src_path=str(src), output_dir=str(ctx.output_dir),
                profile="yt_shorts",
                keypoints_file=str(kp_file) if kp_file.exists() else None,
                duration=duration, coach=coach,
                audio_src=audio_src,
                intro_path=intro_path, intro_seconds=intro_seconds,
                pip_src=pip_src, pip_enabled=pip_enabled,
                hook_enabled=hook_enabled, hook_dur=hook_dur,
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
                pip_src=pip_src, pip_enabled=pip_enabled,
                hook_enabled=hook_enabled, hook_dur=hook_dur,
            )
            if result:
                ctx.set("douyin_vertical_path", result)

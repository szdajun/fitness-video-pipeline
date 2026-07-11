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

2026-07-10: 竖屏源 (9:16 native) 端到端通路
  - is_native_vertical=True 时: 走 _get_video_size() 实测 src 实际宽高
  - 优先 src=normalized_path (避开 EXIF 旋转 src → 走 baked 像素)
  - is_native_vertical=True 时: 不读 kp_file (不用 cx 跟领操)
  - 时长钳到 175s (用户拍板 ≤3 分钟)
  - 不调用 intro/cta PNG 之外不需 cx/PIP
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

        # 2026-07-10: 竖屏源优先拿 normalized_path (EXIF 旋转 baked 像素, 避开 sf bug)
        is_native_vertical = bool(cfg.get("is_native_vertical", False)) or \
                             bool(ctx.get("source_orientation", {}).get("is_vertical"))
        if is_native_vertical:
            normalized = ctx.get("normalized_path")
            if normalized and os.path.exists(str(normalized)):
                src = str(normalized)
                print(f"    [native-vertical] 用 normalized → {Path(src).name}")

        if not src or not os.path.exists(str(src)):
            print("    跳过: 无可用上游视频作为 Shorts 输入")
            ctx.set("shorts_path", None)
            ctx.set("douyin_vertical_path", None)
            return

        # 2026-07-10: 实测 src 实际宽高, 兜底判 is_native_vertical
        from stages.short_vertical import _get_video_size
        src_w, src_h = _get_video_size(str(src))
        if not is_native_vertical and src_w and src_h and src_h > src_w \
                and 0.50 <= (src_w / src_h) <= 0.65:
            is_native_vertical = True
            print(f"    [auto-detect] 源 {src_w}x{src_h} 是竖屏, 走 9:16 native 路径")

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

        # 2026-07-10: 竖源时长钳到 175s (用户拍板 ≤3 分钟 buffer)
        duration = int(cfg.get("shorts_duration", 30))
        if is_native_vertical and duration > 175:
            print(f"    [native-vertical] duration {duration}s > 175s, 钳到 175s")
            duration = 175

        intro_seconds = cfg.get("shorts_intro_seconds", None)
        if is_native_vertical and intro_seconds is None:
            # 2026-07-10: 竖源无片头, 强制 skip=0 (avoid 5s fallback 误砍)
            intro_seconds = 0.0

        # 2026-06-29 BUGFIX: audio_src 必须用 src (final_path), 不是 ctx.input_path (source).
        audio_src = str(src)

        # 检测宽屏 intro (用于 -ss 跳过). 竖源 intro_seconds 已强制 0
        intro_p = ctx.output_dir / f"{stem}_intro.mp4"
        intro_path = str(intro_p) if intro_p.exists() else None

        # 2026-07-07 画中画小窗. 2026-07-10: 竖源 PIP 关掉 (本来就全屏)
        pip_enabled = cfg.get("shorts_pip", True)
        pip_src = None
        if is_native_vertical:
            pip_enabled = False
        elif pip_enabled:
            for _key in ("face_swap_path", "final_path"):
                _cand = ctx.get(_key)
                if _cand and os.path.exists(str(_cand)):
                    pip_src = str(_cand)
                    break

        # 2026-07-07 高燃预览开场. 2026-07-12 用户拍板取消 (竖屏 hook 看起来很乱): 默认 False
        # 抖音 + Shorts 都不再加 hook. --with-hook CLI 还能 opt-in.
        hook_enabled = cfg.get("shorts_hook", False)
        hook_dur = float(cfg.get("shorts_hook_dur", 4.0))

        # 公共 kwargs (含 2026-07-10 新参数)
        common_kwargs = dict(
            keypoints_file=str(kp_file) if kp_file.exists() else None,
            coach=coach, audio_src=audio_src,
            intro_path=intro_path, intro_seconds=intro_seconds,
            pip_src=pip_src, pip_enabled=pip_enabled,
            hook_enabled=hook_enabled, hook_dur=hook_dur,
            is_native_vertical=is_native_vertical,
            src_w=src_w, src_h=src_h,
            force_intro_skip=0.0 if is_native_vertical else None,
        )

        # YouTube Shorts (默认开)
        if cfg.get("shorts_yt", True):
            result = make_vertical(
                src_path=str(src), output_dir=str(ctx.output_dir),
                profile="yt_shorts", duration=duration, **common_kwargs,
            )
            if result:
                ctx.set("shorts_path", result)

        # 抖音竖版 (默认开, 出完整版)
        if cfg.get("shorts_douyin", True):
            result = make_vertical(
                src_path=str(src), output_dir=str(ctx.output_dir),
                profile="douyin", duration=None, **common_kwargs,
            )
            if result:
                ctx.set("douyin_vertical_path", result)
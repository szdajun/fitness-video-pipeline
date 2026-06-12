#!/usr/bin/env python3
"""
make_video.py - 健身视频处理一键工具 (v1.0)

从源视频一键生成：
  - 16:9 长视频 (YouTube)
  - 9:16  跟拍竖版 (抖音)
  - 3:4   跟拍竖版 (小红书)
  - 30s Shorts (YouTube Shorts, 可选)

保留所有现有 v2 stage + 换脸/换背景 能力。
修复已知 bug:
  1. 2K 源视频未缩放 → 强制 1920x1080 输出
  2. F 盘满 → 启动前 df 检查 + 每 stage 完自动清 _temp
  3. intro/outto 静默 → 手工拼音轨覆盖全程
  4. time_base 错导致慢动作 → -video_track_timescale 30
  5. export 失败无报错 → subprocess.run(check=True)

用法:
    python make_video.py                                     # 菜单模式
    python make_video.py source_videos/李刚3.mp4             # CLI 模式
    python make_video.py source_videos/李刚3.mp4 --coach=李刚
    python make_video.py source_videos/李刚3.mp4 --bg-swap
    python make_video.py source_videos/李刚3.mp4 --ratios=9x16
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows 控制台默认 GBK, 强制 UTF-8 避免 emoji 报 UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 加项目根
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 复用现有工具
from make_video_lib import (
    FFMPEG, run, run_ffmpeg_check, run_bgswap,
    get_video_info, check_disk, clean_temp_dirs,
    track_crop, extract_full_audio, build_final_from_png,
    build_full_video, verify_video,
)

# 进度条 + 结构化日志 (logs/YYYY-MM-DD.log)
from lib.observability import setup_logging, get_logger, ProgressTracker, stage_progress

# 配置驱动: 所有路径/阈值从 config.yaml 的 paths 块读
# 找不到 config 用回硬编码缺省 (兼容首次部署)
def _load_paths_cfg():
    cfg_path = PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("paths", {})
    except Exception as e:
        print(f"!! 读 config.yaml 失败, 用缺省路径: {e}")
        return {}

_PATHS = _load_paths_cfg()

def _path(key, default):
    """读 paths.<key>, 没配置就用 default (相对项目根)."""
    v = _PATHS.get(key)
    return Path(v) if v else (PROJECT_ROOT / default)

SOURCE_DIR = _path("source_dir", "source_videos")
OUTPUT_ROOT = _path("output_root", "output")
TEMP_DIR = _path("temp_dir", "_temp")
TRACK3X4_DIR = _path("temp_track_3x4", "_temp_track3x4")
TRACK9X16_DIR = _path("temp_track_9x16", "_temp_track9x16")
COACH_FACES_DIR = _path("coach_faces_dir", "tools")
MIN_DISK_GB = int(_PATHS.get("min_disk_gb", 25))

# ============================================================================
#  配置菜单
# ============================================================================

def load_coaches():
    """从 coach_profiles 读教练列表"""
    try:
        from lib.coach_profiles import COACH_PROFILES
        return list(COACH_PROFILES.keys())
    except Exception as e:
        print(f"!! 读教练列表失败: {e}")
        return []


def show_config_menu(args):
    """显示配置菜单, 缺省值, 一次性确认.
    CLI 模式 (args.source 给定): 完全无 input, 用 args 缺省值跑.
    菜单模式: 交互输入 + 列表给 ffprobe 详情.
    """
    cli_mode = bool(args.source)
    print()
    print("=" * 70)
    print("  🎬  make_video.py v1.0 - 健身视频一键处理")
    print("=" * 70)
    print()

    # 1. 源视频
    source_videos = sorted(SOURCE_DIR.glob("*.mp4")) if SOURCE_DIR.exists() else []
    if not source_videos:
        print(f"!! 源视频目录为空: {SOURCE_DIR}")
        print("   请先把视频放到 source_videos/")
        sys.exit(1)

    print(f"📁 源视频目录: {SOURCE_DIR}")
    if cli_mode:
        # CLI 模式: 只列文件名, 不跑 ffprobe (44 个视频 *3s 太慢)
        idx = next((i for i, v in enumerate(source_videos, 1)
                    if v.name == Path(args.source).name
                    or str(v.resolve()) == str(Path(args.source).resolve())),
                   None)
        if idx is None:
            # 允许传任意路径
            source = Path(args.source).resolve()
        else:
            source = source_videos[idx - 1]
    else:
        # 菜单模式: 跑 ffprobe 给详情
        for i, v in enumerate(source_videos, 1):
            info = get_video_info(v)
            size_mb = v.stat().st_size / (1024**2)
            print(f"   [{i}] {v.name}  ({info['w']}x{info['h']} @ {info['fps']:.1f}fps, {info['duration']:.1f}s, {size_mb:.0f}MB)")
        print()
        choice = input(f"选源视频 [1-{len(source_videos)}, 回车=1]: ").strip()
        idx = int(choice) if choice else 1
        source = source_videos[idx - 1]
    info = get_video_info(source)
    print(f"  ✓ 选中: {source.name}")

    # 2. 教练
    coaches = load_coaches()
    print()
    print("👤 教练 (换脸用):")
    if args.coach or args.no_face_swap or cli_mode:
        coach_name = args.coach if (args.coach and args.coach in coaches) else None
        if args.no_face_swap:
            coach_name = None
    else:
        if coaches:
            for i, c in enumerate(coaches, 1):
                print(f"   [{i}] {c}")
            print(f"   [0] 不换脸 (跳过 face_swap stage)")
        choice = input("选教练 [0=不换脸, 1-N=选, 回车=0]: ").strip()
        choice = int(choice) if choice else 0
        coach_name = None if choice == 0 else coaches[choice - 1]
    print(f"  ✓ 教练: {coach_name or '(不换脸)'}")

    # 3. 换背景
    print()
    if args.bg_swap or cli_mode:
        do_bg_swap = bool(args.bg_swap)
    else:
        choice = input("启用 SAM2 背景替换? [y/n, 回车=n]: ").strip().lower()
        do_bg_swap = choice == "y"
    print(f"  ✓ 换背景: {'是' if do_bg_swap else '否'}")

    # 4. 输出比例
    print()
    if args.ratios:
        ratios = args.ratios.split(",")
    elif cli_mode:
        # CLI 模式默认全部
        ratios = ["16x9", "9x16", "3x4", "shorts"]
    else:
        print("📐 输出比例:")
        print("   [1] 16:9 (1920x1080)")
        print("   [2] 9:16  (1080x1920)  抖音")
        print("   [3] 3:4   (1080x1440)  小红书")
        print("   [4] Shorts 30s (9:16)")
        print("   [5] 全部")
        choice = input("选 [1-5, 回车=5]: ").strip()
        choice = int(choice) if choice else 5
        ratio_map = {1: ["16x9"], 2: ["9x16"], 3: ["3x4"], 4: ["shorts"], 5: ["16x9", "9x16", "3x4", "shorts"]}
        ratios = ratio_map.get(choice, ratio_map[5])
    print(f"  ✓ 输出: {', '.join(ratios)}")

    # 5. 显示配置确认
    print()
    print("⚙️  配置:")
    print(f"   源视频:   {source.name} ({info['w']}x{info['h']} @ {info['fps']:.1f}fps, {info['duration']:.1f}s)")
    print(f"   教练:      {coach_name or '(不换脸)'}")
    print(f"   换背景:    {'是' if do_bg_swap else '否'}")
    print(f"   输出比例:  {', '.join(ratios)}")

    if not cli_mode:
        confirm = input("\n开始处理? [回车=是, n=取消]: ").strip().lower()
        if confirm == "n":
            print("已取消")
            sys.exit(0)

    return {
        "source": source,
        "info": info,
        "coach": coach_name,
        "bg_swap": do_bg_swap,
        "ratios": ratios,
    }


# ============================================================================
#  写 preset (v2 链, 已修过)
# ============================================================================

PRESET_TEMPLATE = """# Auto-generated by make_video.py
# v2 完整链, 强制 16:9 输出 1920x1080
stages:
  stabilize: false
  h2v_convert: false
  body_warp: true
  ken_burns: true
  skin_smooth: true
  skin_tone_filter: false
  beat_flash: true
  energy_bar: true
  intro_outro: true
  audio: true
  face_beautify: false
  face_beautify2: false
  watermark: true
  mascot: true
  pip: false
  danmaku: true
  face_swap: {face_swap}
  bgm_beat: false

color_grade:
  brightness: 5
  contrast: 1.05
  saturation: 1.05
  warmth: 0
  clahe: true
  sharpen: 0.05
  temporal_smooth: 0.3
  adaptive_contrast: 0.2
  highlight_protect: 0.35
  highlight_threshold: 185
  highlight_blur: 5
  white_protect: 0.20
  white_value_threshold: 200
  white_sat_threshold: 60
  white_protect_blur: 5
  light_region_protect: 0.18
  light_region_threshold: 235
  light_region_min_area: 2500
  light_region_blur: 21

body_warp:
  overall_slim: 0.99
  waist_slim: 0.98
  leg_lengthen: 1.02

ken_burns:
  mode: smooth
  smooth_zoom_range: [1.0, 1.02]
  smooth_pan_range: [-0.01, 0.01]

energy_bar:
  width: 32
  margin_right: 20
  margin_bottom: 60
  height: 400
  smoothing: 0.85
  min_fill_ratio: 0.15
  motion_scale: 200

watermark:
  text: "胭脂虎·细柳营"
  position: "top-right"
  size: 24
  color: [255, 255, 255]
  alpha: 0.7
  margin: 20
  show_date: true

mascot:
  enabled: true
  size: 180
  position: bottom-left

skin_smooth:
  enabled: true
  strength: 0.25
  d: 7
  sigmaColor: 15
  sigmaSpace: 15
  downscale: 0.5
  skin_detect: true

danmaku:
  enabled: true
  font_size: 36
  interval: 25

audio:
  enabled: true
  target_lufs: -14.0
  fade_in: 0.5
  fade_out: 1.0
  denoise: 0
  ducking: 0

intro_outro:
  intro_duration: 4.0
  outro_duration: 2.5
  channel_name: "胭脂虎健身团"
  cta_text: "关注不迷路"
  audio_fade_out: 3.0
  fade_in_seconds: 1.0
  fade_out_seconds: 1.5

# 强制 16:9 输出 (修 2K bug)
output:
  width: 1920
  height: 1080
  crf: 22
  audio_bitrate: 128k
"""


def write_preset(coach):
    face_swap = "true" if coach else "false"
    path = PROJECT_ROOT / "presets" / "_tmp_makevideo.yaml"
    path.write_text(PRESET_TEMPLATE.format(face_swap=face_swap), encoding="utf-8")
    return path


# ============================================================================
#  步骤函数
# ============================================================================

def step_run_pipeline(cfg):
    """Step 1: 跑 v2 完整 pipeline (复用 main.py)"""
    log = get_logger("make_video")
    log.info(f"main.py 调用: source={cfg['source'].name} coach={cfg.get('coach')}")
    preset_path = write_preset(cfg["coach"])

    env = os.environ.copy()
    if cfg["coach"]:
        env["FACE_SWAP_COACH"] = cfg["coach"]

    # 跑 main.py process
    cmd = [
        sys.executable, "-u", "main.py", "process", str(cfg["source"]),
        "-c", str(preset_path),
        "--full-video",
    ]
    log.debug(f"cmd: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"v2 pipeline 失败 rc={result.returncode}")

    # 找输出目录
    import datetime
    mtime = os.path.getmtime(cfg["source"])
    date_str = datetime.date.fromtimestamp(mtime).isoformat()
    output_dir = OUTPUT_ROOT / date_str
    if not output_dir.exists():
        # 找最新目录
        candidates = sorted(OUTPUT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime)
        candidates = [c for c in candidates if c.is_dir()]
        if not candidates:
            raise RuntimeError("找不到 pipeline 输出目录")
        output_dir = candidates[-1]

    log.info(f"输出目录: {output_dir}")
    return output_dir


step_run_pipeline = stage_progress("Step 1: 跑 v2 pipeline")(step_run_pipeline)


def step_make_16x9(output_dir, cfg):
    """Step 2a: 拼 16:9 final"""
    if "16x9" not in cfg["ratios"]:
        return None
    log = get_logger("make_video")
    stem = cfg["source"].stem
    intro = output_dir / f"{stem}_intro.mp4"
    body = output_dir / f"{stem}_energybar_watermark_mascot_danmaku.mp4"
    outro = output_dir / f"{stem}_outro.mp4"
    if not all(p.exists() for p in [intro, body, outro]):
        log.warning("缺 intro/body/outro, 跳过 16:9 final")
        return None

    # 构造完整音轨 (38.4s)
    audio_aac = output_dir / "_full_audio.m4a"
    body_dur = get_video_info(body)["duration"]
    extract_full_audio(
        cfg["source"], audio_aac,
        intro_dur=4.0, outro_dur=2.5,
        fade_in=0.5, fade_out_sec=body_dur - 34.0,  # 31.9~34.4
    )

    final = output_dir / f"{stem}_full_16x9_final.mp4"
    build_full_video(intro, body, outro, audio_aac, final, 1920, 1080, fps=cfg["info"]["fps"])
    log.info(f"16:9 final → {final.name}")
    return final


step_make_16x9 = stage_progress("Step 2a: 拼 16:9 final (含音轨+修 time_base)")(step_make_16x9)


def step_make_9x16(output_dir, cfg):
    """Step 2b: 9:16 跟拍 + 拼"""
    if "9x16" not in cfg["ratios"]:
        return None
    log = get_logger("make_video")
    stem = cfg["source"].stem
    body_full = output_dir / f"{stem}_energybar_watermark_mascot_danmaku.mp4"
    kp = output_dir / f"{stem}_keypoints.json"
    if not (body_full.exists() and kp.exists()):
        log.warning("缺 body 或 keypoints, 跳过 9:16")
        return None

    # OpenCV 跟拍
    log.info("9:16 跟拍...")
    n = track_crop(body_full, kp, TRACK9X16_DIR, 1080, 1920, 9/16)

    # 拼主体
    audio_aac = output_dir / "_full_audio.m4a"
    if not audio_aac.exists():
        extract_full_audio(cfg["source"], audio_aac)

    body_916 = output_dir / f"{stem}_tracked_9x16.mp4"
    build_final_from_png(TRACK9X16_DIR, audio_aac, body_916, 1080, 1920, fps=cfg["info"]["fps"])

    # 拼 intro+body+outro
    intro = output_dir / f"{stem}_intro.mp4"
    outro = output_dir / f"{stem}_outro.mp4"
    if not (intro.exists() and outro.exists()):
        return body_916

    final = output_dir / f"{stem}_douyin_full_9x16.mp4"
    build_full_video(intro, body_916, outro, audio_aac, final, 1080, 1920, fps=cfg["info"]["fps"])
    log.info(f"9:16 final → {body_916.name} (跟拍+音轨, 跳过 intro/outro 避免 16:9 拉变形)")
    return body_916


step_make_9x16 = stage_progress("Step 2b: 9:16 跟拍 (抖音, 跳过 intro/outro)")(step_make_9x16)


def step_make_3x4(output_dir, cfg):
    """Step 2c: 3:4 跟拍 (小红书)

    策略: 跟 9:16 一样, 跳过 intro/outro, 只用跟拍后 body.
    """
    if "3x4" not in cfg["ratios"]:
        return None
    log = get_logger("make_video")
    stem = cfg["source"].stem
    body_full = output_dir / f"{stem}_energybar_watermark_mascot_danmaku.mp4"
    kp = output_dir / f"{stem}_keypoints.json"
    if not (body_full.exists() and kp.exists()):
        log.warning("缺 body 或 keypoints, 跳过 3:4")
        return None

    log.info("3:4 跟拍...")
    track_crop(body_full, kp, TRACK3X4_DIR, 1080, 1440, 3/4)

    audio_aac = output_dir / "_full_audio.m4a"
    if not audio_aac.exists():
        extract_full_audio(cfg["source"], audio_aac)

    body_34 = output_dir / f"{stem}_tracked_3x4.mp4"
    build_final_from_png(TRACK3X4_DIR, audio_aac, body_34, 1080, 1440, fps=cfg["info"]["fps"])
    log.info(f"3:4 final → {body_34.name} (跟拍+音轨, 跳过 intro/outro 避免 16:9 拉变形)")
    return body_34


step_make_3x4 = stage_progress("Step 2c: 3:4 跟拍 (小红书, 跳过 intro/outro)")(step_make_3x4)


def step_make_shorts(output_dir, cfg):
    """Step 2d: 30s Shorts 精华(v2 链自动生成, 复用 + 改名统一)"""
    if "shorts" not in cfg["ratios"]:
        return None
    log = get_logger("make_video")
    stem = cfg["source"].stem
    # main.py 内的 _make_shorts.py 自动产出的精华片段
    src = output_dir / f"{stem}_energybar_watermark_mascot_shorts_v2.mp4"
    if not src.exists():
        # 兼容旧命名
        src_alt = output_dir / f"{stem}_shorts_v2.mp4"
        if src_alt.exists():
            src = src_alt
        else:
            log.warning("shorts 没生成, 跳过 (检查 main.py 是否启用了 _make_shorts)")
            return None
    # 统一命名: 李刚3_shorts_9x16.mp4
    final = output_dir / f"{stem}_shorts_9x16.mp4"
    if final.exists() and final.stat().st_mtime >= src.stat().st_mtime:
        log.info(f"shorts 已就绪 → {final.name}")
        return final
    # 复制 (用 ffmpeg copy 保留时基, 不重编码)
    import shutil
    shutil.copy2(src, final)
    log.info(f"shorts → {final.name} (复用 v2 链精华片段)")
    return final


step_make_shorts = stage_progress("Step 2d: Shorts 30s 精华")(step_make_shorts)


def step_bgswap(output_dir, cfg):
    """Step 3: SAM2 背景替换 (可选)"""
    if not cfg.get("bg_swap"):
        return None
    if not cfg.get("coach"):
        print("[bgswap] 跳过: 未指定教练")
        return None

    print()
    print("=" * 60)
    print("Step 3: SAM2 背景替换")
    print("=" * 60)
    stem = cfg["source"].stem
    # 用 final 版本做 bgswap
    target = output_dir / f"{stem}_full_16x9_final.mp4"
    if not target.exists():
        target = output_dir / f"{stem}_energybar_watermark_mascot_danmaku.mp4"
    if not target.exists():
        print("  [SKIP] 找不到视频源")
        return None
    return run_bgswap(target, cfg["coach"])


def step_verify(output_dir, results):
    """Step 4: 抽帧验证"""
    print()
    print("=" * 60)
    print("Step 4: 抽帧验证")
    print("=" * 60)
    verify_dir = output_dir / "_verify"
    for name, path in results.items():
        if path and Path(path).exists():
            verify_video(path, verify_dir, timestamps=(0.5, 15, 30, 38), scale=480)


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="健身视频一键处理工具 v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", nargs="?", help="源视频路径 (省略则显示菜单)")
    parser.add_argument("--coach", help="教练名 (换脸用)")
    parser.add_argument("--no-face-swap", action="store_true", help="不换脸")
    parser.add_argument("--bg-swap", action="store_true", help="SAM2 背景替换")
    parser.add_argument("--bg-image", help="背景图路径 (用于 --bg-swap)")
    parser.add_argument("--ratios", default="",
                        help="输出比例 逗号分隔 (如 16x9,9x16,3x4,shorts)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别 (logs/YYYY-MM-DD.log)")
    args = parser.parse_args()

    # 启动: 先配日志, 后续所有事件落盘 logs/YYYY-MM-DD.log
    log_file = setup_logging(args.log_level)
    log = get_logger("make_video")
    log.info(f"启动 make_video.py | argv={sys.argv[1:]}")
    print(f"\n🎬 make_video.py v1.0  (日志: {log_file})\n")

    # 先清临时目录腾空间, 再检查磁盘 (否则 check_disk 会因为旧 track PNG 误判)
    clean_temp_dirs()
    if not check_disk(MIN_DISK_GB):
        log.error(f"磁盘不足 < {MIN_DISK_GB}GB (清完临时仍不够), 退出")
        sys.exit(1)
    log.debug(f"已清: {TEMP_DIR}, {TRACK3X4_DIR}, {TRACK9X16_DIR}")

    # 配置
    cfg = show_config_menu(args)
    if cfg.get("bg_swap") and args.bg_image:
        cfg["bg_image"] = args.bg_image
    log.info(f"配置: source={cfg['source'].name} coach={cfg.get('coach')} ratios={cfg['ratios']} bg_swap={cfg.get('bg_swap')}")

    # 步骤
    t_total = time.time()
    output_dir = step_run_pipeline(cfg)
    result_16x9 = step_make_16x9(output_dir, cfg)
    result_9x16 = step_make_9x16(output_dir, cfg)
    result_3x4 = step_make_3x4(output_dir, cfg)
    result_shorts = step_make_shorts(output_dir, cfg)
    step_bgswap(output_dir, cfg)

    results = {
        "16x9": result_16x9,
        "9x16": result_9x16,
        "3x4": result_3x4,
        "shorts": result_shorts,
    }
    step_verify(output_dir, results)

    # 总结
    total_dt = time.time() - t_total
    log.info(f"全部完成 ({total_dt:.1f}s 总耗时)")
    print()
    print("=" * 60)
    print(f"✅ 全部完成!  总耗时: {total_dt:.1f}s ({total_dt/60:.1f}min)")
    print("=" * 60)
    for k, v in results.items():
        if v:
            log.info(f"输出 {k}: {v}")
            print(f"   {k}: {v}")
    print("=" * 60)
    print(f"📄 完整日志: {log_file}")


if __name__ == "__main__":
    main()

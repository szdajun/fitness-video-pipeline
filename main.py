"""健身短视频处理流水线 — CLI 入口

用法:
    python main.py input.mp4                          # 默认处理
    python main.py input.mp4 -o output.mp4            # 指定输出路径
    python main.py input.mp4 --preset dramatic        # 使用预设风格
    python main.py input.mp4 --preview                # 预览模式（前3秒）
    python main.py input.mp4 --no-body-warp           # 禁用身体变形
    python main.py input.mp4 --leg-lengthen 1.2       # 自定义参数

批量处理:
    python main.py batch                              # 从桌面读取，输出到桌面/shorts_output/日期/
    python main.py batch -i <dir> -o <dir>            # 指定输入/输出目录
    python main.py batch --segment 45                 # 切割为45秒段（默认）
    python main.py batch --no-segment                 # 不切割
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import load_config, load_preset, _deep_merge
from pipeline.engine import PipelineEngine, PipelineContext
from split_video import split_video

import importlib

def _import_stage(module_name, class_name):
    return getattr(importlib.import_module(f"stages.{module_name}"), class_name)

PoseDetectStage = _import_stage("01_pose_detect", "PoseDetectStage")
StabilizeStage = _import_stage("02_stabilize", "StabilizeStage")
H2VConvertStage = _import_stage("03_h2v_convert", "H2VConvertStage")
KenBurnsStage = _import_stage("04_ken_burns", "KenBurnsStage")
BodyWarpStage = _import_stage("05_body_warp", "BodyWarpStage")
FaceWarpStage = _import_stage("08_face_warp", "FaceWarpStage")
ColorGradeStage = _import_stage("06_color_grade", "ColorGradeStage")
AudioStage = _import_stage("09_audio", "AudioStage")
SkeletonOverlayStage = _import_stage("10_skeleton_overlay", "SkeletonOverlayStage")
PersonCountStage = _import_stage("11_person_count", "PersonCountStage")
LeadBoxStage = _import_stage("12_lead_box", "LeadBoxStage")
LeadGhostStage = _import_stage("13_lead_ghost", "LeadGhostStage")
FaceBlurStage = _import_stage("14_face_blur", "FaceBlurStage")
MotionHeatmapStage = _import_stage("15_motion_heatmap", "MotionHeatmapStage")
SyncScoreStage = _import_stage("16_sync_score", "SyncScoreStage")
BeatFlashStage = _import_stage("17_beat_flash", "BeatFlashStage")
HighlightStage = _import_stage("18_highlight", "HighlightStage")
EnergyBarStage = _import_stage("19_energy_bar", "EnergyBarStage")
IntroOutroStage = _import_stage("20_intro_outro", "IntroOutroStage")
ExportStage = _import_stage("07_export", "ExportStage")

DEFAULT_INPUT_DIR = "C:/Users/18091/Desktop/短视频素材"
DEFAULT_OUTPUT_BASE = "C:/Users/18091/Desktop/shorts_output"


def build_single_parser():
    """单文件处理参数"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("input", help="输入视频路径")
    p.add_argument("-o", "--output", help="输出视频路径")
    p.add_argument("-c", "--config", help="配置文件路径 (.yaml)")
    p.add_argument("--preset", choices=["natural", "dramatic", "clean", "sexy", "night_gym", "gimbal", "beauty", "youtube", "shorts"],
                   help="使用预设风格")
    p.add_argument("--preview", action="store_true", help="预览模式（只处理前3秒）")
    p.add_argument("--preview-seconds", type=int, default=3, help="预览秒数")

    # 禁用阶段
    p.add_argument("--no-stabilize", action="store_true")
    p.add_argument("--no-body-warp", action="store_true")
    p.add_argument("--no-face-warp", action="store_true")
    p.add_argument("--no-color-grade", action="store_true")
    p.add_argument("--no-ken-burns", action="store_true")
    p.add_argument("--skeleton-overlay", action="store_true", help="叠加骨架显示")
    p.add_argument("--no-pose-gpu", action="store_true", help="禁用 pose GPU 加速（用 CPU）")
    p.add_argument("--full-video", action="store_true", help="生成完整视频（跳过精华片段选取）")
    p.add_argument("--audio", action="store_true", help="启用音频处理（响度标准化+背景音乐）")
    p.add_argument("--bg-music", type=str, help="背景音乐文件路径")
    p.add_argument("--bg-volume", type=float, default=0.25, help="背景音乐音量 (0.0-1.0)")
    p.add_argument("--target-lufs", type=float, default=-14.0, help="目标响度 LUFS (默认-14)")

    # 身体变形参数
    p.add_argument("--leg-lengthen", type=float, help="腿部拉长比例 (1.0-1.4)")
    p.add_argument("--leg-slim", type=float, help="腿部瘦比例 (0.7-1.0)")
    p.add_argument("--waist-slim", type=float, help="腰部瘦比例 (0.7-1.0)")
    p.add_argument("--head-ratio", type=float, help="头身比调整 (0.8-1.2)")
    p.add_argument("--overall-slim", type=float, help="整体瘦身 (0.7-1.0)")
    p.add_argument("--chest-enlarge", type=float, help="胸部放大 (1.0-1.3)")
    p.add_argument("--neck-lengthen", type=float, help="脖子拉长 (1.0-1.3)")

    # 色彩参数
    p.add_argument("--brightness", type=int, help="亮度 (-100~100)")
    p.add_argument("--contrast", type=float, help="对比度 (0.5~2.0)")
    p.add_argument("--saturation", type=float, help="饱和度 (0.0~2.0)")
    p.add_argument("--warmth", type=int, help="色温 (-50~50)")
    p.add_argument("--shadow", type=float, help="阴影修正 (0~1)")
    p.add_argument("--auto-wb", action="store_true", help="自动白平衡")
    p.add_argument("--adaptive-contrast", type=float, default=0, help="自适应对比度 (0~1)")
    p.add_argument("--pink-filter", type=float, default=None, help="粉色滤镜强度 (0~1, 默认1.0)")
    p.add_argument("--output-width", type=int, default=None, help="输出宽度 (默认保持原尺寸)")
    p.add_argument("--output-height", type=int, default=None, help="输出高度 (默认保持原尺寸)")
    p.add_argument("--cut", type=str, help="裁切重复片段 (秒), 如: 30-60,120-150")
    p.add_argument("--crf", type=int, help="视频质量 (18-28, 默认26, 越小越大)")
    p.add_argument("--enc-preset", choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
                   help="编码速度 (默认fast)")
    p.add_argument("--audio-bitrate", choices=["64k", "96k", "128k"],
                   help="音频码率 (默认96k)")
    p.add_argument("--video-fade-out", type=float, default=2.0,
                   help="片尾视频淡出秒数 (默认2.0)")

    return p


def build_batch_parser():
    """批量处理参数"""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-i", "--input-dir", default=DEFAULT_INPUT_DIR,
                   help=f"输入目录 (默认: {DEFAULT_INPUT_DIR})")
    p.add_argument("-o", "--output-dir", default=None,
                   help="输出目录 (默认: 桌面/shorts_output/日期/)")
    p.add_argument("--segment", type=int, default=45, help="切割时长秒数 (默认45, 0=不切割)")
    p.add_argument("--no-segment", action="store_true", help="不切割")
    p.add_argument("-c", "--config", help="配置文件路径 (.yaml)")
    p.add_argument("--preset", choices=["natural", "dramatic", "clean", "sexy", "night_gym", "gimbal", "beauty", "youtube", "shorts"],
                   default=None, help="预设风格 (默认: sexy)")
    p.add_argument("--no-stabilize", action="store_true")
    p.add_argument("--no-body-warp", action="store_true")
    p.add_argument("--no-face-warp", action="store_true")
    p.add_argument("--no-color-grade", action="store_true")
    p.add_argument("--no-ken-burns", action="store_true")
    p.add_argument("--skeleton-overlay", action="store_true", help="叠加骨架显示")
    p.add_argument("--no-pose-gpu", action="store_true", help="禁用 pose GPU 加速（用 CPU）")
    p.add_argument("--full-video", action="store_true", help="生成完整视频（跳过精华片段选取）")
    p.add_argument("--audio", action="store_true", help="启用音频处理（响度标准化+背景音乐）")
    p.add_argument("--bg-music", type=str, help="背景音乐文件路径")
    p.add_argument("--bg-volume", type=float, default=0.25, help="背景音乐音量 (0.0-1.0)")
    p.add_argument("--target-lufs", type=float, default=-14.0, help="目标响度 LUFS (默认-14)")
    p.add_argument("--preview", action="store_true", help="预览模式")
    p.add_argument("--cut", type=str, help="裁切重复片段 (秒), 如: 30-60,120-150")
    p.add_argument("--crf", type=int, help="视频质量 (18-28, 默认26, 越小越大)")
    p.add_argument("--enc-preset", choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
                   help="编码速度 (默认fast)")
    p.add_argument("--audio-bitrate", choices=["64k", "96k", "128k"],
                   help="音频码率 (默认96k)")
    p.add_argument("--video-fade-out", type=float, default=2.0,
                   help="片尾视频淡出秒数 (默认2.0)")
    p.add_argument("--auto-preset", action="store_true",
                   help="自动选择 preset：单人视频→gimbal，多人→beauty")
    p.add_argument("--workers", type=int, default=3,
                   help="并行处理视频数 (默认3, 内存不足时减少)")

    return p


def build_parser():
    """主解析器，包含 subparser"""
    p = argparse.ArgumentParser(
        description="健身短视频处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # 单文件处理（默认命令）
    single = sub.add_parser("process", parents=[build_single_parser()],
                            help="处理单个视频", aliases=["single"])
    single.set_defaults(command="process")

    # 批量处理
    batch = sub.add_parser("batch", parents=[build_batch_parser()],
                           help="批量处理文件夹内所有视频")
    batch.set_defaults(command="batch")

    return p


def _apply_cli_overrides(config, args):
    """将 CLI 参数合并到 config"""
    if hasattr(args, 'preview') and args.preview:
        config["preview"] = True

    if args.no_stabilize:
        config["stages"]["stabilize"] = False
    if args.no_body_warp:
        config["stages"]["body_warp"] = False
    if args.no_face_warp:
        config["stages"]["face_warp"] = False
    if args.no_color_grade:
        config["stages"]["color_grade"] = False
    if args.no_ken_burns:
        config["stages"]["ken_burns"] = False
    if args.no_pose_gpu:
        config["pose_gpu"] = False
    if args.full_video:
        config["full_video"] = True
        config["stages"]["highlight"] = False

    if getattr(args, 'audio', False):
        config["stages"]["audio"] = True
        config.setdefault("audio", {})["enabled"] = True
    if getattr(args, 'bg_music', None):
        config.setdefault("audio", {})["bg_music"] = args.bg_music
    if getattr(args, 'bg_volume', None) is not None:
        config.setdefault("audio", {})["bg_volume"] = args.bg_volume
    if getattr(args, 'target_lufs', None) is not None:
        config.setdefault("audio", {})["target_lufs"] = args.target_lufs
    if getattr(args, 'auto_preset', False):
        config["auto_preset"] = True

    if hasattr(args, 'leg_lengthen') and args.leg_lengthen is not None:
        config["body_warp"]["leg_lengthen"] = args.leg_lengthen
    if hasattr(args, 'leg_slim') and args.leg_slim is not None:
        config["body_warp"]["leg_slim"] = args.leg_slim
    if hasattr(args, 'waist_slim') and args.waist_slim is not None:
        config["body_warp"]["waist_slim"] = args.waist_slim
    if hasattr(args, 'head_ratio') and args.head_ratio is not None:
        config["body_warp"]["head_ratio"] = args.head_ratio
    if hasattr(args, 'overall_slim') and args.overall_slim is not None:
        config["body_warp"]["overall_slim"] = args.overall_slim
    if hasattr(args, 'chest_enlarge') and args.chest_enlarge is not None:
        config["body_warp"]["chest_enlarge"] = args.chest_enlarge
    if hasattr(args, 'neck_lengthen') and args.neck_lengthen is not None:
        config["body_warp"]["neck_lengthen"] = args.neck_lengthen

    if hasattr(args, 'brightness') and args.brightness is not None:
        config["color_grade"]["brightness"] = args.brightness
    if hasattr(args, 'contrast') and args.contrast is not None:
        config["color_grade"]["contrast"] = args.contrast
    if hasattr(args, 'saturation') and args.saturation is not None:
        config["color_grade"]["saturation"] = args.saturation
    if hasattr(args, 'warmth') and args.warmth is not None:
        config["color_grade"]["warmth"] = args.warmth
    if hasattr(args, 'shadow') and args.shadow is not None:
        config["color_grade"]["shadow"] = args.shadow
    if getattr(args, 'auto_wb', False):
        config["color_grade"]["auto_wb"] = True
    if hasattr(args, 'adaptive_contrast') and args.adaptive_contrast is not None:
        config["color_grade"]["adaptive_contrast"] = args.adaptive_contrast
    if hasattr(args, 'pink_filter') and args.pink_filter is not None:
        config["color_grade"]["pink_filter"] = args.pink_filter

    if hasattr(args, 'cut') and args.cut:
        ranges = []
        for part in args.cut.split(","):
            try:
                start, end = part.strip().split("-")
                ranges.append([float(start), float(end)])
            except ValueError:
                print(f"  警告: 无法解析裁切范围: {part}")
        if ranges:
            config.setdefault("output", {})["cut_ranges"] = ranges

    if hasattr(args, 'output_width') and args.output_width:
        config.setdefault("output", {})["width"] = args.output_width
    if hasattr(args, 'output_height') and args.output_height:
        config.setdefault("output", {})["height"] = args.output_height
    if hasattr(args, 'crf') and args.crf is not None:
        config.setdefault("output", {})["crf"] = args.crf
    if hasattr(args, 'enc_preset') and args.enc_preset:
        config.setdefault("output", {})["preset"] = args.enc_preset
    if hasattr(args, 'audio_bitrate') and args.audio_bitrate:
        config.setdefault("output", {})["audio_bitrate"] = args.audio_bitrate


def run_single(args):
    """处理单个视频"""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        sys.exit(1)

    config = load_config(args.config or "config.yaml")
    if args.preset:
        _deep_merge(config, load_preset(args.preset))

    if args.preview:
        config["preview"] = True
        config["preview_seconds"] = args.preview_seconds

    _apply_cli_overrides(config, args)

    if hasattr(args, 'output') and args.output:
        output_path = Path(args.output)
        config["output_dir"] = str(output_path.parent)
        config["output_file"] = output_path.name
    else:
        # 默认输出到源文件创建日期目录
        import os, datetime
        mtime = os.path.getmtime(input_path)
        file_date = datetime.date.fromtimestamp(mtime).isoformat()
        config["output_dir"] = str(Path("output") / file_date)

    ctx = PipelineContext(str(input_path), config)
    if "output_dir" in config:
        ctx.output_dir = Path(config["output_dir"])

    engine = PipelineEngine(config)
    stages_cfg = config["stages"]

    engine.add_stage("pose_detect", PoseDetectStage(),
                     enabled=stages_cfg.get("pose_detect", True))
    engine.add_stage("stabilize", StabilizeStage(),
                     enabled=stages_cfg.get("stabilize", True))
    engine.add_stage("h2v_convert", H2VConvertStage(),
                     enabled=stages_cfg.get("h2v_convert", True))
    engine.add_stage("body_warp", BodyWarpStage(),
                     enabled=stages_cfg.get("body_warp", True))
    engine.add_stage("ken_burns", KenBurnsStage(),
                     enabled=stages_cfg.get("ken_burns", False))
    engine.add_stage("face_warp", FaceWarpStage(),
                     enabled=stages_cfg.get("face_warp", False))
    engine.add_stage("color_grade", ColorGradeStage(),
                     enabled=stages_cfg.get("color_grade", True))
    engine.add_stage("audio", AudioStage(),
                     enabled=stages_cfg.get("audio", False))
    engine.add_stage("skeleton_overlay", SkeletonOverlayStage(),
                     enabled=stages_cfg.get("skeleton_overlay", False))
    engine.add_stage("person_count", PersonCountStage(),
                     enabled=stages_cfg.get("person_count", False))
    engine.add_stage("lead_box", LeadBoxStage(),
                     enabled=stages_cfg.get("lead_box", False))
    engine.add_stage("lead_ghost", LeadGhostStage(),
                     enabled=stages_cfg.get("lead_ghost", False))
    engine.add_stage("face_blur", FaceBlurStage(),
                     enabled=stages_cfg.get("face_blur", False))
    engine.add_stage("motion_heatmap", MotionHeatmapStage(),
                     enabled=stages_cfg.get("motion_heatmap", False))
    engine.add_stage("sync_score", SyncScoreStage(),
                     enabled=stages_cfg.get("sync_score", False))
    engine.add_stage("beat_flash", BeatFlashStage(),
                     enabled=stages_cfg.get("beat_flash", False))
    engine.add_stage("highlight", HighlightStage(),
                     enabled=stages_cfg.get("highlight", False))
    engine.add_stage("energy_bar", EnergyBarStage(),
                     enabled=stages_cfg.get("energy_bar", False))
    engine.add_stage("intro_outro", IntroOutroStage(),
                     enabled=stages_cfg.get("intro_outro", False))
    engine.add_stage("export", ExportStage(),
                     enabled=stages_cfg.get("export", True))

    engine.run(ctx)


def _quick_person_count(video_path: Path) -> int:
    """快速检测视频中出现的不同人数（使用姿态关键点）"""
    import cv2
    import importlib
    _pose_mod = importlib.import_module("stages.01_pose_detect")
    PoseDetectStage = getattr(_pose_mod, "PoseDetectStage")

    # 运行 pose 检测（只取前 300 帧采样，避免全量检测耗时）
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # 用 pose_detect stage
    ctx = PipelineContext(str(video_path), {"process_frames": min(300, frame_count)})
    try:
        PoseDetectStage().run(ctx)
    except Exception:
        return 1  # 检测失败默认单人

    keypoints = ctx.get("keypoints")
    if not keypoints:
        return 1

    # 简单位置追踪：统计不同 x 中心的人数
    import numpy as np
    tracks = {}
    for fi, pose_data in keypoints.items():
        if not pose_data:
            continue
        for pi, person_kps in enumerate(pose_data):
            kps = np.array(person_kps)
            vis = kps[:, 2] > 0.5
            if vis.sum() < 6:
                continue
            shoulders_cx = (kps[11][0] + kps[12][0]) / 2
            hips_cx = (kps[23][0] + kps[24][0]) / 2
            cx = (shoulders_cx + hips_cx) / 2

            best_tid = None
            best_dist = float('inf')
            for tid, trk in tracks.items():
                prev = np.median(trk)
                if abs(cx - prev) < best_dist:
                    best_dist = abs(cx - prev)
                    best_tid = tid
            if best_tid is not None and best_dist < 0.15:
                tracks[best_tid].append(cx)
            else:
                tracks[len(tracks)] = [cx]

    return max(len(tracks), 1)


def _scan_videos(input_dir: Path):
    """扫描目录中的 mp4 文件，跳过已处理的"""
    videos = []
    for f in sorted(input_dir.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() != ".mp4":
            continue
        # 跳过已处理的文件
        if "_final" in f.stem or "_preview" in f.stem:
            continue
        if "_kenburns" in f.stem or "_warped" in f.stem or "_color" in f.stem:
            continue
        if "_h2v" in f.stem:
            continue
        # 跳过隐藏文件
        if f.name.startswith("."):
            continue
        videos.append(f)
    return videos


def run_batch(args):
    """批量处理（多进程并行）"""
    import concurrent.futures
    import os

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 确定输出基础目录
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(DEFAULT_OUTPUT_BASE)

    # 加载基础配置
    config = load_config(args.config)
    preset_name = args.preset or "sexy"
    _deep_merge(config, load_preset(preset_name))
    _apply_cli_overrides(config, args)

    # 扫描视频
    videos = _scan_videos(input_dir)
    if not videos:
        print(f"在 {input_dir} 中未找到可处理的 mp4 文件")
        return

    total = len(videos)
    max_workers = max(1, args.workers)

    print("=" * 50)
    print(f"  批量处理模式 ({preset_name})")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_base}/<文件日期>/")
    print(f"  共 {total} 个视频，并行 {max_workers} 个")
    print(f"  切割: {'否' if args.no_segment else f'{args.segment}s'}")
    print("=" * 50)

    batch_start = time.time()

    # 自动 preset：先预扫描人数
    video_presets = {}
    if config.get("auto_preset"):
        print("  [预扫描] 检测视频人数...")
        scan_start = time.time()
        for video_path in videos:
            person_count = _quick_person_count(video_path)
            preset = "gimbal" if person_count <= 1 else "beauty"
            video_presets[video_path] = preset
        scan_elapsed = time.time() - scan_start
        gimbal_count = sum(1 for p in video_presets.values() if p == "gimbal")
        beauty_count = sum(1 for p in video_presets.values() if p == "beauty")
        print(f"  预扫描完成 ({scan_elapsed:.1f}s): 单人→gimbal({gimbal_count}个), 多人→beauty({beauty_count}个)")

    # 准备每个视频的任务参数
    video_tasks = []
    for video_path in videos:
        mtime = os.path.getmtime(video_path)
        file_date = date.fromtimestamp(mtime).isoformat()
        file_output_dir = output_base / file_date
        file_output_dir.mkdir(parents=True, exist_ok=True)
        this_preset = video_presets.get(video_path, preset_name)
        # 序列化 CLI overrides 为 dict（multiprocessing 需要）
        cli_overrides = _get_cli_overrides_dict(args)
        video_tasks.append((video_path, file_output_dir, this_preset, cli_overrides))

    # 多进程执行
    completed = 0
    success_count = 0
    fail_count = 0
    fail_list = []
    date_dirs = set()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_video_task, task): task
            for task in video_tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            video_path, file_output_dir, this_preset, _ = task
            date_dirs.add(str(file_output_dir))
            completed += 1
            try:
                success, name, err = future.result()
                if success:
                    success_count += 1
                    print(f"\n  [{completed}/{total}] [OK] {name}")
                else:
                    fail_count += 1
                    fail_list.append((name, err))
                    print(f"\n  [{completed}/{total}] [FAIL] {name}: {err}")
            except Exception as e:
                fail_count += 1
                fail_list.append((video_path.name, str(e)))
                print(f"\n  [{completed}/{total}] [FAIL] {video_path.name}: {e}")

    batch_elapsed = time.time() - batch_start

    print(f"\n{'=' * 50}")
    print(f"  批量处理完成")
    print(f"  成功: {success_count}/{total}")
    print(f"  失败: {fail_count}")
    if fail_list:
        for name, err in fail_list:
            print(f"    - {name}: {err}")
    print(f"  总耗时: {batch_elapsed:.0f}s ({batch_elapsed / 60:.1f}min)")
    print(f"  输出目录:")
    for d in sorted(date_dirs):
        print(f"    {d}")
    print(f"{'=' * 50}")


def _get_cli_overrides_dict(args):
    """从 args 提取可序列化的 overrides dict"""
    return {
        'no_stabilize': getattr(args, 'no_stabilize', False),
        'no_body_warp': getattr(args, 'no_body_warp', False),
        'no_face_warp': getattr(args, 'no_face_warp', False),
        'no_color_grade': getattr(args, 'no_color_grade', False),
        'no_ken_burns': getattr(args, 'no_ken_burns', False),
        'no_pose_gpu': getattr(args, 'no_pose_gpu', False),
        'full_video': getattr(args, 'full_video', False),
        'audio': getattr(args, 'audio', False),
        'bg_music': getattr(args, 'bg_music', None),
        'bg_volume': getattr(args, 'bg_volume', 0.25),
        'target_lufs': getattr(args, 'target_lufs', -14.0),
        'leg_lengthen': getattr(args, 'leg_lengthen', None),
        'leg_slim': getattr(args, 'leg_slim', None),
        'waist_slim': getattr(args, 'waist_slim', None),
        'overall_slim': getattr(args, 'overall_slim', None),
        'brightness': getattr(args, 'brightness', None),
        'contrast': getattr(args, 'contrast', None),
        'saturation': getattr(args, 'saturation', None),
        'warmth': getattr(args, 'warmth', None),
        'shadow': getattr(args, 'shadow', None),
        'auto_wb': getattr(args, 'auto_wb', False),
        'adaptive_contrast': getattr(args, 'adaptive_contrast', None),
        'cut': getattr(args, 'cut', None),
        'output_width': getattr(args, 'output_width', None),
        'output_height': getattr(args, 'output_height', None),
        'no_segment': getattr(args, 'no_segment', False),
        'segment': getattr(args, 'segment', 45),
        'crf': getattr(args, 'crf', None),
        'enc_preset': getattr(args, 'enc_preset', None),
        'audio_bitrate': getattr(args, 'audio_bitrate', None),
        'video_fade_out': getattr(args, 'video_fade_out', 2.0),
    }


def _process_video_task(task):
    """在子进程中处理单个视频"""
    from pathlib import Path as _Path
    from pipeline.config import load_config, load_preset, _deep_merge
    from pipeline.engine import PipelineEngine, PipelineContext
    import importlib
    from split_video import split_video

    video_path, file_output_dir, this_preset, cli_overrides = task

    try:
        # 动态导入 stage（子进程中需要重新 import）
        _pose = importlib.import_module("stages.01_pose_detect")
        _stab = importlib.import_module("stages.02_stabilize")
        _h2v = importlib.import_module("stages.03_h2v_convert")
        _kb = importlib.import_module("stages.04_ken_burns")
        _bw = importlib.import_module("stages.05_body_warp")
        _cg = importlib.import_module("stages.06_color_grade")
        _audio = importlib.import_module("stages.09_audio")
        _exp = importlib.import_module("stages.07_export")
        PoseDetectStage = getattr(_pose, "PoseDetectStage")
        StabilizeStage = getattr(_stab, "StabilizeStage")
        H2VConvertStage = getattr(_h2v, "H2VConvertStage")
        KenBurnsStage = getattr(_kb, "KenBurnsStage")
        BodyWarpStage = getattr(_bw, "BodyWarpStage")
        ColorGradeStage = getattr(_cg, "ColorGradeStage")
        AudioStage = getattr(_audio, "AudioStage")
        ExportStage = getattr(_exp, "ExportStage")

        # 重新构建 config
        file_config = load_config(None)
        _deep_merge(file_config, load_preset(this_preset))
        _apply_cli_overrides_from_dict(file_config, cli_overrides)

        ctx = PipelineContext(str(video_path), file_config)
        ctx.output_dir = _Path(file_output_dir)

        engine = PipelineEngine(file_config)
        stages_cfg = file_config["stages"]

        engine.add_stage("pose_detect", PoseDetectStage(),
                         enabled=stages_cfg.get("pose_detect", True))
        engine.add_stage("stabilize", StabilizeStage(),
                         enabled=stages_cfg.get("stabilize", True))
        engine.add_stage("h2v_convert", H2VConvertStage(),
                         enabled=stages_cfg.get("h2v_convert", True))
        engine.add_stage("body_warp", BodyWarpStage(),
                         enabled=stages_cfg.get("body_warp", True))
        engine.add_stage("ken_burns", KenBurnsStage(),
                         enabled=stages_cfg.get("ken_burns", False))
        engine.add_stage("face_warp", getattr(importlib.import_module("stages.08_face_warp"), "FaceWarpStage")(),
                         enabled=stages_cfg.get("face_warp", False))
        engine.add_stage("color_grade", ColorGradeStage(),
                         enabled=stages_cfg.get("color_grade", True))
        engine.add_stage("audio", AudioStage(),
                         enabled=stages_cfg.get("audio", False))
        engine.add_stage("export", ExportStage(),
                         enabled=stages_cfg.get("export", True))

        engine.run(ctx)

        # 切割
        final_path = ctx.get("final_path")
        if final_path and not cli_overrides.get('no_segment') and cli_overrides.get('segment', 0) > 0:
            split_video(final_path, cli_overrides['segment'])

        return (True, video_path.name, None)
    except Exception as e:
        return (False, video_path.name, str(e))


def _apply_cli_overrides_from_dict(config, overrides):
    """将 overrides dict 合并到 config"""
    if overrides.get('no_stabilize'):
        config["stages"]["stabilize"] = False
    if overrides.get('no_body_warp'):
        config["stages"]["body_warp"] = False
    if overrides.get('no_face_warp'):
        config["stages"]["face_warp"] = False
    if overrides.get('no_color_grade'):
        config["stages"]["color_grade"] = False
    if overrides.get('no_ken_burns'):
        config["stages"]["ken_burns"] = False
    if overrides.get('no_pose_gpu'):
        config["pose_gpu"] = False
    if overrides.get('full_video'):
        config["full_video"] = True
        config["stages"]["highlight"] = False

    if overrides.get('audio'):
        config["stages"]["audio"] = True
        config.setdefault("audio", {})["enabled"] = True
    if overrides.get('bg_music'):
        config.setdefault("audio", {})["bg_music"] = overrides['bg_music']
    if overrides.get('bg_volume') is not None:
        config.setdefault("audio", {})["bg_volume"] = overrides['bg_volume']
    if overrides.get('target_lufs') is not None:
        config.setdefault("audio", {})["target_lufs"] = overrides['target_lufs']

    for key, cfg_key in [('leg_lengthen', 'leg_lengthen'),
                          ('leg_slim', 'leg_slim'),
                          ('waist_slim', 'waist_slim'),
                          ('overall_slim', 'overall_slim')]:
        if overrides.get(key) is not None:
            config.setdefault("body_warp", {})[cfg_key] = overrides[key]

    for key, cfg_key in [('brightness', 'brightness'),
                          ('contrast', 'contrast'),
                          ('saturation', 'saturation'),
                          ('warmth', 'warmth')]:
        if overrides.get(key) is not None:
            config.setdefault("color_grade", {})[cfg_key] = overrides[key]

    if overrides.get('shadow') is not None:
        config.setdefault("color_grade", {})["shadow"] = overrides['shadow']
    if overrides.get('auto_wb'):
        config.setdefault("color_grade", {})["auto_wb"] = True
    if overrides.get('adaptive_contrast') is not None:
        config.setdefault("color_grade", {})["adaptive_contrast"] = overrides['adaptive_contrast']

    if overrides.get('cut'):
        ranges = []
        for part in overrides['cut'].split(","):
            try:
                start, end = part.strip().split("-")
                ranges.append([float(start), float(end)])
            except ValueError:
                pass
        if ranges:
            config.setdefault("output", {})["cut_ranges"] = ranges

    if overrides.get('output_width'):
        config.setdefault("output", {})["width"] = overrides['output_width']
    if overrides.get('output_height'):
        config.setdefault("output", {})["height"] = overrides['output_height']
    if overrides.get('crf') is not None:
        config.setdefault("output", {})["crf"] = overrides['crf']
    if overrides.get('enc_preset'):
        config.setdefault("output", {})["preset"] = overrides['enc_preset']
    if overrides.get('audio_bitrate'):
        config.setdefault("output", {})["audio_bitrate"] = overrides['audio_bitrate']
    if overrides.get('video_fade_out') is not None:
        config.setdefault("output", {})["video_fade_out"] = overrides['video_fade_out']


def main():
    parser = build_parser()

    # 如果没有 subcommand，且第一个参数看起来像文件路径，当作单文件处理
    args = parser.parse_args()

    if not args.command:
        # 兼容旧用法: python main.py input.mp4
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # 重新解析为单文件模式
            single_parser = build_single_parser()
            single_args = single_parser.parse_args(sys.argv[1:])
            run_single(single_args)
        else:
            parser.print_help()
    elif args.command == "batch":
        run_batch(args)
    elif args.command == "process":
        run_single(args)


if __name__ == "__main__":
    main()

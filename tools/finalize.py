"""
tools/finalize.py - 独立 final 合成工具

背景: main.py 的 07_export.py 在 GPU NVENC 上失败, 大量中间产物丢音/损坏.
此工具绕开 main.py, 直接用源视频 + keypoints.json 合成所有 final.

输出:
  - {stem}_full_16x9_final.mp4    (intro+body+outro+音轨, 16:9 letterbox)
  - {stem}_douyin_full_9x16.mp4   (纯跟拍+音轨, 9:16, 抖音)
  - {stem}_xhs_3x4_full.mp4       (纯跟拍+音轨, 3:4, 小红书)
  - {stem}_shorts_9x16.mp4        (30s 精华, 复用 main.py _make_shorts 产物)

用法:
  python tools/finalize.py source_videos/李刚3.mp4

设计原则:
  - 全部 CPU libx264 (稳, 不依赖 GPU)
  - 源视频音轨直接 copy 进 final (不重新编码, 不走 loudnorm/afade 等复杂 filter)
  - 9:16/3:4 跳过 intro/outro (避免 16:9 横版被强拉变形)
  - 16:9 final 仍拼 intro+body+outro (YouTube 习惯)
  - 用 build_full_video / build_final_from_png 的 fit+pad letterbox 防御
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from make_video_lib import (
    run, get_video_info, clean_temp_dirs, track_crop,
    extract_full_audio, build_final_from_png, build_full_video,
)
from lib.observability import setup_logging, get_logger, stage_progress


def ffmpeg_concat_video_audio(video_path, audio_path, output_path, fps=30):
    """最简单的: 视频 + 源视频音轨, CPU libx264 编码, mp4 faststart 适合上传."""
    audio_dur = get_video_info(audio_path)["duration"]
    video_dur = get_video_info(video_path)["duration"]
    print(f"  [F] concat {video_path.name} ({video_dur:.1f}s @ {fps}fps) + audio ({audio_dur:.1f}s) → {output_path.name}")
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-vf", f"fps={fps}",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ], check=True)


@stage_progress("Step A: 抽完整音轨 (intro+body+outro)")
def step_a_extract_full_audio(source, output_dir, fps):
    """从源视频抽音频, intro 淡入 + body 完整 + outro 淡出 (38.4s 完整版)."""
    audio = output_dir / "_full_audio.m4a"
    if not audio.exists():
        extract_full_audio(source, audio,
                           intro_dur=4.0, outro_dur=2.5,
                           fade_in=0.5, fade_out_sec=2.5)
    return audio


@stage_progress("Step B: 16:9 final (intro+body+outro)")
def step_b_16x9(source, output_dir, fps, full_audio):
    """拼 intro + body(mascot_danmaku 16:9) + outro + 完整音轨."""
    stem = source.stem
    intro = output_dir / f"{stem}_intro.mp4"
    body = output_dir / f"{stem}_energybar_watermark_mascot_danmaku.mp4"
    outro = output_dir / f"{stem}_outro.mp4"
    if not all(p.exists() for p in [intro, body, outro]):
        log = get_logger("finalize")
        log.warning(f"缺 intro/body/outro, 跳过 16:9 final")
        return None

    final = output_dir / f"{stem}_full_16x9_final.mp4"
    build_full_video(intro, body, outro, full_audio, final, 1920, 1080, fps=fps)
    return final


@stage_progress("Step C: 9:16 跟拍 (抖音)")
def step_c_9x16(source, output_dir, fps, full_audio):
    """9:16 跟拍 (跳过 intro/outro, 避免 16:9 拉变形)."""
    stem = source.stem
    body_full = output_dir / f"{stem}_energybar_watermark_mascot_danmaku.mp4"
    kp = output_dir / f"{stem}_keypoints.json"
    if not (body_full.exists() and kp.exists()):
        log = get_logger("finalize")
        log.warning(f"缺 body 或 keypoints, 跳过 9:16")
        return None

    track_dir = PROJECT_ROOT / "_temp_track9x16"
    n = track_crop(body_full, kp, track_dir, 1080, 1920, 9/16)
    out = output_dir / f"{stem}_tracked_9x16.mp4"
    build_final_from_png(track_dir, full_audio, out, 1080, 1920, fps=fps)
    # 重命名成 douyin 输出名
    douyin = output_dir / f"{stem}_douyin_full_9x16.mp4"
    shutil.copy2(out, douyin)
    return douyin


@stage_progress("Step D: 3:4 跟拍 (小红书)")
def step_d_3x4(source, output_dir, fps, full_audio):
    """3:4 跟拍 (跳过 intro/outro)."""
    stem = source.stem
    body_full = output_dir / f"{stem}_energybar_watermark_mascot_danmaku.mp4"
    kp = output_dir / f"{stem}_keypoints.json"
    if not (body_full.exists() and kp.exists()):
        return None

    track_dir = PROJECT_ROOT / "_temp_track3x4"
    n = track_crop(body_full, kp, track_dir, 1080, 1440, 3/4)
    out = output_dir / f"{stem}_tracked_3x4.mp4"
    build_final_from_png(track_dir, full_audio, out, 1080, 1440, fps=fps)
    xhs = output_dir / f"{stem}_xhs_3x4_full.mp4"
    shutil.copy2(out, xhs)
    return xhs


@stage_progress("Step E: shorts 30s 精华")
def step_e_shorts(source, output_dir, fps):
    """30s 精华: 直接从源视频抽最精彩 30s + 源视频音轨.
    不用 main.py 的 _shorts_v2 (因为没音), 自己重做.
    """
    import json
    stem = source.stem
    kp = output_dir / f"{stem}_keypoints.json"
    if not kp.exists():
        return None
    # 找运动能量最高的 30s 窗口 (基于 keypoints 的 frame displacement)
    with open(kp, encoding="utf-8") as f:
        data = json.load(f)
    frames = data["keypoints"]
    # 用 nose x 位移作为运动量指标 (跟原 main.py _make_shorts 类似)
    energy = []
    prev = None
    for k in sorted(frames.keys(), key=lambda x: int(x)):
        persons = frames[k]
        best = None
        for p in persons:
            score = sum(kp[2] for kp in p if kp[2] > 0)
            if best is None or score > best["_s"]:
                best = {"_s": score, "nose": p[0]}
        if best and best["nose"][2] > 0:
            if prev is not None:
                dx = abs(best["nose"][0] - prev)
                energy.append((k, dx))
            prev = best["nose"][0]
    # 滑窗 30s 找最大能量窗口
    if not energy:
        return None
    window = int(fps * 30)
    best_start = energy[0][0]
    best_score = -1
    for i in range(len(energy) - window + 1):
        score = sum(e[1] for e in energy[i:i+window])
        if score > best_score:
            best_score = score
            best_start = energy[i][0]
    start_t = int(best_start) / fps
    out = output_dir / f"{stem}_shorts_9x16.mp4"
    print(f"  [F] 抽 30s 精华 from t={start_t:.1f}s → {out.name}")
    run([
        "ffmpeg", "-y",
        "-ss", str(start_t),
        "-i", str(source),
        "-t", "30",
        "-vf", f"fps={fps},scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out),
    ], check=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="final 合成工具 (绕开 main.py export NVENC bug)")
    parser.add_argument("source", help="源视频路径")
    parser.add_argument("--ratios", default="16x9,9x16,3x4,shorts",
                        help="输出比例, 逗号分隔 (默认全部)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    log_file = setup_logging(args.log_level)
    log = get_logger("finalize")
    source = Path(args.source).resolve()
    if not source.exists():
        log.error(f"源视频不存在: {source}")
        sys.exit(1)
    log.info(f"源: {source.name}")

    # 输出目录
    import datetime
    mtime = source.stat().st_mtime
    date_str = datetime.date.fromtimestamp(mtime).isoformat()
    output_dir = PROJECT_ROOT / "output" / date_str
    if not output_dir.exists():
        log.error(f"输出目录不存在: {output_dir} (需要先跑 main.py)")
        sys.exit(1)
    log.info(f"输出目录: {output_dir}")

    # 源 fps
    info = get_video_info(source)
    fps = info["fps"]
    log.info(f"源 fps: {fps:.2f}")

    # Step A: 抽音轨
    full_audio = step_a_extract_full_audio(source, output_dir, fps)

    # Steps B-E
    ratios = set(args.ratios.split(","))
    results = {}
    if "16x9" in ratios:
        results["16x9"] = step_b_16x9(source, output_dir, fps, full_audio)
    if "9x16" in ratios:
        results["9x16"] = step_c_9x16(source, output_dir, fps, full_audio)
    if "3x4" in ratios:
        results["3x4"] = step_d_3x4(source, output_dir, fps, full_audio)
    if "shorts" in ratios:
        results["shorts"] = step_e_shorts(source, output_dir, fps)

    # 总结
    print()
    print("=" * 60)
    print("✅ 全部完成")
    print("=" * 60)
    for k, v in results.items():
        if v:
            size_mb = Path(v).stat().st_size / 1024 / 1024
            print(f"   {k}: {v} ({size_mb:.1f}MB)")
        else:
            print(f"   {k}: (skipped)")
    print(f"📄 日志: {log_file}")


if __name__ == "__main__":
    main()

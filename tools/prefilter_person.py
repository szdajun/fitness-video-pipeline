#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prefilter_person.py — 删掉人物不完整片段 (出画/缺头/缺脚/被画面边缘裁切), 拼接保留段.

定位: 换背景 (tools/bg_swap.py) 前的清洗预处理. RVM 抠像 + 合成静态背景时,
人物若部分出画 → 抠像 alpha 半透明残缺 + 合成后贴纸感/悬浮感加重, 故先把这些片段剪掉.

判据 (复用 student_closeup.detect_pose 的 YOLOv8-pose, blaze33 归一化坐标):
  - head_cut / head_miss: 鼻子被上边缘裁切或检测不到 → 缺头
  - feet_cut / feet_miss: 脚踝被下边缘裁切或检测不到 → 缺脚
  - side_out:            肩/髋横坐标贴近左右边缘 → 走出画面
  - few_kp(N):           全身 7 关键点 (鼻/双肩/双髋/双脚) 可见 < MIN_KP → 人不全/太小
  - no_person:           本帧无人 (孤立 dropout 由形态学后处理吸收, 不直接判 cut)

形态学后处理 (治检测抖动):
  1. 先把 < MIN_SEG 的 cut 段 (检测瞬时漏检) 填成 keep
  2. 再把 < MIN_SEG 的 keep 段 (噪声小岛) 抹成 cut

用法:
  python tools/prefilter_person.py <video> -o cleaned.mp4
  python tools/prefilter_person.py <video> -o cleaned.mp4 --min-seg 0.5 --preview
  python tools/prefilter_person.py <video> --preview   # 只出 timeline, 不渲染 (先看判据)

输出:
  - cleaned.mp4 (默认): 逐段精确重编码 + concat, 音频同步保留
  - --preview: stdout 打印逐帧 timeline + 汇总 (kept/cut 帧数 + 段数 + 各 cut 原因计数)

注意 (像素优先, 不靠视觉模型): 全身 7 关键点的可见性 + 边缘贴近度是客观信号;
但「人物填满画面是常态还是出画信号」因构图而异, 调 --margin 适配. 边界 ±1s 精度对
「删不完整片段」够用; 需逐帧精确请用 --accurate.
"""
import os
import sys
import subprocess
import shutil
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # tools/ 自身: import student_closeup
from student_closeup import detect_pose  # noqa: E402

# ffmpeg 可移植解析 (与 bg_swap.py 一致: 已知好路径优先于 PATH, 因 Winget 版有编码 bug)
_KNOWN_FFMPEG = r"C:/Users/18091/ffmpeg/ffmpeg.exe"
def _resolve_ffmpeg(override=None):
    """--ffmpeg CLI > BG_FFMPEG env > 已知好路径 (if exists) > PATH > 兜底.
    已知好路径优先于 PATH: Winget 版 ffmpeg 有编码 bug (CLAUDE.md). 换机器落到 PATH."""
    if override:
        return override
    env = os.environ.get("BG_FFMPEG")
    if env:
        return env
    if os.path.isfile(_KNOWN_FFMPEG):
        return _KNOWN_FFMPEG
    found = shutil.which("ffmpeg")
    if found:
        return found
    return _KNOWN_FFMPEG
FFMPEG = None  # main() 里赋值

# student_closeup.to_blaze33 映射出的「真实」COCO 关键点索引 (其余 33 槽位是重复占位)
NOSE, LSH, RSH, LHIP, RHIP, LANK, RANK = 0, 11, 12, 23, 24, 27, 28
REAL_KP = (NOSE, LSH, RSH, LHIP, RHIP, LANK, RANK)


def pick_best_person(persons):
    """多人帧里选纵向跨度最大的 (近/全身者); 跨度并列时无所谓."""
    best = None
    best_score = -1.0
    for p in persons:
        arr = np.asarray(p, dtype=float)
        vis = arr[arr[:, 2] > VIS]
        if len(vis) < 3:
            continue
        ext = vis[:, 1].max() - vis[:, 1].min()  # 归一化高度
        if ext > best_score:
            best_score = ext
            best = arr
    return best


def assess(p, top_m, bot_m, side_m, min_kp):
    """单帧完整性. 返回 (complete: bool, reasons: list[str])."""
    if p is None:
        return False, ["no_person"]
    reasons = []
    # 头
    nose = p[NOSE]
    if nose[2] <= VIS:
        reasons.append("head_miss")
    elif nose[1] <= top_m:
        reasons.append("head_cut")
    # 脚
    feet = [p[i] for i in (LANK, RANK) if p[i][2] > VIS]
    feet_in = [f for f in feet if f[1] < 1 - bot_m]
    if not feet_in:
        reasons.append("feet_cut" if feet else "feet_miss")
    # 横向 (肩 + 髋)
    body = [p[i] for i in (LSH, RSH, LHIP, RHIP) if p[i][2] > VIS]
    if body:
        xs = [b[0] for b in body]
        if min(xs) < side_m or max(xs) > 1 - side_m:
            reasons.append("side_out")
    # 关键点总数
    vis_n = sum(1 for i in REAL_KP if p[i][2] > VIS)
    if vis_n < min_kp:
        reasons.append(f"few_kp{vis_n}")
    return (len(reasons) == 0), (reasons or ["ok"])


def _runs(status):
    """把 [0/1] list 压成 [(value, start, end_inclusive)]."""
    out = []
    if not status:
        return out
    cur = status[0]
    s = 0
    for i in range(1, len(status)):
        if status[i] != cur:
            out.append((cur, s, i - 1))
            cur = status[i]
            s = i
    out.append((cur, s, len(status) - 1))
    return out


def _fill_short(status, target_val, other_val, min_len):
    """把长度 < min_len 的 target_val 段刷成 other_val (原位改 list)."""
    for val, s, e in _runs(status):
        if val == target_val and (e - s + 1) < min_len:
            for i in range(s, e + 1):
                status[i] = other_val
    return status


def build_keep(statuses, min_seg_frames):
    """statuses: list[(complete, reasons)] → keep[0/1] list (长度 = N).
    先吸收检测 dropout (短 cut 填 keep), 再抹掉噪声 keep 小岛."""
    raw = [0 if (s is None or not s[0]) else 1 for s in statuses]
    # 1. 短 cut 段 (检测瞬时漏检) → keep
    raw = _fill_short(raw, target_val=0, other_val=1, min_len=min_seg_frames)
    # 2. 短 keep 小岛 (噪声) → cut
    raw = _fill_short(raw, target_val=1, other_val=0, min_len=min_seg_frames)
    return raw


def keep_segments(keep):
    """keep[0/1] → [(start_frame, end_frame_inclusive), ...]."""
    segs = []
    for val, s, e in _runs(keep):
        if val == 1:
            segs.append((s, e))
    return segs


def render_clean(src, segments, fps, out_path, audio, accurate):
    """逐段精确重编码 + concat demuxer. audio=True 同步保留音频."""
    tmp_dir = out_path.parent / f"_tmp_{out_path.stem}"
    tmp_dir.mkdir(exist_ok=True)
    list_path = tmp_dir / "concat.txt"
    parts = []
    for i, (s, e) in enumerate(segments):
        ss = s / fps
        to = (e + 1) / fps
        part = tmp_dir / f"part_{i:03d}.mp4"
        cmd = [FFMPEG, "-y", "-loglevel", "error"]
        if accurate:
            cmd += ["-i", str(src), "-ss", f"{ss:.3f}", "-to", f"{to:.3f}"]
        else:
            cmd += ["-ss", f"{ss:.3f}", "-to", f"{to:.3f}", "-i", str(src)]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p"]
        if audio:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]
        cmd += [str(part)]
        subprocess.run(cmd, check=True)
        parts.append(part)
        list_path.write_text("".join(f"file '{p.name}'\n" for p in parts),
                             encoding="utf-8")
    # concat (同编码参数 → -c copy 安全)
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
           "-i", str(list_path), "-c", "copy", str(out_path)]
    subprocess.run(cmd, check=True)
    # 清临时段 (保留 concat.txt 无意义)
    for p in parts:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        list_path.unlink()
    except OSError:
        pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


def main():
    global FFMPEG, VIS
    ap = argparse.ArgumentParser(
        description="删掉人物不完整片段, 拼接保留段 (换背景前清洗).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("video", help="输入视频")
    ap.add_argument("-o", "--output", help="输出清洗后视频 (省略则仅 --preview)")
    ap.add_argument("--min-seg", type=float, default=0.5,
                    help="最短保留/裁剪段秒数 (短于此吸收为检测噪声)")
    ap.add_argument("--margin-top", type=float, default=0.04,
                    help="上边缘归一化裕度 (鼻子 y < 此值判缺头)")
    ap.add_argument("--margin-bot", type=float, default=0.04,
                    help="下边缘归一化裕度 (脚踝 y > 1-此值判缺脚)")
    ap.add_argument("--margin-side", type=float, default=0.03,
                    help="左右边缘归一化裕度 (肩/髋 x 贴边判出画)")
    ap.add_argument("--vis", type=float, default=0.3,
                    help="关键点可见性阈值")
    ap.add_argument("--min-kp", type=int, default=5,
                    help="7 全身关键点中至少可见几个 (少于此判人不全)")
    ap.add_argument("--no-audio", dest="audio", action="store_false",
                    help="丢弃音频 (默认保留, 与视频同步重编码)")
    ap.add_argument("--accurate", action="store_true",
                    help="逐帧精确 (慢, -ss 在 -i 后); 默认关键帧对齐 (±1s, 快)")
    ap.add_argument("--preview", action="store_true",
                    help="只打印 timeline + 汇总, 不渲染 (配合省略 -o 亦可)")
    ap.add_argument("--ffmpeg", help="ffmpeg 路径覆盖 (默认 PATH/env/兜底)")
    ap.add_argument("--cache-dir", default="output/prefilter_cache",
                    help="pose 缓存目录")
    args = ap.parse_args()

    FFMPEG = _resolve_ffmpeg(args.ffmpeg)
    VIS = args.vis
    src = Path(args.video)
    if not src.exists():
        sys.exit(f"[ERROR] 视频不存在: {src}")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{src.stem}_keypoints.json"
    print(f"[1/3] 姿态检测 (缓存 {cache.name}) ...")
    pose, info = detect_pose(str(src), cache)
    n = int(info.get("frames", 0))
    fps = float(info.get("fps", 30) or 30)
    w = int(info.get("width", 0)); h = int(info.get("height", 0))
    print(f"      {w}x{h} @ {fps:.2f}fps {n} 帧, 检测到 {len(pose)} 帧")

    print("[2/3] 完整性判定 ...")
    statuses = []
    for fi in range(n):
        persons = pose.get(fi, [])
        best = pick_best_person(persons) if persons else None
        statuses.append(assess(best, args.margin_top, args.margin_bot,
                               args.margin_side, args.min_kp))

    min_seg_frames = max(1, int(round(args.min_seg * fps)))
    keep = build_keep(statuses, min_seg_frames)
    segments = keep_segments(keep)
    kept = sum(keep)
    cut_reasons = {}
    for i, (complete, reasons) in enumerate(statuses):
        if keep[i] == 0:
            for r in reasons:
                cut_reasons[r] = cut_reasons.get(r, 0) + 1

    total_dur = n / fps if fps else 0
    kept_dur = kept / fps if fps else 0
    print(f"      保留 {kept}/{n} 帧 ({kept_dur:.2f}s / {total_dur:.2f}s), "
          f"{len(segments)} 段")
    if cut_reasons:
        print("      裁剪原因统计 (帧数, 一帧可命中多个): "
              + ", ".join(f"{k}={v}" for k, v in sorted(
                  cut_reasons.items(), key=lambda kv: -kv[1])))

    # timeline (仅 cut 段精简打印; 全量用 --preview)
    if args.preview or not args.output:
        print("--- timeline (C=cut K=keep, 每 1 字符 = 1 帧) ---")
        line = "".join("K" if b else "C" for b in keep)
        # 每行最多 60 帧 + 帧号标尺
        for i in range(0, len(line), 60):
            chunk = line[i:i + 60]
            print(f"{i:>4}: {chunk}")
        # 列出每个 cut 段的主因
        for val, s, e in _runs(keep):
            if val == 0:
                rs = []
                for fi in range(s, e + 1):
                    rs += statuses[fi][1]
                # 取该段出现最多的原因
                top = Counter(rs).most_common(1)[0][0]
                print(f"  cut 段 [{s}-{e}] ({(e - s + 1) / fps:.2f}s) 主因: {top}")

    if not args.output:
        print("[done] 未指定 -o, 仅预览 (加 -o cleaned.mp4 渲染)")
        return
    if not segments:
        sys.exit("[ERROR] 没有可保留片段 — 放宽 --margin / --min-kp / --min-seg 重试")
    if len(segments) == 1 and segments[0] == (0, n - 1):
        print("      (全程完整, 无需裁剪)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[3/3] 渲染清洗后视频 → {out} ...")
    render_clean(str(src), segments, fps, out, args.audio, args.accurate)
    print(f"[done] {out} ({kept_dur:.2f}s, {len(segments)} 段)")


if __name__ == "__main__":
    main()

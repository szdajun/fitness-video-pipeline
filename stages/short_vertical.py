"""stages/short_vertical.py — 抖音 + YouTube Shorts 共用的 9:16 竖版生成

设计目标 (2026-06-27 重构):
  - 复用 youtube 宽屏 final_path, 不再独立跑 douyin preset
  - 单入口 make_vertical(src, profile, duration)
  - 9:16 裁切跟随领操人: 前 N 帧 cx 中位数 → 静态 crop_x
  - profile ∈ {'yt_shorts', 'douyin'}:
      yt_shorts: 30s + 英文片头 + CTA
      douyin:   完整 + 中文片头 + 无 CTA
  - 砍掉宽屏 intro_outro 时间段 (ffmpeg -ss 跳过)

纯函数层 (本文件):
  - compute_crop_x_from_kp: 算 9:16 裁切窗口 cx
  - get_overlay_filters:    profile → drawtext filter 列表
  - resolve_intro_skip:     intro 时长探测 (用于 -ss 跳过)
"""
import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# 复用 face_swap 的领操人识别 (BlazePose 33 kp 假设)
from tools.face_swap import find_lead_person


# ── 1. cx 自适应裁切 ──────────────────────────────────

# 2026-06-27: PIL CTA 渲染器 (避开 ffmpeg 8.1 drawtext 解析 UTF-8 bug)
from tools.render_cta_overlay import render_cta_png as _render_cta_png  # noqa: E402

def render_cta_overlay(output_path: str):
    """2026-06-27: 用 PIL 渲染 CTA PNG, 给 ffmpeg overlay 用"""
    return _render_cta_png(output_path)

DEFAULT_CROP_W = 608         # 9:16 裁切宽度 (1080 * 9/16)
DEFAULT_FRAME_W = 1920       # 宽屏原始宽度
DEFAULT_LOOKHEAD = 60        # 前 N 帧 cx 中位数 (~2 秒 @ 30fps)
CROP_PADDING = 30            # 左右各留 30px 防贴边


def _extract_lead_cx(keypoints_dict: dict, frame_w: int,
                     lookhead_frames: int) -> List[float]:
    """从 kp_file 读前 N 帧, 每帧用 find_lead_person 找领操人 cx (归一化 0~1).

    Returns:
        list of cx values (归一化). 空 list 表示没找到任何 lead.
    """
    cxs = []
    # kp_file 格式: {frame_idx_str: [person, ...]} 或 {frame_idx_int: ...}
    sorted_keys = sorted(keypoints_dict.keys(),
                         key=lambda k: int(k) if str(k).isdigit() else 0)
    for k in sorted_keys[:lookhead_frames]:
        persons = keypoints_dict[k]
        if not persons:
            continue
        lead = find_lead_person(persons, frame_w=frame_w, frame_h=1080)
        if lead and len(lead) > 0 and len(lead[0]) >= 2:
            # cx = person kp 0~12 可见点 x 平均 (与 find_lead_person 评分一致)
            valid = [p for p in lead if len(p) > 2 and p[2] > 0.3]
            if len(valid) >= 3:
                cx_norm = sum(p[0] for p in valid[:13]) / min(len(valid), 13)
                cxs.append(cx_norm)
    return cxs


def compute_crop_x_from_kp(keypoints_dict: dict,
                           frame_w: int = DEFAULT_FRAME_W,
                           crop_w: int = DEFAULT_CROP_W,
                           lookhead_frames: int = DEFAULT_LOOKHEAD,
                           padding: int = CROP_PADDING) -> int:
    """算 9:16 裁切的 crop_x (左上角 x 坐标, 像素).

    Args:
        keypoints_dict: kp_file 解析后的 dict, {frame_idx: [person, ...]}
        frame_w: 宽屏原始宽度 (默认 1920)
        crop_w: 9:16 裁切窗口宽度 (默认 608)
        lookhead_frames: 取前 N 帧 cx 中位数 (默认 60)
        padding: 左右钳制留白 (默认 30px)

    Returns:
        crop_x 像素值, 钳制到 [padding, frame_w - crop_w - padding]
        当无 lead 数据时, fallback 到居中 crop_x = (frame_w - crop_w) // 2
    """
    fallback = (frame_w - crop_w) // 2

    if not keypoints_dict:
        return fallback

    cxs = _extract_lead_cx(keypoints_dict, frame_w, lookhead_frames)
    if not cxs:
        return fallback

    # 中位数 (排序后取中间)
    cxs_sorted = sorted(cxs)
    n = len(cxs_sorted)
    median_cx = (cxs_sorted[n // 2] if n % 2 == 1
                 else (cxs_sorted[n // 2 - 1] + cxs_sorted[n // 2]) / 2)

    # cx 归一化 → 像素 crop_x
    # 公式: 让 crop 中心 = median_cx * frame_w
    #        crop_x = median_cx * frame_w - crop_w / 2
    raw_crop_x = int(median_cx * frame_w - crop_w / 2)
    clamped = max(padding, min(raw_crop_x, frame_w - crop_w - padding))
    return clamped


# ── 1b. 逐段领操人裁切 (合并视频治本) ─────────────────
# 2026-07-01: 旧 compute_crop_x_from_kp 只取前 60 帧 cx → 单一静态 crop_x,
#   合并视频第二段领操人移位/背身就被裁出画面. 改逐段跟踪:
#   每帧最大体型人 cx (lead 代理, 领操人通常最近/最大, 自然跨机位切割) +
#   v21 分段 (cx 突变 >0.08 持续 = 段边界) + 段内可靠帧中位数 → 逐段 crop_x.
#   不用 build_tracks: 合并视频切割处 track id 会断/列表下标错位 (见 38_smart_crop.py:221 注释).


def _per_frame_lead_cx(kp_dict: dict, total_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """每帧领操人 torso cx (归一化 0~1) + 躯干宽 (可靠性指标).

    每帧取"最大体型(bbox 面积)人"作为 lead 代理. torso cx = 肩(11,12)+髋(23,24)可见点
    x 均值; 躯干宽 = 肩宽/髋宽均值 (越大检测越可靠, smart_crop v17 同款指标).

    Returns:
        (cxs, torso_ws): 均 np.float32 [total_frames]. 缺失帧 cx=0.5/torso_w=0.
    """
    cxs = np.full(total_frames, 0.5, dtype=np.float32)
    torso_ws = np.zeros(total_frames, dtype=np.float32)
    torso_idx = [11, 12, 23, 24]
    for fi in range(total_frames):
        pose = kp_dict.get(fi)
        if not pose:
            continue
        best_cx, best_size, best_tw = 0.5, -1.0, 0.0
        for person in pose:
            if not person or len(person) < 29:
                continue
            try:
                kps = np.array(person, dtype=np.float32)
            except (ValueError, TypeError):
                continue
            if kps.ndim != 2 or kps.shape[1] < 3:
                continue
            vis = kps[:, 2] > 0.3
            if vis.sum() < 4:
                continue
            torso_vis = vis[torso_idx]
            if torso_vis.sum() >= 2:
                cx = float(np.mean(kps[torso_idx][torso_vis, 0]))
                widths = []
                if vis[11] and vis[12]:
                    widths.append(abs(kps[12][0] - kps[11][0]))
                if vis[23] and vis[24]:
                    widths.append(abs(kps[24][0] - kps[23][0]))
                tw = float(np.mean(widths)) if widths else 0.0
            else:
                cx = float(np.mean(kps[vis, 0]))
                tw = 0.0
            xs, ys = kps[vis, 0], kps[vis, 1]
            size = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
            if size > best_size:
                best_cx, best_size, best_tw = cx, size, tw
        cxs[fi] = best_cx
        torso_ws[fi] = best_tw
    return cxs, torso_ws


def _detect_crop_segments(cxs: np.ndarray, jump_thr: float = 0.08) -> List[Tuple[int, int]]:
    """v21 分段: 滑动窗口中位数突变 (>jump_thr 且持续) → 段边界.

    Returns:
        [(start_frame, end_frame), ...]. 单段视频返回 [(0, n)].
    """
    n = len(cxs)
    if n < 60:
        return [(0, n)]
    seg_win = max(30, min(150, n // 20))
    roll_med = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo, hi = max(0, i - seg_win // 2), min(n, i + seg_win // 2)
        roll_med[i] = float(np.median(cxs[lo:hi]))

    breakpoints = []
    i = seg_win
    while i < n - seg_win:
        before = float(np.median(roll_med[i - seg_win:i]))
        after = float(np.median(roll_med[i:i + seg_win]))
        if abs(after - before) > jump_thr:
            # 持续性确认: 前后各扩 30 帧看是否仍 > 0.7*thr (过滤单帧噪声)
            lo2 = max(0, i - seg_win - 30)
            hi2 = max(0, i - 30)
            lo3 = min(n, i + 30)
            hi3 = min(n, i + seg_win + 30)
            b2 = float(np.median(roll_med[lo2:hi2])) if hi2 > lo2 else before
            a2 = float(np.median(roll_med[lo3:hi3])) if hi3 > lo3 else after
            if abs(a2 - b2) > jump_thr * 0.7:
                breakpoints.append(i)
                i += seg_win  # 跳过一个窗口, 避免同一边界重复检出
                continue
        i += 5

    # 合并过近断点 (< seg_win)
    merged: List[int] = []
    for bp in breakpoints:
        if merged and bp - merged[-1] < seg_win:
            continue
        merged.append(bp)
    bounds = [0] + merged + [n]
    segs = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
    # 合并过短段 (< seg_win 帧 ≈ 5s) 到前一段, 杀掉背身噪声产生的碎段
    merged_segs: List[Tuple[int, int]] = []
    for s, e in segs:
        if merged_segs and (e - s) < seg_win:
            merged_segs[-1] = (merged_segs[-1][0], e)
        else:
            merged_segs.append((s, e))
    return merged_segs if merged_segs else [(0, n)]


def _seg_target_cx(cxs: np.ndarray, torso_ws: np.ndarray,
                   seg_start: int, seg_end: int,
                   reliable_thr: float = 0.04) -> float:
    """段内目标 cx: 可靠帧 (torso_w > thr) 中位数; 不足 5 个可靠帧退全段中位数."""
    seg_cx = cxs[seg_start:seg_end]
    seg_tw = torso_ws[seg_start:seg_end]
    reliable = seg_tw > reliable_thr
    if reliable.sum() >= 5:
        return float(np.median(seg_cx[reliable]))
    return float(np.median(seg_cx))


def compute_crop_x_segments(kp_dict: dict,
                            frame_w: int = DEFAULT_FRAME_W,
                            crop_w: int = DEFAULT_CROP_W,
                            fps: float = 30.0,
                            padding: int = CROP_PADDING
                            ) -> Tuple[List[Tuple[int, int, int]], str]:
    """逐段算 crop_x (像素) + ffmpeg crop x 表达式, 跟随领操人跨合并视频各段.

    Args:
        kp_dict: pose 关键点 dict {frame_idx: [person_kps, ...]} (key 可 int/str)
        frame_w: 宽屏宽 (默认 1920)
        crop_w: 9:16 裁切宽 (默认 608)
        fps: 源帧率 (段边界 frame→秒)
        padding: 左右钳制留白

    Returns:
        (segments, crop_x_expr):
          segments: [(start_frame, end_frame, crop_x_px), ...]
          crop_x_expr: ffmpeg crop 的 x 参数. 单段=常量字符串; 多段=if(lt(t,T),x,...)
                       嵌套表达式 (内部逗号已转义 \\, 防被当 filter 链分隔).
          无 kp 数据返回 ([], "") 由调用方 fallback 居中.
    """
    if not kp_dict:
        return [], ""
    # 归一化 key 为 int
    nk = {}
    for k, v in kp_dict.items():
        try:
            nk[int(k)] = v
        except (ValueError, TypeError):
            continue
    if not nk:
        return [], ""
    total_frames = max(nk.keys()) + 1

    cxs, torso_ws = _per_frame_lead_cx(nk, total_frames)
    # 平滑: 滚动中位数 (~3s 窗) 抑制单帧错检 (领操人转背身时"最大体型人"可能瞬跳到旁人),
    # 只让持续位移(机位切换/领操人横移)触发分段, 不让背身抖动触发.
    smooth_win = max(60, int(round(3.0 * (fps or 30.0))))
    half = smooth_win // 2
    smoothed = np.zeros_like(cxs)
    for i in range(total_frames):
        lo, hi = max(0, i - half), min(total_frames, i + half)
        smoothed[i] = float(np.median(cxs[lo:hi]))
    cxs = smoothed
    segs = _detect_crop_segments(cxs)

    # 物理 cx 范围 (crop 窗口能放下的 cx 边界)
    min_cx = (crop_w / 2 + padding) / frame_w
    max_cx = 1.0 - (crop_w / 2 + padding) / frame_w

    segments: List[Tuple[int, int, int]] = []
    for (s, e) in segs:
        target = _seg_target_cx(cxs, torso_ws, s, e)
        target = max(min_cx, min(target, max_cx))
        cx_px = int(round(target * frame_w - crop_w / 2))
        cx_px = max(padding, min(cx_px, frame_w - crop_w - padding))
        segments.append((s, e, cx_px))

    # 合并 crop_x 相近 (<20px) 的相邻段, 缩短表达式 (相邻同值段无视觉差异)
    dedup: List[Tuple[int, int, int]] = []
    for seg in segments:
        if dedup and abs(seg[2] - dedup[-1][2]) < 20:
            dedup[-1] = (dedup[-1][0], seg[1], dedup[-1][2])
        else:
            dedup.append(seg)
    segments = dedup

    # ffmpeg crop x 表达式
    if len(segments) <= 1:
        crop_x_expr = str(segments[0][2]) if segments else ""
    else:
        # 从末段往前嵌套: if(lt(t, T_{i+1}), x_i, <后续>)
        # T_{i+1} = 下一段起始帧/fps = 段 i 与 i+1 的边界时间
        expr = str(segments[-1][2])
        for i in range(len(segments) - 2, -1, -1):
            next_start = segments[i + 1][0]
            T = next_start / fps if fps > 0 else 0.0
            expr = f"if(lt(t,{T:.3f}),{segments[i][2]},{expr})"
        # 转义表达式内逗号 → \, (否则 ffmpeg filtergraph 会当成 filter 链分隔)
        crop_x_expr = expr.replace(",", "\\,")
    return segments, crop_x_expr


# ── 2. profile → overlay filters ────────────────────────

# 沿用 39_shorts.py 成熟版滤镜链 (含诗词 + coach_profiles 映射 + 渐显渐隐).
# 不在 short_vertical 里重写滤镜, 复用原版保证视觉一致.
# 注: 抖音 profile = 用同一套 opening (诗词 + 教练英文/中文标题), 不加 CTA
#      yt_shorts profile = opening + CTA 片尾
from stages import shorts_legacy_filters as _legacy


def get_overlay_filters(profile: str, coach: str = "",
                        duration: Optional[float] = 30) -> str:
    """根据 profile 返回 drawtext overlay filter 字符串 (逗号分隔).

    复用 39_shorts.py 原版 _opening_overlay_filter (含诗词+渐显渐隐).
    profile 差异:
        yt_shorts: opening + CTA 片尾
        douyin:    opening (不加 CTA, 抖音不要订阅引导)

    Args:
        profile: 'yt_shorts' 或 'douyin'
        coach: 教练名 (用于 coach_profiles 查表)
        duration: 视频时长. None 表示完整版 (抖音默认).

    Returns:
        完整 filter 字符串 (可直接喂 ffmpeg filter_complex)
    """
    if profile not in ("yt_shorts", "douyin"):
        raise ValueError(f"unknown profile: {profile!r} (expected 'yt_shorts' or 'douyin')")

    dur = duration if duration else 30.0
    parts = [_legacy._opening_overlay_filter(coach or "", dur)]
    if profile == "yt_shorts":
        parts.append(_legacy._ending_cta_filter(dur))
    return ",".join(parts)


# ── 3. intro_outro 时长探测 ────────────────────────────


def resolve_intro_skip(intro_path: Optional[str] = None,
                       outro_path: Optional[str] = None,
                       intro_seconds: Optional[float] = None) -> float:
    """算出 -ss 跳过的秒数 (跳过宽屏 intro 时间段).

    优先级:
        1. 显式 intro_seconds
        2. ffprobe intro_path
        3. ffprobe outro_path (fallback)
        4. 默认 5.0 秒

    Returns:
        跳过的秒数 (>=0)
    """
    if intro_seconds is not None:
        return float(intro_seconds)

    for p in (intro_path, outro_path):
        if p and os.path.exists(str(p)):
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries",
                     "format=duration", "-of", "default=nw=1:nk=1",
                     str(p)],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0 and r.stdout.strip():
                    dur = float(r.stdout.strip())
                    if dur > 0:
                        return dur
            except Exception:
                pass

    return 5.0  # 宽屏常见 intro 时长 fallback


def _get_duration(video_path: str) -> float:
    """ffprobe 视频时长(秒). 失败返 0."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


def _get_fps(video_path: str) -> float:
    """ffprobe 视频帧率. 失败返 0. 形如 "30/1" 或 "30000/1001"."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            s = r.stdout.strip()
            if "/" in s:
                num, den = s.split("/")
                d = float(den)
                return float(num) / d if d else float(num)
            return float(s)
    except Exception:
        pass
    return 0.0


# ── 4. make_vertical 主入口 ────────────────────────────

# 沿用 39_shorts.py 的 FFMPEG 路径策略: 优先 ~/ffmpeg, 后 winget
_FFMPEG_CANDIDATES = [
    r"C:\Users\18091\ffmpeg\ffmpeg.exe",
    "ffmpeg",
]


def _resolve_ffmpeg() -> str:
    for p in _FFMPEG_CANDIDATES:
        try:
            r = subprocess.run([p, "-version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return p
        except Exception:
            continue
    return "ffmpeg"  # fallback


FFMPEG = _resolve_ffmpeg()


def make_vertical(src_path: str, output_dir: str, profile: str,
                  keypoints_file: Optional[str] = None,
                  duration: Optional[float] = 30,
                  coach: str = "",
                  audio_src: Optional[str] = None,
                  intro_path: Optional[str] = None,
                  outro_path: Optional[str] = None,
                  intro_seconds: Optional[float] = None,
                  overwrite: bool = True) -> Optional[str]:
    """单入口生成 9:16 竖版 (抖音或 YouTube Shorts).

    设计 (2026-06-27 重构):
      - 复用 youtube 宽屏 final_path, 不再独立跑 douyin preset
      - profile 决定 overlay 风格 (yt_shorts 英文 CTA / douyin 中文)
      - duration=None → 完整版 (抖音默认)
      - duration=30 → Shorts 截前 30 秒
      - cx 自适应裁切 (前 60 帧 cx 中位数 → 静态 crop_x)
      - -ss 跳过宽屏 intro 时间段

    Args:
        src_path: 1920x1080 宽屏视频 (final_path)
        output_dir: 输出目录
        profile: 'yt_shorts' 或 'douyin'
        keypoints_file: YOLO/BlazePose 关键点 JSON (可空, 空则居中裁切)
        duration: 输出时长(秒). None=完整. 默认 30.
        coach: 教练名 (用于片头 overlay)
        audio_src: 音频源 (默认同 src_path)
        intro_path: 宽屏 intro 视频路径, 用于探测 -ss 跳过时长
        intro_seconds: 显式 intro 时长, 优先于 intro_path
        overwrite: True=覆盖已有产物

    Returns:
        输出 mp4 路径, 失败 None
    """
    src_path = str(src_path)
    output_dir = str(output_dir)
    src_stem = Path(src_path).stem

    # 1. 算 crop_x (逐段跟随领操人, 治合并视频第二段领操人被裁出画面)
    #    2026-07-01: 旧逻辑只取前 60 帧 cx 中位数 → 单一静态 crop_x, 合并视频第二段
    #    领操人移位/背身就被裁出. 改逐段: 每帧最大体型人 cx + v21 分段 + 段内中位数.
    fps = _get_fps(src_path) or 30.0
    fallback_x = (DEFAULT_FRAME_W - DEFAULT_CROP_W) // 2  # 默认居中
    crop_x = fallback_x                                  # 用于日志/兜底
    crop_x_expr = str(fallback_x)                        # ffmpeg crop 的 x 参数
    crop_segments: List[Tuple[int, int, int]] = []
    if keypoints_file and os.path.exists(str(keypoints_file)):
        try:
            with open(keypoints_file, encoding="utf-8") as f:
                kp_data = json.load(f)
            kp_dict = kp_data.get("keypoints", kp_data)
            crop_segments, crop_x_expr = compute_crop_x_segments(
                kp_dict, frame_w=DEFAULT_FRAME_W,
                crop_w=DEFAULT_CROP_W, fps=fps)
            if crop_segments:
                crop_x = crop_segments[0][2]
                if len(crop_segments) == 1:
                    print(f"    [crop] 单段 crop_x={crop_x} (全片 cx 中位数)")
                else:
                    seg_log = " ".join(f"[{s}-{e}]={x}" for s, e, x in crop_segments)
                    print(f"    [crop] 逐段 crop_x ({len(crop_segments)}段, fps={fps:.2f}): {seg_log}")
            else:
                print(f"    [crop] 无 kp 数据, fallback 居中 crop_x={crop_x}")
        except Exception as e:
            print(f"    [crop] kp 解析失败, fallback 居中: {e}")

    # 2. -ss 跳过宽屏 intro
    skip = resolve_intro_skip(intro_path=intro_path,
                              intro_seconds=intro_seconds)
    print(f"    [skip] -ss {skip:.2f}s (跳过宽屏 intro)")

    # 3. duration 处理 (2026-06-27 修复:
    #    抖音完整版也要截掉 outro, 不然片尾调出 5s 出来
    #    yt_shorts 在 duration 小于总长时也要 -t, 不然会包含 outro
    outro_dur = resolve_intro_skip(intro_path=None, outro_path=outro_path)
    if duration is None:
        # 抖音完整版: -t = total - outro_dur (intro 已 -ss 跳过)
        total_dur = _get_duration(src_path)
        if total_dur > 0:
            t_dur = max(1.0, total_dur - skip - outro_dur)
            t_opt = ["-t", str(t_dur)]
        else:
            t_opt = []
        out_name = f"{src_stem}_douyin.mp4"
    else:
        t_opt = ["-t", str(duration)]
        out_name = (f"{src_stem}_yt_shorts.mp4"
                    if profile == "yt_shorts"
                    else f"{src_stem}_douyin_{int(duration)}s.mp4")

    out_path = os.path.join(output_dir, out_name)
    if os.path.exists(out_path) and not overwrite:
        print(f"    跳过: {out_name} 已存在")
        return out_path

    # 4. crop+scale+pad 算法
    # 2026-06-28: 改用 1920x1080 居中裁 + scale 到 1080x1920 (放大到竖版全屏)
    #             之前算法: crop=608:1080 + scale 不变 (608x1080 居中,左右黑边)
    #             → 画面被压到中间 608 宽,左右各 236 黑边,人物显示小
    #    正确算法: crop 中心 1080:1080 (1920x1080 的正中间 1:1) → scale 1080:1920
    #             → 视频填满 1080x1920 全屏,无黑边,人物显示 18% 大
    #    视频 1920x1080 → 居中裁 1080x1080 (cx 偏移) → scale 1080:1920
    #    注意: scale 1080:1920 会把正方形拉伸到 9:16,人物会变窄高
    #    为了不变形,改成 crop 1080:1920 等比裁 (从 1920x1080 裁出 9:16 是不可能的:
    #    1920:1080 = 16:9, 9:16 需要 607.5:1080 = 608:1080)
    #    所以: crop=608:1080 保持, scale=1080:1920 (把 608x1080 拉伸到 9:16 1080x1920)
    #    → 视频填满,但人物水平被拉宽 (608→1080 = 1.78 倍) -- 这跟抖音/YouTube 9:16 算法一致
    crop_vf = (f"crop={DEFAULT_CROP_W}:1080:{crop_x_expr}:0,"
               f"scale=1080:1920:flags=lanczos")

    # 5. 2026-06-28: 用 3 段 ffmpeg + concat, 避开 ffmpeg 8.1 的 enable=between + audio input bug
    #    段1 (0~3.5s): 视频 + opening PNG 整段 overlay (无 enable 表达式)
    #    段2 (3.5~cta_start): 视频, 无 overlay
    #    段3 (cta_start~end): 视频 + cta PNG 整段 overlay (无 enable 表达式)
    #    3 段 concat 起来, 音频从原片截取
    from stages.render_short_overlay import render_opening, render_cta

    opening_png = os.path.join(output_dir, f"_opening_overlay_{profile}_{src_stem}.png")
    render_opening(opening_png, coach=coach, duration=duration or 30.0)

    cta_png = os.path.join(output_dir, f"_cta_overlay_{profile}_{src_stem}.png")
    render_cta(cta_png)

    # 段时长计算 (实际编码时长)
    # 2026-06-29 BUGFIX: douyin (duration=None) 之前 fallback 30.0 → 抖音输出 30s 且字节与 yt_shorts 相同.
    #             281-289 行算的 t_opt/t_dur 是死代码, 从没被实际编码步骤用到. 现在用完整时长.
    if duration is None:
        raw_total = _get_duration(src_path)
        total_dur = max(1.0, raw_total - skip - outro_dur) if raw_total > 0 else 30.0
    else:
        total_dur = float(duration)
    opening_end = min(6.5, total_dur - 4.0) if total_dur > 4.0 else 0.0  # 2026-06-29: 3.5→6.5 诗词多显3s 方便阅读
    cta_dur = min(4.0, total_dur - opening_end) if total_dur > opening_end + 4.0 else 0.0
    middle_dur = max(0.0, total_dur - opening_end - cta_dur)

    # 中间产物路径
    part1_path = os.path.join(output_dir, f"_part_opening_{profile}_{src_stem}.mp4")
    part2_path = os.path.join(output_dir, f"_part_middle_{profile}_{src_stem}.mp4")
    part3_path = os.path.join(output_dir, f"_part_cta_{profile}_{src_stem}.mp4")
    concat_list = os.path.join(output_dir, f"_concat_list_{profile}_{src_stem}.txt")

    audio_input = audio_src or src_path

    def _build_crop_only_cmd(in_path, out_p, ss, t_len):
        """只 crop+scale+pad, 无 overlay, 输出无音频
        2026-06-28: 用 setpts=PTS-STARTPTS 让段首帧 PTS=0, 否则 filter_complex concat
                    会因为段1 start_pts=0.033 丢失首段
        """
        if t_len <= 0:
            return None
        # 在 filter_complex 加 setpts=PTS-STARTPTS 重置时间戳
        crop_reset = (f"{crop_vf},setpts=PTS-STARTPTS[vout]")
        cmd = [
            FFMPEG, "-y",
            "-ss", str(ss), "-t", f"{t_len:.3f}",
            "-i", in_path,
            "-filter_complex", crop_reset,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an",
            out_p,
        ]
        return cmd

    def _build_overlay_cmd(in_path, png_path, out_p, ss, t_len):
        """crop+scale+pad + 整段 overlay 一张 PNG, 无音频
        2026-06-28: 加 setpts=PTS-STARTPTS 重置时间戳
        """
        if t_len <= 0 or not os.path.exists(png_path):
            return None
        overlay_vf = (f"[0:v]{crop_vf},setpts=PTS-STARTPTS[bg];"
                      f"[1:v]format=rgba[ol];"
                      f"[bg][ol]overlay=0:0[vout]")
        cmd = [
            FFMPEG, "-y",
            "-ss", str(ss), "-t", f"{t_len:.3f}",
            "-i", in_path,
            "-i", png_path,
            "-filter_complex", overlay_vf,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an",
            out_p,
        ]
        return cmd

    # 段 1: 开头 (0~3.5s) + opening 诗词
    # 2026-06-28: 改 2 步法避开 ffmpeg 8.1 audio+overlay enable bug
    #    step1: 单 ffmpeg 命令, 视频 + opening + cta overlay, **不带 audio input** (用 -an)
    #    step2: 单独 ffmpeg 命令, 把 step1 的视频 + 原音频合并
    #    这样 enable=between 在 3 input (视频+opening+cta, 无 audio) 环境下不会触发 ffmpeg 8.1 bug

    # step1: 视频 crop+scale+pad + opening overlay + cta overlay
    video_only_path = os.path.join(output_dir, f"_video_only_{profile}_{src_stem}.mp4")

    # 决定 3 个 overlay 是否启用 (duration 太短就别画了)
    use_opening = opening_end > 0 and os.path.exists(opening_png)
    # 2026-06-29: 抖音完整版不加英文 CTA (CLAUDE.md 设计: douyin 无 CTA, 中文平台不放 SUBSCRIBE)
    use_cta = (cta_dur > 0 and os.path.exists(cta_png)
               and profile != "douyin")

    if use_opening and use_cta:
        # 3 input: video, opening, cta
        step1_vf = (f"[0:v]{crop_vf},setpts=PTS-STARTPTS[bg];"
                    f"[1:v]format=rgba[op];"
                    f"[2:v]format=rgba[cta];"
                    f"[bg][op]overlay=0:0:enable='between(t,0,{opening_end})'[v1];"
                    f"[v1][cta]overlay=0:0:enable='between(t,{opening_end + middle_dur},{total_dur})'[vout]")
        step1_inputs = ["-i", src_path, "-loop", "1", "-i", opening_png, "-loop", "1", "-i", cta_png]
    elif use_opening:
        step1_vf = (f"[0:v]{crop_vf},setpts=PTS-STARTPTS[bg];"
                    f"[1:v]format=rgba[op];"
                    f"[bg][op]overlay=0:0:enable='between(t,0,{opening_end})'[vout]")
        step1_inputs = ["-i", src_path, "-loop", "1", "-i", opening_png]
    elif use_cta:
        step1_vf = (f"[0:v]{crop_vf},setpts=PTS-STARTPTS[bg];"
                    f"[1:v]format=rgba[cta];"
                    f"[bg][cta]overlay=0:0:enable='between(t,{opening_end + middle_dur},{total_dur})'[vout]")
        step1_inputs = ["-i", src_path, "-loop", "1", "-i", cta_png]
    else:
        # 无 overlay, 只 crop+scale+pad
        step1_vf = f"[0:v]{crop_vf},setpts=PTS-STARTPTS[vout]"
        step1_inputs = ["-i", src_path]

    print(f"    [1/2] 视频 + overlay (无 audio) -> _video_only_{profile}_{src_stem}.mp4")
    step1_cmd = [
        FFMPEG, "-y",
        "-ss", str(skip), "-t", f"{total_dur:.3f}",
    ] + step1_inputs + [
        "-filter_complex", step1_vf,
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an",
        "-t", f"{total_dur:.3f}",
        video_only_path,
    ]
    r = subprocess.run(step1_cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    if r.returncode != 0:
        print(f"    [FFMPEG FAIL] step1: {r.stderr[-500:]}")
        return None

    # step2: 合并 audio (1 video input + 1 audio input, 简单安全)
    print(f"    [2/2] 合并 audio + 编码")
    audio_input = audio_src or src_path
    step2_cmd = [
        FFMPEG, "-y",
        "-i", video_only_path,
        "-ss", str(skip), "-t", f"{total_dur:.3f}",
        "-i", audio_input,
        "-map", "0:v", "-map", "1:a:0",
        "-c:v", "copy",  # 视频流直接 copy (step1 已经编码好)
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        out_path,
    ]
    profile_label = "YouTube Shorts" if profile == "yt_shorts" else "抖音竖版"
    print(f"    [{profile_label}] {out_name}  crop_x={crop_x}  skip={skip:.1f}s"
          + (f"  dur={duration}s" if duration else "  full"))
    r = subprocess.run(step2_cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)

    if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        size_mb = os.path.getsize(out_path) // 1024 // 1024
        print(f"    [OK] {Path(out_path).name} ({size_mb}MB)")
    else:
        print(f"    [FFMPEG FAIL] step2 ret={r.returncode}")
        print(f"    stderr: {r.stderr[-2000:]}")
        return None

    # 清理 video_only 中间产物
    try:
        os.remove(video_only_path)
    except OSError:
        pass
    return out_path

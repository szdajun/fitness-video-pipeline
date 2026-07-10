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


# ── 1c. 画中画小窗位置 (竖屏右上, 避开领操人) ──────────
# 2026-07-07: 竖屏 9:16 从 16:9 裁切, 左右画面丢失. 在竖屏右上放一个 16:9 全景小窗
#   补整体场景. 小窗 = 换脸后横屏 (face_swap_path) 缩成 16:9, 诗词结束后全程常驻.
#   位置不写死: pose keypoints 算领操人上半身 bbox 在竖屏的分布, 右上贴边扫 y, 找
#   最小 y (最靠上) 使"领操人覆盖小窗的帧占比 < 阈值" → 不挡领操人.

# 小窗避让用上半身关键点 (小窗在上方, 主要避让举手)
_PIP_UPPER_KP_IDX = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 23, 24]


def compute_pip_rect(kp_dict: dict,
                     crop_segments: List[Tuple[int, int, int]],
                     frame_w: int = DEFAULT_FRAME_W,
                     frame_h: int = 1080,
                     crop_w: int = DEFAULT_CROP_W,
                     vert_w: int = 1080,
                     vert_h: int = 1920,
                     target_w: int = 600,
                     margin: int = 24,
                     overlap_thr: float = 0.08,
                     area_frac: float = 0.05,
                     y_step: int = 8) -> Tuple[int, int, int, int]:
    """算画中画小窗在竖屏 (vert_w×vert_h) 的位置 (右上, 避开领操人).

    16:9 小窗宽 target_w (高 = round(target_w×9/16)). x 贴右边; y 从上往下扫, 取最小
    y0 使"领操人上半身 bbox 与小窗重叠 (重叠面积 > area_frac×小窗面积) 的帧占比 <
    overlap_thr". 领操人 = 每帧最大体型人 (与 _per_frame_lead_cx 一致); bbox 用上半身
    kp (头/肩/肘/腕/髋), 因小窗在上方主要避让举手.

    坐标映射 (kp 归一化相对 frame_w×frame_h):
        竖屏 x = (kp_x×frame_w - crop_x) × vert_w / crop_w
        竖屏 y =  kp_y×frame_h            × vert_h / frame_h

    Args:
        kp_dict: {frame_idx: [person, ...]}, person=33×3 [x,y,conf] 归一化
        crop_segments: [(start, end, crop_x_px), ...] 来自 compute_crop_x_segments
        target_w: 小窗宽 (像素), 默认 480 (竖屏 1080 的 44%)
        overlap_thr: 允许的领操人覆盖小窗帧占比上限 (默认 0.08)
        area_frac: 重叠面积阈值占小窗面积比 (默认 0.05, 小于不算"挡住")

    Returns:
        (x, y, w, h) 竖屏像素坐标. 无 kp/crop_segments → fallback 固定右上.
    """
    pip_h = max(1, round(target_w * 9 / 16))
    fallback = (vert_w - target_w - margin, margin, target_w, pip_h)
    if not kp_dict or not crop_segments:
        return fallback

    nk = {}
    for k, v in kp_dict.items():
        try:
            nk[int(k)] = v
        except (ValueError, TypeError):
            continue
    if not nk:
        return fallback

    upper = _PIP_UPPER_KP_IDX
    sx_v = vert_w / crop_w        # 横屏 crop 列 → 竖屏 x
    sy_v = vert_h / frame_h       # 横屏 y → 竖屏 y

    # 逐段逐帧算领操人上半身 bbox (竖屏像素). 段内 crop_x 固定.
    boxes: List[Tuple[float, float, float, float]] = []
    for (s, e, crop_x) in crop_segments:
        for fi in range(s, e):
            pose = nk.get(fi)
            if not pose:
                continue
            best_size, best_box = -1.0, None
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
                xs_all, ys_all = kps[vis, 0], kps[vis, 1]
                size = float((xs_all.max() - xs_all.min()) *
                             (ys_all.max() - ys_all.min()))
                if size <= best_size:
                    continue
                # 上半身 bbox (小窗在上方, 避让举手为主); 上半身不可见 fallback 全身
                up_mask = vis[upper]
                chosen = kps[upper][up_mask] if up_mask.sum() >= 2 else kps[vis]
                hx = np.asarray(chosen[:, 0] * frame_w, dtype=np.float32)
                hy = np.asarray(chosen[:, 1] * frame_h, dtype=np.float32)
                # 2026-07-07 背向补头: 脸 kp(0-6 鼻/眼/耳) 背向时低置信度被 vis 过滤 →
                # bbox 丢头 → PIP 压在 (后) 脑上检测不到 (用户报"背向时挡头"). 脸不可见
                # 但双肩(11,12)可见时, 从肩宽推断头位 (头在肩中点上方 ~1×肩宽, 横 ±0.5×
                # 肩宽) 补进 bbox. 脸可见时用真实脸 kp, 不触发 (现有测试假人 conf=0.95 不受影响).
                if vis[0:7].sum() < 2 and vis[11] and vis[12]:
                    sh_x = kps[[11, 12], 0] * frame_w
                    sh_y = kps[[11, 12], 1] * frame_h
                    mid_x = float(sh_x.mean())
                    mid_y = float(sh_y.mean())
                    sw = float(abs(sh_x[0] - sh_x[1]))  # 肩宽 (frame px)
                    if sw > 1.0:
                        hx = np.concatenate(
                            [hx, np.array([mid_x - sw * 0.5, mid_x + sw * 0.5],
                                          dtype=np.float32)])
                        hy = np.concatenate(
                            [hy, np.array([mid_y - sw, mid_y], dtype=np.float32)])
                vx = (hx - crop_x) * sx_v
                vy = hy * sy_v
                best_box = (float(vx.min()), float(vy.min()),
                            float(vx.max()), float(vy.max()))
            if best_box is not None:
                boxes.append(best_box)
    if not boxes:
        return fallback

    x0 = vert_w - target_w - margin
    min_overlap_area = area_frac * target_w * pip_h
    y_lo, y_hi = margin, vert_h - pip_h - margin

    # 扫 y0, 找最小 y0 使领操人覆盖小窗帧占比 < overlap_thr
    chosen_y = y_hi  # 兜底: 尽量靠下避开举手
    y0 = y_lo
    while y0 <= y_hi:
        overlap = 0
        win_x1, win_y1 = x0 + target_w, y0 + pip_h
        for (bx_min, by_min, bx_max, by_max) in boxes:
            ox = (bx_max if bx_max < win_x1 else win_x1) - (bx_min if bx_min > x0 else x0)
            if ox <= 0:
                continue
            oy = (by_max if by_max < win_y1 else win_y1) - (by_min if by_min > y0 else y0)
            if oy > 0 and ox * oy > min_overlap_area:
                overlap += 1
        if overlap / len(boxes) < overlap_thr:
            chosen_y = y0
            break
        y0 += y_step

    # 整列都被占 (领操人频繁举手到右上): 缩小 target_w 重试一次
    if chosen_y >= y_hi and target_w > 360:
        return compute_pip_rect(
            kp_dict, crop_segments, frame_w, frame_h, crop_w,
            vert_w, vert_h, int(target_w * 0.8), margin,
            overlap_thr, area_frac, y_step)

    return (x0, chosen_y, target_w, pip_h)


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


# ── 2026-07-10: 竖屏源端到端通路 — 探测 src 真实宽高 ──────────────────────────
def _get_video_size(video_path: str) -> Tuple[int, int]:
    """ffprobe 探测 (w, h), 失败返 (0, 0). 抄 stages/07_export.py:235-243 模板."""
    try:
        # 复用 _resolve_ffmpeg 的 ffmpeg 候选路径思路找 ffprobe.exe
        for cand in (FFMPEG.replace("ffmpeg.exe", "ffprobe.exe"),
                     r"C:\Users\18091\ffmpeg\ffprobe.exe",
                     "ffprobe"):
            r = subprocess.run(
                [cand, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", str(video_path)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                w, h = r.stdout.strip().split(",")[:2]
                return int(w), int(h)
    except Exception:
        pass
    return 0, 0


def compute_hook_window(
    kp_dict: dict,
    crop_segments: List[Tuple[int, int, int]],
    fps: float,
    total_dur: float,
    hook_dur: float,
    skip_sec: float = 0.0,
    frame_w: int = DEFAULT_FRAME_W,
    crop_w: int = DEFAULT_CROP_W,
    padding: int = CROP_PADDING,
    exclude_head_frac: float = 0.10,
    exclude_tail_frac: float = 0.10,
    min_total_dur: float = 10.0,
) -> Optional[Tuple[float, int]]:
    """选全片最燃的 hook_dur 秒窗做"高燃预览开场" (2026-07-07).

    返回 (hook_start_sec, hook_crop_x) 或 None:
      hook_start_sec: 正片相对时间 (0~total_dur), 给 step0 `-ss {skip+hook_start}` 用
      hook_crop_x:    hook 窗口起点所在 crop_segment 的 crop_x (像素, 静态常量)

    算法:
      1. 逐帧领操人 (最大体型 person) motion = 相邻帧可见关键点 (conf>0.3) 位移均值
         (复用 35_intensity_burst.py:58-78 的 motion 食谱, 但只取领操人, 非全 person 累加)
      2. 滑动窗 (hook_dur*fps 帧) 在 [head_frac .. (1-tail_frac)-hook_dur] 扫,
         取 mean motion 最大起点
      3. 排除首尾各 10% (片头诗词区 + 片尾噪声孤峰, 如李刚1 的 105s)
    滑动窗自身抗单帧尖刺 (105s 孤峰在 4s 窗=120 帧贡献 0.0036 可忽略), 不需额外 conf 过滤.
    """
    # 早返守卫
    if not kp_dict or total_dur < min_total_dur or hook_dur <= 0:
        return None
    usable = total_dur * (1.0 - exclude_head_frac - exclude_tail_frac)
    if hook_dur >= usable:
        return None

    fps = float(fps) or 30.0
    n_wf = int(round(total_dur * fps))      # workout (正片相对) 帧数
    win = max(1, int(round(hook_dur * fps)))
    src0 = int(round(skip_sec * fps))       # 正片帧 0 对应的源帧

    def _lead(persons):
        """选最大体型 person (bbox 面积最大), 领操人启发式."""
        best, best_sp = None, -1.0
        for p in persons:
            xs = [c[0] for c in p if len(c) >= 3 and c[2] > 0.3]
            ys = [c[1] for c in p if len(c) >= 3 and c[2] > 0.3]
            if len(xs) < 6:
                continue
            sp = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if sp > best_sp:
                best_sp, best = sp, p
        return best

    # 逐 workout 帧 motion (源帧空间查 kp, 累积正片相对 motion)
    motion = {}
    prev_lead = None
    for wf in range(n_wf):
        src_f = src0 + wf
        persons = kp_dict.get(src_f)
        if persons is None:
            persons = kp_dict.get(str(src_f))
        if not persons:
            prev_lead = None
            continue
        lead = _lead(persons)
        if lead is None or prev_lead is None:
            prev_lead = lead
            continue
        disp = [(b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
                for a, b in zip(prev_lead, lead)
                if len(a) >= 3 and len(b) >= 3 and a[2] > 0.3 and b[2] > 0.3]
        if disp:
            motion[wf] = sum(d ** 0.5 for d in disp) / len(disp)
        prev_lead = lead

    if not motion:
        return None

    # 滑动窗找最优起点 (排除首尾)
    lo = int(exclude_head_frac * n_wf)
    hi = int(total_dur * (1.0 - exclude_tail_frac) * fps) - win
    best_sf, best_score = lo, -1.0
    for sf in range(lo, max(lo, hi + 1)):
        win_vals = [motion.get(f, 0.0) for f in range(sf, sf + win)]
        if not win_vals:
            continue
        score = sum(win_vals) / len(win_vals)
        if score > best_score:
            best_score, best_sf = score, sf

    hook_start_sec = round(best_sf / fps, 3)

    # hook_crop_x = hook 起点源帧落在的 crop_segment 的 crop_x
    src_at_hook = src0 + best_sf
    hook_crop_x = (crop_segments[0][2] if crop_segments
                   else (frame_w - crop_w) // 2)
    for s, e, x in crop_segments:
        if s <= src_at_hook <= e:
            hook_crop_x = x
            break
    hook_crop_x = int(max(padding, min(frame_w - crop_w - padding, hook_crop_x)))

    print(f"    [hook] 预览窗 {win / fps:.1f}s @ 正片 {hook_start_sec:.1f}s "
          f"(crop_x={hook_crop_x}, mean motion={best_score:.4f})")
    return (hook_start_sec, hook_crop_x)


def make_vertical(src_path: str, output_dir: str, profile: str,
                  keypoints_file: Optional[str] = None,
                  duration: Optional[float] = 30,
                  coach: str = "",
                  audio_src: Optional[str] = None,
                  intro_path: Optional[str] = None,
                  outro_path: Optional[str] = None,
                  intro_seconds: Optional[float] = None,
                  pip_src: Optional[str] = None,
                  pip_enabled: bool = True,
                  pip_target_w: int = 600,
                  hook_enabled: bool = False,
                  hook_dur: float = 4.0,
                  overwrite: bool = True,
                  # 2026-07-10: 竖屏源端到端通路
                  is_native_vertical: bool = False,
                  src_w: Optional[int] = None,
                  src_h: Optional[int] = None,
                  force_intro_skip: Optional[float] = None) -> Optional[str]:
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
        pip_src: 画中画源 (换脸后横屏 16:9). None=不加小窗. 诗词结束后全程常驻右上
        pip_enabled: 是否启用画中画小窗 (默认 True)
        pip_target_w: 小窗宽 (像素), 默认 600 (竖屏 1080 的 56%)
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
    pip_rect: Optional[Tuple[int, int, int, int]] = None  # 画中画 (x,y,w,h) 竖屏坐标
    kp_dict: dict = {}                                     # 关键点 (hook 窗口选择用)
    hook_window: Optional[Tuple[float, int]] = None        # (hook_start_sec, hook_crop_x)
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
            # 画中画小窗位置 (复用已解析的 kp_dict + crop_segments, 不重读文件)
            if pip_enabled and pip_src and crop_segments:
                try:
                    pip_rect = compute_pip_rect(kp_dict, crop_segments,
                                                target_w=pip_target_w)
                    print(f"    [pip] 小窗 {pip_rect[2]}x{pip_rect[3]} "
                          f"at ({pip_rect[0]},{pip_rect[1]}) 避开领操人")
                except Exception as pe:
                    print(f"    [pip] 位置计算失败, 跳过: {pe}")
        except Exception as e:
            print(f"    [crop] kp 解析失败, fallback 居中: {e}")

    # 2. -ss 跳过宽屏 intro
    skip = resolve_intro_skip(intro_path=intro_path,
                              intro_seconds=intro_seconds if force_intro_skip is None else force_intro_skip)
    print(f"    [skip] -ss {skip:.2f}s (跳过宽屏 intro)")

    # 3. duration 处理 (2026-06-27 修复:
    #    抖音完整版也要截掉 outro, 不然片尾调出 5s 出来
    #    yt_shorts 在 duration 小于总长时也要 -t, 不然会包含 outro
    # 2026-07-10: 竖源无片尾, force_intro_skip=0 时 outro_dur 也强制 0
    outro_dur = (0.0 if force_intro_skip == 0.0
                 else resolve_intro_skip(intro_path=None, outro_path=outro_path))
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

    # 2026-07-10: 竖屏源 (9:16 native) 不裁, 只 scale 到 1080x1920
    if is_native_vertical and src_h and src_w and src_h > src_w:
        crop_vf = "scale=1080:1920:flags=lanczos:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        crop_x_expr = "0"
        # 跳过 cx 跟领操 + PIP (竖源本来就全屏)
        pip_rect = None
        pip_enabled = False
        pip_src = None
        print(f"    [native-vertical] {src_w}x{src_h} → scale 1080x1920, 跳过 cx/PIP")

    # 5. 2026-06-28: 用 3 段 ffmpeg + concat, 避开 ffmpeg 8.1 的 enable=between + audio input bug
    #    段1 (0~3.5s): 视频 + opening PNG 整段 overlay (无 enable 表达式)
    #    段2 (3.5~cta_start): 视频, 无 overlay
    #    段3 (cta_start~end): 视频 + cta PNG 整段 overlay (无 enable 表达式)
    #    3 段 concat 起来, 音频从原片截取
    from stages.render_short_overlay import render_opening, render_cta, render_preview

    opening_png = os.path.join(output_dir, f"_opening_overlay_{profile}_{src_stem}.png")
    render_opening(opening_png, coach=coach, duration=duration or 30.0)

    cta_png = os.path.join(output_dir, f"_cta_overlay_{profile}_{src_stem}.png")
    render_cta(cta_png)

    hook_png = os.path.join(output_dir, f"_hook_overlay_{profile}_{src_stem}.png")
    # 注: hook_window 在下面 (line 846) 才算出, render_preview 不能放这里 (此时仍 None → PNG 不生成)
    #     真正的渲染移到 step0 块内 (hook_window 已就绪后)

    # 段时长计算 (实际编码时长)
    # 2026-06-29 BUGFIX: douyin (duration=None) 之前 fallback 30.0 → 抖音输出 30s 且字节与 yt_shorts 相同.
    #             281-289 行算的 t_opt/t_dur 是死代码, 从没被实际编码步骤用到. 现在用完整时长.
    if duration is None:
        raw_total = _get_duration(src_path)
        total_dur = max(1.0, raw_total - skip - outro_dur) if raw_total > 0 else 30.0
    else:
        total_dur = float(duration)
    # 高燃预览开场 (2026-07-07): 选全片最燃 hook_dur 秒窗, 拼到竖版最前.
    # yt_shorts + douyin 都加 (2026-07-07: 用户要抖音版也有爆燃预警片段).
    # 2026-07-10: 竖屏源用户拍板"幅面小元素不能堆", hook 关掉; 抖音 9:16 也保留 hook (用户拍板保留)
    if hook_enabled and profile in ("yt_shorts", "douyin") and kp_dict:
        # 2026-07-10: 竖源 hook 关掉 (user 拍板 9:16 元素精简)
        if is_native_vertical:
            print("    [native-vertical] hook 已关 (9:16 幅面小, 元素精简)")
            hook_enabled = False
        try:
            hook_window = compute_hook_window(
                kp_dict, crop_segments, fps=fps,
                total_dur=total_dur, hook_dur=hook_dur, skip_sec=skip)
        except Exception as he:
            print(f"    [hook] 窗口计算失败, 跳过: {he}")
            hook_window = None
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

    # 画中画小窗 (2026-07-07): 换脸后横屏缩 16:9, 诗词结束后全程常驻右上
    if pip_enabled and pip_rect and pip_src and os.path.exists(str(pip_src)):
        px, py, pw, ph = pip_rect
        pip_idx = step1_inputs.count("-i")  # 新 input 的 0-based index
        # 小窗滤镜: scale 到小窗尺寸 + 3px 白边框 (和主画面区分, 不抢眼)
        pip_filter = (f"[{pip_idx}:v]scale={pw}:{ph}:flags=lanczos,setsar=1,"
                      f"drawbox=w=iw:h=ih:color=white@0.9:t=3[pip];")
        # 现有链路输出 [vout] 改名 [vpre], 追加 pip overlay (enable: opening 后到结尾)
        step1_vf = (pip_filter
                    + step1_vf.replace("[vout]", "[vpre]")
                    + f";[vpre][pip]overlay={px}:{py}"
                      f":enable='between(t,{opening_end:.3f},{total_dur:.3f})'[vout]")
        # pip input seek: face_swap_path 无片头/片尾 (workout-only; 实测 179.6s vs
        #   final 188.6s, 差正好 intro 4s + outro 5s). 不能套用主视频的 skip(=intro 4s) —
        #   否则小窗多跳 4s → PIP 比主画面提前 4s (workout[T+skip] vs workout[T]),
        #   即用户报的"画中画和主视频不同步". 仅当 pip_src 就是 src(final_path, 含片头) 时
        #   才 -ss skip; face_swap_path(workout-only) seek 0. 两路都让 PIP t=0=workout[0]
        #   对齐主画面.
        pip_is_intro_src = os.path.realpath(str(pip_src)) == os.path.realpath(str(src_path))
        pip_seek = skip if pip_is_intro_src else 0.0
        step1_inputs = (step1_inputs + ["-ss", f"{pip_seek:.3f}", "-t", f"{total_dur:.3f}",
                                        "-i", str(pip_src)])

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

    # ── hook 高燃预览 (2026-07-07): step0 编码静音预览 + step1.5 concat 到正片前 ──
    # 预览段独立编码 (静音+字幕), 正片 step1 不动, concat demuxer -c copy 零重编码.
    # 正片 step1vf 的 enable=between(t,...) 和 crop_x_expr 的 t 仍是正片相对时间,
    # concat 只改输出 PTS 改不了 step1 已 baked 的像素 → 正片节奏零偏移.
    final_video_path = video_only_path
    hook_paths_to_clean = []
    if hook_window:
        hook_start, hook_crop_x = hook_window
        # 渲染高燃预览字幕 PNG (此时 hook_window 已算出, 保证 step0 有图可叠)
        render_preview(hook_png, duration=hook_dur)
        hook_silent_path = os.path.join(
            output_dir, f"_hook_silent_{profile}_{src_stem}.mp4")
        hook_concat_list = os.path.join(
            output_dir, f"_hook_concat_{profile}_{src_stem}.txt")
        # step0: 从 src 取最燃窗 (正片内 hook_start), crop+scale+叠字幕 PNG, 静音
        ss_hook = skip + hook_start
        step0_cmd = [
            FFMPEG, "-y",
            "-ss", f"{ss_hook:.3f}", "-t", f"{hook_dur:.3f}",
            "-i", src_path,
            "-loop", "1", "-t", f"{hook_dur:.3f}", "-i", hook_png,
            "-filter_complex",
            # 2026-07-10: 竖源 hook 用 scale 替代 crop, hook_crop_x 不再用
            (f"[0:v]scale=1080:1920:flags=lanczos:force_original_aspect_ratio=decrease,"
             f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setpts=PTS-STARTPTS[bg];"
             f"[1:v]format=rgba[ol];[bg][ol]overlay=0:0[vout]")
            if is_native_vertical
            else
            (f"[0:v]crop={DEFAULT_CROP_W}:1080:{hook_crop_x}:0,"
             f"scale=1080:1920:flags=lanczos,setpts=PTS-STARTPTS[bg];"
             f"[1:v]format=rgba[ol];[bg][ol]overlay=0:0[vout]"),
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an",
            "-t", f"{hook_dur:.3f}",
            hook_silent_path,
        ]
        print(f"    [hook] step0 预览段 (静音+字幕 {hook_dur:.1f}s) "
              f"-> {Path(hook_silent_path).name}")
        r0 = subprocess.run(step0_cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=300)
        if r0.returncode != 0 or not os.path.exists(hook_silent_path):
            print(f"    [hook] step0 失败, 回退无预览: {r0.stderr[-300:]}")
            hook_window = None
        else:
            # step1.5: concat demuxer hook + 正片 (-c copy 零重编码, 两段参数一致)
            video_only_final = os.path.join(
                output_dir, f"_video_final_{profile}_{src_stem}.mp4")
            # 2026-07-07: concat demuxer 路径必须绝对 + 正斜杠 (Windows 反斜杠被当转义符,
            #  _temp\hook_test\_xxx 被 demuxer 解析成乱路径). 照 03_h2v_convert.py:222 既定模式.
            with open(hook_concat_list, "w", encoding="utf-8", newline="\n") as cf:
                cf.write(f"file '{Path(hook_silent_path).resolve().as_posix()}'\n")
                cf.write(f"file '{Path(video_only_path).resolve().as_posix()}'\n")
            step15_cmd = [
                FFMPEG, "-y", "-f", "concat", "-safe", "0",
                "-i", hook_concat_list, "-c", "copy", video_only_final,
            ]
            print(f"    [hook] step1.5 concat 预览+正片 "
                  f"-> {Path(video_only_final).name}")
            r15 = subprocess.run(step15_cmd, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=300)
            if r15.returncode != 0 or not os.path.exists(video_only_final):
                print(f"    [hook] concat 失败, 回退无预览: {r15.stderr[-300:]}")
                hook_window = None
            else:
                final_video_path = video_only_final
                hook_paths_to_clean = [hook_silent_path, video_only_final,
                                       hook_concat_list]

    # step2: 合并 audio
    # hook on: anullsrc 真 4s 静音 + 主音频 concat → 预览段静音, 正片音频对齐正片画面, 零错位.
    #   ⚠️ 2026-07-07: 旧用 adelay={ms}|{ms} 产生的 4s 前导静音被 AAC gapless 当 encoder_delay
    #   side data 在解码时整体丢弃 → 默认解码只剩 30s 主音频无静音, 主音频从 t=0 越过预览播放
    #   = 音视频错位 (违反"零错位"). anullsrc 是真实零样本, 不会被 gapless 剥 (李刚1 实测
    #   0-4s mean -91dB 真·数字静音, 全片 silence_start=0/end=4.0 干净).
    # hook off: 原逻辑 (-shortest)
    print(f"    [2/2] 合并 audio + 编码")
    audio_input = audio_src or src_path
    if hook_window:
        step2_cmd = [
            FFMPEG, "-y",
            "-i", final_video_path,                               # 0: 视频 (hook+正片 concat, 34s)
            "-ss", str(skip), "-t", f"{total_dur:.3f}",
            "-i", audio_input,                                    # 1: 正片主音频 (skip..skip+total)
            "-f", "lavfi", "-t", f"{hook_dur:.3f}",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",  # 2: 真 hook_dur 秒静音
            "-filter_complex",
            "[1:a]aresample=44100[a1];[2:a][a1]concat=n=2:v=0:a=1[a]",  # 静音+主音频拼接=hook_dur+total
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-t", f"{hook_dur + total_dur:.3f}",
            out_path,
        ]
    else:
        step2_cmd = [
            FFMPEG, "-y",
            "-i", final_video_path,
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
          + (f"  dur={duration}s" if duration else "  full")
          + ("  +hook" if hook_window else ""))
    r = subprocess.run(step2_cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)

    if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        size_mb = os.path.getsize(out_path) // 1024 // 1024
        print(f"    [OK] {Path(out_path).name} ({size_mb}MB)")
    else:
        print(f"    [FFMPEG FAIL] step2 ret={r.returncode}")
        print(f"    stderr: {r.stderr[-2000:]}")
        return None

    # 清理中间产物 (video_only + hook 三件)
    for _p in [video_only_path] + hook_paths_to_clean:
        try:
            os.remove(_p)
        except OSError:
            pass
    return out_path

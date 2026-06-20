"""H2V 裁切策略模块

提取自 stages/03_h2v_convert.py，负责人体追踪、领操人识别和帧分类策略。
"""

import numpy as np


def build_tracks(keypoints, total_frames):
    """追踪各人体，分配 track_id。

    Args:
        keypoints: dict，{frame_idx: [[person_kps], ...]}
        total_frames: 总帧数

    Returns:
        tracks: dict，{track_id: {"cx_list": [], "body_size_list": [], "count": int}}
    """
    tracks = {}
    for fi in range(total_frames):
        pose_data = keypoints.get(fi)
        if not pose_data:
            continue

        frame_detections = []
        for person_kps in pose_data:
            cx = _body_center_x(person_kps)
            body_size = _body_size_score(person_kps)
            if cx is None:
                cx = 0.5
            frame_detections.append((cx, body_size))

        assigned = set()
        for cx, body_size in frame_detections:
            best_tid = None
            best_dist = float("inf")
            for tid, trk in tracks.items():
                if tid in assigned:
                    continue
                prev_cx = np.median(trk["cx_list"]) if trk["cx_list"] else cx
                dist = abs(cx - prev_cx)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            # 距离阈值 >0.1 认为是新人 (2026-06-20 修复: 0.2 太宽, 容易把不同人混在一起)
            if best_tid is not None and best_dist < 0.1:
                tracks[best_tid]["cx_list"].append(cx)
                tracks[best_tid]["body_size_list"].append(body_size)
                tracks[best_tid]["count"] += 1
                assigned.add(best_tid)
            else:
                new_tid = len(tracks)
                tracks[new_tid] = {
                    "cx_list": [cx],
                    "body_size_list": [body_size],
                    "count": 1,
                }
                assigned.add(new_tid)

    if not tracks:
        tracks = {0: {"cx_list": [0.5], "body_size_list": [1.0], "count": total_frames}}

    return tracks


def score_track(track, first_n_frames=None):
    """领操人评分: 直接用 bbox 体型大小

    first_n_frames: 只用前 N 帧的统计量做评分（避免拼接视频后半段选错人）
                    None 表示用整段统计

    2026-06-20 修复 (v5): 之前尝试加稳定性/中心性/帧数奖励,
    但实际数据: tid=7 size=0.009 拿稳定性奖励后 score=0.046,
                tid=3 size=0.032 但 score=0.021
    奖励压倒了 size 主项, 选错人。
    现在直接用 size, 不加任何奖励。
    """
    sizes = track["body_size_list"]
    if first_n_frames is not None:
        sizes = sizes[:first_n_frames]
    if not sizes:
        return 0.0
    return float(np.mean(sizes))


def select_lead_track(tracks, first_n_frames=None):
    """从 tracks 中选择领操人（评分最高）

    first_n_frames: 只看前 N 帧的统计量选人（用于拼接视频: 用前半段选人避免背面错选）
    """
    if not tracks:
        return 0, {"cx_list": [0.5], "body_size_list": [1.0], "count": 0}
    lead_tid = max(tracks, key=lambda tid: score_track(tracks[tid], first_n_frames=first_n_frames))
    return lead_tid, tracks[lead_tid]


def classify_frame(num_people, pan_threshold=3):
    """单帧分类。

    Args:
        num_people: 检测到的人数
        pan_threshold: 多人大于等于此值时判定为全景

    Returns:
        "lead"  — 1-2人，领操人特写
        "multi" — >=pan_threshold 人数，全景
        "other" — 0人，无人帧
    """
    if num_people == 0:
        return "other"
    if num_people >= pan_threshold:
        return "multi"
    return "lead"


def classify_frames(keypoints, total_frames, pan_threshold=3):
    """逐帧分类场景类型。

    Args:
        keypoints: dict，{frame_idx: [person_kps, ...]}
        total_frames: 总帧数
        pan_threshold: 判定全景的人数阈值

    Returns:
        frame_decisions: list，按帧顺序的分类结果
        stats: dict，各类型帧数统计
    """
    frame_decisions = []
    lead_frames = other_frames = multi_frames = 0

    for fi in range(total_frames):
        pose_data = keypoints.get(fi)
        num = len(pose_data) if pose_data else 0
        decision = classify_frame(num, pan_threshold)
        frame_decisions.append(decision)

        if decision == "lead":
            lead_frames += 1
        elif decision == "other":
            other_frames += 1
        else:
            multi_frames += 1

    stats = {
        "lead": lead_frames,
        "other": other_frames,
        "multi": multi_frames,
    }
    return frame_decisions, stats


def merge_segments(decisions, min_frames):
    """合并相邻同类短片段。

    Args:
        decisions: list，按帧顺序的分类结果
        min_frames: 最短段长度，不够则合并到前一段

    Returns:
        segments: list of (start_frame, end_frame, dtype)
    """
    if not decisions:
        return []

    raw = []
    start = 0
    cur = decisions[0]
    for i in range(1, len(decisions)):
        if decisions[i] == cur:
            continue
        raw.append((start, i - 1, cur))
        start = i
        cur = decisions[i]
    raw.append((start, len(decisions) - 1, cur))

    merged = [raw[0]] if raw else []
    for seg in raw[1:]:
        last = merged[-1]
        seg_len = seg[1] - seg[0] + 1
        if seg_len < min_frames:
            merged[-1] = (last[0], seg[1], last[2])
        else:
            merged.append(seg)
    return merged


def get_lead_center_in_segment(keypoints, segments, lead_tid, lead_cx,
                                start_f, end_f, orig_w, crop9_w):
    """获取某 segment 内领操人的水平中心位置。

    Returns:
        seg_lead_cx: 归一化水平中心
    """
    cx_list = []
    for fi in range(start_f, end_f + 1):
        pose_data = keypoints.get(fi)
        if pose_data:
            for person_kps in pose_data:
                cx = _body_center_x(person_kps)
                if cx is not None and abs(cx - lead_cx) < 0.15:
                    cx_list.append(cx)
    seg_lead_cx = np.median(cx_list) if cx_list else lead_cx
    return seg_lead_cx


# ---- 内部辅助函数（与原 stage 保持一致） ----

def _body_center_x(person_kps):
    """计算人体水平中心（归一化）"""
    kps = np.array(person_kps)
    vis = kps[:, 2] > 0.5
    if vis.sum() < 6:
        return None
    shoulders_cx = (kps[11][0] + kps[12][0]) / 2
    hips_cx = (kps[23][0] + kps[24][0]) / 2
    return (shoulders_cx + hips_cx) / 2


def _body_size_score(person_kps):
    """计算人体大小评分（bbox 包围盒面积）

    2026-06-20 修复: 原公式用 nose + ankle 算身高, 背身时 nose/ankle 检测不准。
    改用所有可见关键点的 (x_max-x_min) × (y_max-y_min) 作为 bbox 面积。
    领操人通常距离镜头最近, bbox 最大, 应该胜出。
    """
    kps = np.array(person_kps)
    vis = kps[:, 2] > 0.3
    if vis.sum() < 5:
        return 0.0
    xs = kps[vis, 0]
    ys = kps[vis, 1]
    bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    return float(bbox_area)

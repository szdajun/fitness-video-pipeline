"""检测视频中的重复段落

基于姿态关键点分析，找出动作相似的时间段，建议裁切范围。

用法:
    python detect_repetitions.py <视频路径>
    python detect_repetitions.py <视频路径> --seg 10 --gap 15 --threshold 0.82
"""

import json
import sys
import numpy as np
from pathlib import Path


def load_keypoints(kp_path):
    with open(kp_path) as f:
        data = json.load(f)
    # 文件格式: {"keypoints": {"0": [[kp], ...], ...}, "video_info": {...}}
    kp_data = data["keypoints"]
    return {int(k): v for k, v in kp_data.items()}, data["video_info"]


def frame_descriptor(pose_data):
    """提取帧姿态描述子（中心化+归一化的关键点坐标）"""
    if not pose_data or not pose_data[0]:
        return None
    kps = np.array(pose_data[0])[:, :2]  # x, y (归一化 0~1)
    hip_c = (kps[23] + kps[24]) / 2
    centered = kps - hip_c
    shoulder_c = (kps[11] + kps[12]) / 2
    torso = max(np.linalg.norm(shoulder_c - hip_c), 0.01)
    return (centered / torso).flatten()  # 66维向量


def detect(keypoints, fps, seg_sec=10, stride_sec=2, min_gap_sec=15, threshold=0.82):
    """检测重复段落"""
    total_frames = max(keypoints.keys()) + 1
    seg_len = int(seg_sec * fps)
    stride_len = int(stride_sec * fps)

    # 计算每帧描述子
    descs = [None] * total_frames
    last = np.zeros(66)
    for i in range(total_frames):
        d = frame_descriptor(keypoints.get(i))
        if d is not None:
            last = d.copy()
        descs[i] = last

    # 滑动窗口: 构建段落描述子
    segments = []
    for s in range(0, total_frames - seg_len + 1, stride_len):
        frames = np.array(descs[s:s + seg_len])
        mean_d = frames.mean(axis=0)
        std_d = frames.std(axis=0)
        vel = np.diff(frames, axis=0).mean(axis=0)  # 平均运动速度
        desc = np.concatenate([mean_d, std_d, vel])
        segments.append((s / fps, (s + seg_len) / fps, desc))

    n = len(segments)
    print(f"  分析 {n} 个窗口...")

    # 向量化余弦相似度
    mat = np.array([s[2] for s in segments])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat_n = mat / norms
    sim = mat_n @ mat_n.T

    # 找相似对（跳过相邻窗口）
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if segments[j][0] - segments[i][1] < min_gap_sec:
                continue
            if sim[i, j] > threshold:
                pairs.append((i, j, float(sim[i, j])))
    pairs.sort(key=lambda x: -x[2])

    # 选非重叠的裁切（保留先出现的，裁掉后出现的）
    used = set()
    cuts = []
    for i, j, s in pairs:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        cuts.append((segments[i][0], segments[i][1],
                      segments[j][0], segments[j][1], s))

    return cuts


def merge_ranges(ranges):
    """合并重叠/相邻的时间范围"""
    if not ranges:
        return []
    r = sorted(ranges)
    merged = [list(r[0])]
    for s, e in r[1:]:
        if s <= merged[-1][1] + 3:  # 3秒容差内合并
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def main():
    if len(sys.argv) < 2:
        print("用法: python detect_repetitions.py <视频路径>")
        print("选项: --seg=10 --stride=2 --gap=15 --threshold=0.82")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    # 解析选项
    seg, stride, gap, threshold = 10, 2, 15, 0.82
    for arg in sys.argv[2:]:
        if arg.startswith("--seg="):
            seg = int(arg.split("=")[1])
        elif arg.startswith("--stride="):
            stride = int(arg.split("=")[1])
        elif arg.startswith("--gap="):
            gap = int(arg.split("=")[1])
        elif arg.startswith("--threshold="):
            threshold = float(arg.split("=")[1])

    # 查找关键点文件
    kp_file = None
    for d in [Path("output"), video_path.parent,
              Path("F:/wkspace/fitness-video-pipeline/output")]:
        f = d / f"{video_path.stem}_keypoints.json"
        if f.exists():
            kp_file = f
            break

    if not kp_file:
        print("未找到关键点文件，请先运行一次流水线处理")
        sys.exit(1)

    keypoints, video_info = load_keypoints(kp_file)
    fps = video_info["fps"]
    total_frames = max(keypoints.keys()) + 1
    duration = total_frames / fps

    print(f"视频: {total_frames}帧, {fps:.0f}fps, {duration:.0f}秒")
    print(f"参数: 窗口={seg}s, 步长={stride}s, 最小间隔={gap}s, 阈值={threshold}")
    print()

    cuts = detect(keypoints, fps, seg, stride, gap, threshold)

    if not cuts:
        print("未检测到明显重复段落")
        return

    print(f"检测到 {len(cuts)} 组重复:")
    raw_cuts = []
    for keep_s, keep_e, cut_s, cut_e, sim in cuts:
        print(f"  {sim:.0%}: 保留 [{keep_s:.0f}s~{keep_e:.0f}s] "
              f"裁掉 [{cut_s:.0f}s~{cut_e:.0f}s]")
        raw_cuts.append((cut_s, cut_e))

    merged = merge_ranges(raw_cuts)
    total_cut = sum(e - s for s, e in merged)

    cut_str = ",".join(f"{int(s)}-{int(e)}" for s, e in merged)
    print(f"\n合并后建议裁切: --cut {cut_str}")
    print(f"预计裁掉: {total_cut:.0f}秒 ({duration:.0f}s → {duration - total_cut:.0f}s)")


if __name__ == "__main__":
    main()

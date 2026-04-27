"""轻量质量指标模块

每次 pipeline 结束后输出 run_metrics.json，记录关键质量指标。
"""

import json, cv2, numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def compute_pose_detect_rate(keypoints: dict) -> float:
    """关键点检测成功率：检测到人的帧数 / 总帧数"""
    if not keypoints:
        return 0.0
    frames_with_detection = sum(
        1 for v in keypoints.values() if v and len(v) > 0
    )
    return round(frames_with_detection / len(keypoints), 4)


def compute_avg_person_count(keypoints: dict) -> float:
    """平均每帧人数"""
    if not keypoints:
        return 0.0
    counts = []
    for frame_data in keypoints.values():
        if frame_data:
            counts.append(len(frame_data))
    return round(sum(counts) / len(counts), 3) if counts else 0.0


def compute_lead_center_jitter(keypoints: dict, lead_tid: int) -> float:
    """领操人中心抖动程度（帧间位移标准差），单位 px@720p"""
    if not keypoints:
        return 0.0

    cx_list = []
    prev_cx = None
    for frame_idx in sorted(keypoints.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        frame_data = keypoints[frame_idx]
        if not frame_data:
            continue
        # 找 lead_tid 的人
        for person_kps in frame_data:
            if len(person_kps) < 13:
                continue
            kps = np.array(person_kps)
            shoulders_cx = (kps[5][0] + kps[6][0]) / 2
            hips_cx = (kps[11][0] + kps[12][0]) / 2
            cx = (shoulders_cx + hips_cx) / 2
            if prev_cx is not None:
                cx_list.append(abs(cx - prev_cx))
            prev_cx = cx
            break  # 只取第一个人

    if len(cx_list) < 2:
        return 0.0
    return round(float(np.std(cx_list)), 4)


def compute_output_frame_delta(actual: int, expected: int) -> int:
    """输出帧数偏差：实际帧数 - 预期帧数"""
    return actual - expected


def load_metrics_json(output_dir: Path, video_stem: str) -> Optional[Dict]:
    """加载已有的 metrics.json"""
    p = output_dir / f"{video_stem}_metrics.json"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def dump_metrics(output_dir: Path, video_stem: str, ctx, stage_times: Dict[str, float]):
    """输出 run_metrics.json"""
    metrics_path = output_dir / f"{video_stem}_metrics.json"
    vi = ctx.get("video_info", {})
    fps = vi.get("fps", 30)
    expected_frames = vi.get("frames", 0)
    final_path = ctx.get("final_path")

    # 实际帧数
    actual_frames = 0
    if final_path and Path(final_path).exists():
        cap = cv2.VideoCapture(final_path)
        if cap.isOpened():
            actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    # 关键点指标
    keypoints = ctx.get("keypoints")
    pose_detect_rate = compute_pose_detect_rate(keypoints) if keypoints else 0.0
    avg_person_count = compute_avg_person_count(keypoints) if keypoints else 0.0

    lead_tid = ctx.get("lead_tid")
    lead_jitter = compute_lead_center_jitter(keypoints, lead_tid) if keypoints and lead_tid is not None else 0.0

    metrics = {
        "video_duration_sec": round(actual_frames / fps, 3) if fps > 0 else 0,
        "output_frame_delta": compute_output_frame_delta(actual_frames, expected_frames),
        "pose_detect_rate": pose_detect_rate,
        "avg_person_count": avg_person_count,
        "lead_center_jitter": lead_jitter,
        "stage_times": stage_times,
    }

    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
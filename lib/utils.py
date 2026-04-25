"""工具函数: 帧读写、进度条"""

import cv2
import numpy as np
import os
import ctypes
from pathlib import Path


# Windows short path API for Chinese path support
_GetShortPathNameW = getattr(ctypes.windll.kernel32, 'GetShortPathNameW', None)
if _GetShortPathNameW:
    _GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    _GetShortPathNameW.restype = ctypes.c_uint


def _to_short_path(path_str: str) -> str:
    """将路径转换为 Windows 8.3 短路径（绕过中文/空格问题）"""
    if _GetShortPathNameW is None:
        return path_str
    buf_size = _GetShortPathNameW(path_str, None, 0)
    if buf_size == 0:
        return path_str
    buf = ctypes.create_unicode_buffer(buf_size)
    _GetShortPathNameW(path_str, buf, buf_size)
    return buf.value


def path_exists(path_str: str) -> bool:
    """检查文件是否存在（Windows 中文路径兼容）

    Windows 上 pathlib.Path.exists() 对中文路径返回 False，但 cv2.VideoCapture 能打开。
    FFmpeg 也能处理这些路径。本函数使用 cv2.VideoCapture 作为判定标准。
    对于非视频文件（如 JSON），检查文件大小 > 0。
    """
    if not path_str:
        return False
    p = Path(path_str)
    # 快速检查：先尝试 Path.exists()（对英文路径有效）
    if p.exists():
        return True
    # Windows 中文路径 bug：Path.exists() 返回 False 但文件实际存在
    # 用 cv2.VideoCapture 探测（对视频文件有效）
    cap = cv2.VideoCapture(path_str)
    opened = cap.isOpened()
    cap.release()
    if opened:
        return True
    # 非视频文件：检查文件大小 > 0（绕过 Path.exists 的中文 bug）
    try:
        if p.stat().st_size > 0:
            return True
    except Exception:
        pass
    return False


def create_writer(output_path: str, fps: float, width: int, height: int):
    """创建视频写入器（H.264 优先，失败则 fallback 到 mp4v）

    自动处理中文路径: 使用 Windows GetShortPathNameW API 获取 8.3 短路径。
    """
    # 使用 Windows 短路径绕过中文路径问题
    final_path = _to_short_path(str(output_path))

    writer = None
    for codec in ["avc1", "mp4v"]:
        writer = cv2.VideoWriter(final_path,
                                 cv2.VideoWriter_fourcc(*codec),
                                 fps, (width, height))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output_path}")

    return writer


def iter_frames(video_path: str, max_frames: int = None, to_rgb: bool = False):
    """逐帧迭代视频，yield (frame_idx, frame_bgr)"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    while max_frames is None or count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield count, frame
        count += 1

    cap.release()


def keypoint_bbox(kps):
    """从关键点计算边界框 (normalized 0-1)"""
    kps = np.array(kps)
    xs = kps[:, 0]
    ys = kps[:, 1]
    vis = kps[:, 2]
    mask = vis > 0.5
    if not mask.any():
        return None
    x_min, x_max = xs[mask].min(), xs[mask].max()
    y_min, y_max = ys[mask].min(), ys[mask].max()
    return x_min, y_min, x_max, y_max


def body_center(kps):
    """计算人体中心点 (normalized) — 使用肩髋中点"""
    kps = np.array(kps)
    # 左肩(11), 右肩(12), 左髋(23), 右髋(24)
    shoulders = (kps[11] + kps[12]) / 2
    hips = (kps[23] + kps[24]) / 2
    center = (shoulders + hips) / 2
    return center[0], center[1]


def transform_keypoints(keypoints, crop_x, crop_y, crop_w, crop_h, orig_w, orig_h):
    """将关键点从原始坐标系变换到裁剪坐标系"""
    result = []
    for kp in keypoints:
        x = (kp[0] * orig_w - crop_x) / crop_w
        y = (kp[1] * orig_h - crop_y) / crop_h
        vis = kp[2]
        result.append([x, y, vis])
    return result


# ========== 领操人追踪（增强版） ==========
# 所有彩蛋阶段共享同一个追踪函数，保证领操人 ID 一致性

def track_lead_person(keypoints, lead_lock_tid=None, lead_lock_cx=None,
                      match_threshold=0.2, lock_grace_frames=30):
    """增强版领操人追踪

    特性：
    - 一旦锁定领操人（出现次数最多），不再切换
    - 临时遮挡时保持锁定状态
    - 返回: (lead_tid, lead_cx, tracks)

    Args:
        keypoints: 关键点字典
        lead_lock_tid: 如果已知，强制使用此 tid 作为领操人
        lead_lock_cx: 领操人已知的 x 中心位置
        match_threshold: x 匹配阈值
        lock_grace_frames: 领操人消失后保持锁定的帧数
    """
    tracks = {}
    confirmed_lead_tid = lead_lock_tid
    grace_frames_left = 0

    for fi, frame_data in sorted(keypoints.items()):
        fi = int(fi)
        if not frame_data:
            continue
        for pi, person_kps in enumerate(frame_data):
            kps = np.array(person_kps)
            vis = kps[:, 2] > 0.5
            if vis.sum() < 6:
                cx = 0.5
            else:
                shoulders_cx = (kps[5][0] + kps[6][0]) / 2
                hips_cx = (kps[11][0] + kps[12][0]) / 2
                cx = (shoulders_cx + hips_cx) / 2

            best_tid = None
            best_dist = float('inf')
            for tid, trk in tracks.items():
                prev_cx = np.median(trk["cx_list"]) if trk["cx_list"] else cx
                dist = abs(cx - prev_cx)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None and best_dist < match_threshold:
                tracks[best_tid]["cx_list"].append(cx)
                tracks[best_tid]["count"] += 1
            else:
                new_tid = len(tracks)
                tracks[new_tid] = {"cx_list": [cx], "count": 1}

        # 如果上一帧有锁定的人没出现，减少 grace 计数
        if confirmed_lead_tid is not None:
            frame_tids = set()
            if frame_data:
                for pi2, person_kps2 in enumerate(frame_data):
                    kps2 = np.array(person_kps2)
                    vis2 = kps2[:, 2] > 0.5
                    if vis2.sum() < 6:
                        continue
                    shoulders_cx2 = (kps2[5][0] + kps2[6][0]) / 2
                    hips_cx2 = (kps2[11][0] + kps2[12][0]) / 2
                    cx2 = (shoulders_cx2 + hips_cx2) / 2
                    for tid, trk in tracks.items():
                        prev_cx = np.median(trk["cx_list"][-5:]) if trk["cx_list"] else cx2
                        if abs(cx2 - prev_cx) < match_threshold:
                            frame_tids.add(tid)
                            break

            if confirmed_lead_tid not in frame_tids:
                grace_frames_left -= 1
                if grace_frames_left < -lock_grace_frames:
                    confirmed_lead_tid = None

    # 确定领操人：优先使用锁定的，否则选出现次数最多的
    if confirmed_lead_tid is not None and confirmed_lead_tid in tracks:
        lead_tid = confirmed_lead_tid
    else:
        if tracks:
            lead_tid = max(tracks, key=lambda tid: tracks[tid]["count"])
        else:
            lead_tid = None

    lead_cx = np.median(tracks[lead_tid]["cx_list"]) if lead_tid and tracks[lead_tid]["cx_list"] else 0.5

    return lead_tid, lead_cx, tracks


# ========== 中文字体渲染 ==========
import subprocess
from pathlib import Path

# 尝试找中文字体
_FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",    # 黑体
    "C:/Windows/Fonts/simsun.ttc",    # 宋体
    "C:/Windows/Fonts/arial.ttf",     # 备用
]

def _find_chinese_font():
    for p in _FONT_PATHS:
        if Path(p).exists():
            return p
    return None


def draw_chinese_text(frame, text, position, font_size=20, color=(255, 255, 255), bg_color=None):
    """在 OpenCV 帧上渲染中文文字

    Args:
        frame: OpenCV BGR 图像
        text: 中文字符串
        position: (x, y) 左下角坐标
        font_size: 字体大小（像素）
        color: BGR 颜色
        bg_color: 背景色 BGR，如果有的话

    Returns:
        叠加了文字的帧（in-place 修改）
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # 没装 Pillow，回退到英文渲染（会有部分乱码）
        cv2.putText(frame, text, position,
                   cv2.FONT_HERSHEY_DUPLEX, font_size / 20,
                   color, 1, cv2.LINE_AA)
        return frame

    font_path = _find_chinese_font()
    if font_path is None:
        cv2.putText(frame, text, position,
                   cv2.FONT_HERSHEY_DUPLEX, font_size / 20,
                   color, 1, cv2.LINE_AA)
        return frame

    x, y = position
    # 创建 TrueType 字体
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    # OpenCV BGR → PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 计算文字边界
    bbox = draw.textbbox((x, y - font_size), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # 背景
    if bg_color is not None:
        pad = 4
        draw.rectangle([x - pad, y - font_size - pad, x + tw + pad, y + pad],
                      fill=bg_color)

    # 绘制文字（PIL 使用 RGB，需要转换 color）
    draw.text((x, y - font_size), text, font=font,
             fill=(color[2], color[1], color[0]))  # BGR → RGB

    # 转换回 OpenCV BGR
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    # 拷贝回原帧（保持原始数据类型）
    frame[:] = result

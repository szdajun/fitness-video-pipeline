"""夜景高光优化三层处理模块

- 第一层：LAB L通道高光压缩（压灯区/亮地面/白衣高光）
- 第二层：HSV 白衣/低饱和亮区保护（高亮+低饱和区域单独处理）
- 第三层：连通域分析大面积灯区额外抑制
"""

import cv2
import numpy as np


def protect_highlights(frame, strength=0.35, threshold=185, blur_ksize=5):
    """第一层：基础高光压缩（LAB L通道 soft roll-off）

    只压过亮区域，不伤害正常中间调。
    适用于：顶灯、亮地面、白衣高光、柱面反光。

    Args:
        frame: BGR 图像
        strength: 0~1，越大压得越明显
        threshold: 从该亮度开始压高光（LAB L通道）
        blur_ksize: mask 平滑核大小（奇数）
    """
    if strength <= 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]

    excess = np.maximum(l - threshold, 0.0)
    if np.max(excess) < 1:
        return frame

    # soft roll-off: 比硬截断更平滑
    range_val = max(1.0, 255.0 - threshold)
    compressed = threshold + excess / (1.0 + strength * excess / range_val)

    mask = np.clip(excess / range_val, 0.0, 1.0)
    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    l_new = l * (1.0 - mask) + compressed * mask
    lab[:, :, 0] = np.clip(l_new, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def protect_bright_neutral_regions(frame, strength=0.20, value_threshold=200,
                                   sat_threshold=60, blur_ksize=5):
    """第二层：白衣/低饱和亮区保护（HSV空间）

    识别"高亮+低饱和"区域（白衣、浅灰衣物、反光区域），
    轻微压缩亮度，避免纹理发飘。

    Args:
        frame: BGR 图像
        strength: 0~1，只压一点，避免白衣发灰
        value_threshold: V通道高亮判定阈值
        sat_threshold: S通道低饱和判定阈值
        blur_ksize: mask 平滑核大小（奇数）
    """
    if strength <= 0:
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    bright_mask = np.clip((v - value_threshold) / max(1.0, 255.0 - value_threshold), 0.0, 1.0)
    low_sat_mask = np.clip((sat_threshold - s) / max(1.0, sat_threshold), 0.0, 1.0)
    mask = bright_mask * low_sat_mask

    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

    # 只轻微压缩，避免白衣发灰
    v_new = v * (1.0 - strength * mask)
    hsv[:, :, 2] = np.clip(v_new, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def suppress_large_light_regions(frame, strength=0.18, threshold=235,
                                 min_area=2500, blur_ksize=21):
    """第三层：大面积灯区与灯光溢出保护

    识别亮度通道中面积较大的连续亮斑（顶灯、灯柱泛白区域），
    对这些区域额外压一点亮度。

    Args:
        frame: BGR 图像
        strength: 0~1，额外压制强度
        threshold: 超亮区域判定阈值
        min_area: 连通域最小面积（平方像素）
        blur_ksize: mask 平滑核大小（较大值使边缘更柔和）
    """
    if strength <= 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]

    binary = (l >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    mask = np.zeros_like(l, dtype=np.float32)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask[labels == i] = 1.0

    if np.max(mask) < 0.5:
        return frame

    if blur_ksize >= 3 and blur_ksize % 2 == 1:
        mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
        mask = np.clip(mask, 0.0, 1.0)

    l_new = l * (1.0 - strength * mask)
    lab[:, :, 0] = np.clip(l_new, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def optimize_night_highlights(frame, cfg):
    """三层高光处理统一入口

    推荐调用顺序：
        1. protect_highlights        — 基础高光压缩
        2. protect_bright_neutral_regions  — 白衣保护
        3. suppress_large_light_regions    — 大面积灯区

    Args:
        frame: BGR 图像
        cfg: 配置字典，包含各层参数

    Returns:
        处理后的 BGR 图像
    """
    frame = protect_highlights(
        frame,
        strength=cfg.get("highlight_protect", 0.0),
        threshold=cfg.get("highlight_threshold", 185),
        blur_ksize=cfg.get("highlight_blur", 5),
    )

    frame = protect_bright_neutral_regions(
        frame,
        strength=cfg.get("white_protect", 0.0),
        value_threshold=cfg.get("white_value_threshold", 200),
        sat_threshold=cfg.get("white_sat_threshold", 60),
        blur_ksize=cfg.get("white_protect_blur", 5),
    )

    frame = suppress_large_light_regions(
        frame,
        strength=cfg.get("light_region_protect", 0.0),
        threshold=cfg.get("light_region_threshold", 235),
        min_area=cfg.get("light_region_min_area", 2500),
        blur_ksize=cfg.get("light_region_blur", 21),
    )

    return frame

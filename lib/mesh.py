"""三角网格: 身体区域的 Delaunay 三角化"""

import numpy as np
from scipy.spatial import Delaunay


def create_body_mesh(kps, img_h, img_w, padding=40):
    """基于姿态关键点创建身体区域的三角网格

    Args:
        kps: 关键点列表 [[x, y, vis], ...] (normalized 0-1)
        img_h, img_w: 图像尺寸
        padding: 网格到身体边缘的像素填充

    Returns:
        points: 原始网格点坐标 (N, 2) in pixels
        triangles: Delaunay 三角索引
        regions: 每个点的身体区域标签
    """
    kps = np.array(kps)
    h, w = img_h, img_w

    # 提取关键身体点（像素坐标）
    pts = {}

    def kp_px(idx):
        x, y, vis = kps[idx]
        if vis > 0.5:
            return np.array([x * w, y * h])
        return None

    # 头部
    nose = kp_px(0)
    left_ear = kp_px(7)
    right_ear = kp_px(8)

    # 上身
    left_shoulder = kp_px(11)
    right_shoulder = kp_px(12)
    left_elbow = kp_px(13)
    right_elbow = kp_px(14)
    left_wrist = kp_px(15)
    right_wrist = kp_px(16)

    # 下身
    left_hip = kp_px(23)
    right_hip = kp_px(24)
    left_knee = kp_px(25)
    right_knee = kp_px(26)
    left_ankle = kp_px(27)
    right_ankle = kp_px(28)

    # 计算中点
    shoulder_mid = (left_shoulder + right_shoulder) / 2 if left_shoulder is not None and right_shoulder is not None else None
    hip_mid = (left_hip + right_hip) / 2 if left_hip is not None and right_hip is not None else None

    # 生成网格点列表: (point, region)
    # regions: head, torso, waist, left_leg, right_leg, left_arm, right_arm
    grid_points = []
    regions = []

    def add_pt(p, region):
        if p is not None:
            grid_points.append(p.copy())
            regions.append(region)

    # 头部区域
    add_pt(nose, "head")
    add_pt(left_ear, "head")
    add_pt(right_ear, "head")
    if left_ear is not None and right_ear is not None:
        head_top = (left_ear + right_ear) / 2
        head_top[1] -= padding * 0.5
        add_pt(head_top, "head")

    # 躯干区域
    add_pt(left_shoulder, "torso")
    add_pt(right_shoulder, "torso")
    add_pt(shoulder_mid, "torso")
    add_pt(left_hip, "waist")
    add_pt(right_hip, "waist")
    add_pt(hip_mid, "waist")

    # 腰部中间点
    if shoulder_mid is not None and hip_mid is not None:
        waist_left = np.array([left_hip[0] if left_hip is not None else (shoulder_mid[0] - 30), (shoulder_mid[1] + hip_mid[1]) / 2])
        waist_right = np.array([right_hip[0] if right_hip is not None else (shoulder_mid[0] + 30), (shoulder_mid[1] + hip_mid[1]) / 2])
        add_pt(waist_left, "waist")
        add_pt(waist_right, "waist")

    # 手臂
    add_pt(left_elbow, "left_arm")
    add_pt(right_elbow, "right_arm")
    add_pt(left_wrist, "left_arm")
    add_pt(right_wrist, "right_arm")

    # 腿部
    add_pt(left_knee, "left_leg")
    add_pt(right_knee, "right_leg")
    add_pt(left_ankle, "left_leg")
    add_pt(right_ankle, "right_leg")

    # 腿底部填充点
    if left_ankle is not None:
        add_pt(np.array([left_ankle[0], left_ankle[1] + padding]), "left_leg")
    if right_ankle is not None:
        add_pt(np.array([right_ankle[0], right_ankle[1] + padding]), "right_leg")

    if len(grid_points) < 3:
        return None, None, None

    points = np.array(grid_points)
    regions = np.array(regions)

    # Delaunay 三角化
    tri = Delaunay(points)

    return points, tri.simplices, regions


def get_body_region(y_norm, kps):
    """根据 y 坐标和关键点判断身体区域 (normalized 0-1)"""
    kps = np.array(kps)
    shoulder_y = (kps[11][1] + kps[12][1]) / 2
    hip_y = (kps[23][1] + kps[24][1]) / 2
    knee_y = (kps[25][1] + kps[26][1]) / 2

    if y_norm < shoulder_y:
        return "head"
    elif y_norm < (shoulder_y + hip_y) / 2:
        return "torso"
    elif y_norm < hip_y:
        return "waist"
    elif y_norm < knee_y:
        return "thigh"
    else:
        return "lower_leg"

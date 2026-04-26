"""网格变形引擎 v4: TPS风格自然身体塑形

核心改进:
- TPS (Thin Plate Spline) 风格平滑插值，过渡自然无折痕
- 身体遮罩: 变形仅作用于身体区域，背景不动
- 多层平滑: 躯干/腰部/大腿分别瘦身，无硬边界
- Poisson 风格混合: 变形区域与背景无缝融合
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# 模块级 meshgrid 缓存: (img_h, img_w) -> (yy, xx)
_mesh_cache = {}

# 身体关键区域索引 (COCO 17点)
# 0=鼻, 1-2=眼, 3-4=耳, 5-6=肩, 7=肘, 8=手, 9-10=髋, 11-12=膝, 13-14=脚
BODY_INDICES = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}


def create_displacement_map(kps, img_h, img_w, config):
    kps_arr = np.array(kps)

    leg_lengthen = config.get("leg_lengthen", 1.0)
    waist_slim = config.get("waist_slim", 1.0)
    overall_slim = config.get("overall_slim", 1.0)
    leg_slim = config.get("leg_slim", 1.0)
    chest_enlarge = config.get("chest_enlarge", 1.0)
    neck_lengthen = config.get("neck_lengthen", 1.0)

    # 关键 y 坐标 (像素) - COCO 17点格式
    nose_y = kps_arr[0][1] * img_h
    shoulder_y = (kps_arr[5][1] + kps_arr[6][1]) / 2 * img_h
    hip_y = (kps_arr[11][1] + kps_arr[12][1]) / 2 * img_h
    ankle_y = (kps_arr[15][1] + kps_arr[16][1]) / 2 * img_h

    # 身体中心 x
    body_cx = (kps_arr[11][0] + kps_arr[12][0] + kps_arr[5][0] + kps_arr[6][0]) / 4 * img_w
    shoulder_w = abs(kps_arr[6][0] - kps_arr[5][0]) * img_w
    hip_w = abs(kps_arr[12][0] - kps_arr[11][0]) * img_w

    # 身体半宽（覆盖肩膀到边缘）
    body_half = shoulder_w * 0.8

    # 复用 meshgrid 缓存，避免每帧重新分配 ~16MB (1080p)
    key = (img_h, img_w)
    if key not in _mesh_cache:
        _mesh_cache[key] = (np.mgrid[0:img_h, 0:img_w][0].astype(np.float64),
                             np.mgrid[0:img_h, 0:img_w][1].astype(np.float64))
    yy, xx = _mesh_cache[key]
    map_y = yy.copy()
    map_x = xx.copy()

    dist_to_center = np.abs(xx - body_cx)
    # 平滑的身体权重: sigmoid，防止 overflow 保证边界柔和
    edge_width = body_half * 0.5
    exp_arg = np.clip((dist_to_center - body_half) / edge_width, -30, 30)
    h_weight = 1.0 / (1.0 + np.exp(exp_arg))

    # ================================================================
    # 1. 腿拉长: 混合方案
    #    - 躯干轻度压缩（最多15%），上身不会太短
    #    - 腿部内部轻度拉伸，靠近脚踝时渐弱（脚不变形）
    #    - 脚踝以下完全不动
    # ================================================================
    if leg_lengthen != 1.0 and (ankle_y - hip_y) > 10 and (hip_y - shoulder_y) > 10:
        leg_h = ankle_y - hip_y
        torso_h = hip_y - shoulder_y
        target_extra = leg_h * (leg_lengthen - 1.0)

        # 躯干最多压缩15%
        torso_extra = min(target_extra, torso_h * 0.15)
        # 剩余通过腿部内部拉伸
        stretch_extra = target_extra - torso_extra

        # --- 躯干压缩 ---
        if torso_extra > 0:
            new_hip_y = hip_y - torso_extra
            new_torso_h = new_hip_y - shoulder_y
            torso_mask = (yy >= shoulder_y) & (yy < new_hip_y)
            if torso_mask.any() and new_torso_h > 1:
                t = (yy[torso_mask] - shoulder_y) / new_torso_h
                map_y[torso_mask] = shoulder_y + t * torso_h
        else:
            new_hip_y = hip_y

        # --- 腿部拉伸 (new_hip ~ ankle) ---
        # 拉伸效果在脚踝附近平滑衰减，裤脚/鞋子不变形
        total_leg_output = leg_h + torso_extra + stretch_extra
        transition_start = 0.75  # 从腿的75%位置开始衰减
        leg_mask = (yy >= new_hip_y) & (yy <= ankle_y)
        if leg_mask.any() and (ankle_y - new_hip_y) > 1:
            t = (yy[leg_mask] - new_hip_y) / (ankle_y - new_hip_y)

            # 基础拉伸: 把 leg_h 内容映射到更大的输出范围
            t_sample = t * (leg_h / (ankle_y - new_hip_y))

            # 在脚踝附近衰减: t > transition_start 时渐弱
            # 用 smoothstep 让衰减平滑
            blend = np.clip((t - transition_start) / (1.0 - transition_start), 0, 1)
            blend = blend * blend * (3 - 2 * blend)  # smoothstep

            # 混合: 上部用拉伸映射, 下部(裤脚)用原始映射(不动)
            final_t = t_sample * (1 - blend) + t * blend
            map_y[leg_mask] = hip_y + final_t * leg_h

        # --- 脚踝以下: 完全不动 ---

        # 只在身体附近生效
        final_map_y = yy.astype(np.float64) * (1 - h_weight) + map_y * h_weight
        map_y = final_map_y

    # ================================================================
    # 1.5 脖子拉长
    #    鼻子到肩膀之间，从更小的范围采样 → 脖子看起来更长
    # ================================================================
    if neck_lengthen != 1.0 and (shoulder_y - nose_y) > 5:
        neck_h = shoulder_y - nose_y
        extra = neck_h * (neck_lengthen - 1.0)
        # 额外空间从肩膀以下（躯干）借，头部完全不动
        torso_h = hip_y - shoulder_y
        borrow = min(extra, torso_h * 0.15)  # 从躯干借，头部不动

        # 肩膀位置上移（脖子变长），肩膀以下整体上移 borrow
        new_shoulder_y = shoulder_y - borrow
        new_neck_h = new_shoulder_y - nose_y

        # 颈部区域拉伸显示
        neck_mask = (yy >= nose_y) & (yy < new_shoulder_y)
        if neck_mask.any() and new_neck_h > 1:
            t = (yy[neck_mask] - nose_y) / new_neck_h
            map_y[neck_mask] = nose_y + t * neck_h

        # 肩膀以下整体上移 borrow 像素（从躯干借的空间）
        below_shoulder = yy >= new_shoulder_y
        if below_shoulder.any() and borrow > 0:
            map_y[below_shoulder] -= borrow

        # 只在身体附近生效
        map_y = yy.astype(np.float64) * (1 - h_weight) + map_y * h_weight

    # ================================================================
    # 2. 胸部放大
    #    从更窄的输入范围采样 → 胸部看起来更宽/更丰满
    #    只作用于肩膀以下、腰以上的胸部区域
    # ================================================================
    if chest_enlarge != 1.0:
        torso_h = hip_y - shoulder_y
        chest_center_y = shoulder_y + torso_h * 0.2
        chest_sigma_y = torso_h * 0.18

        y_weight = np.exp(-0.5 * ((yy - chest_center_y) / chest_sigma_y) ** 2)
        # 水平: 中间强两侧弱，只在身体附近
        x_weight = np.exp(-0.5 * (dist_to_center / body_half) ** 2)
        combined = y_weight * x_weight

        # 从更窄范围采样: chest_enlarge > 1 → 从更近中线的位置采样 → 放大
        new_x = body_cx + (xx - body_cx) / chest_enlarge
        map_x = map_x * (1 - combined) + new_x * combined

    # ================================================================
    # 3. 腰部收窄 (TPS 风格)
    # ================================================================
    if waist_slim != 1.0:
        waist_center = (shoulder_y * 0.35 + hip_y * 0.65)
        waist_sigma_y = abs(hip_y - shoulder_y) * 0.30

        # TPS 风格: 余弦插值让过渡更平滑
        y_dist = np.abs(yy - waist_center)
        y_weight = np.exp(-0.5 * (y_dist / waist_sigma_y) ** 2)
        # 余弦平滑: 中心到边缘线性衰减
        t = np.clip(y_dist / (waist_sigma_y * 2), 0, 1)
        cos_smooth = 0.5 + 0.5 * np.cos(np.pi * t)
        y_weight = y_weight * cos_smooth

        combined = y_weight * h_weight

        # TPS 风格: 从中心到边缘分段采样，避免硬边界
        slim_strength = waist_slim
        new_x = body_cx + (xx - body_cx) * slim_strength
        map_x = map_x * (1 - combined) + new_x * combined

    # ================================================================
    # 3. 整体瘦身 (TPS 风格，仅身体区域)
    # ================================================================
    if overall_slim != 1.0:
        in_body = (yy >= nose_y - 20) & (yy <= ankle_y + 10)
        if in_body.any():
            # TPS 风格: 头部不动，下身逐渐减弱
            body_top = nose_y - 20
            body_bottom = ankle_y + 10
            t_body = np.clip((yy[in_body] - body_top) / (body_bottom - body_top), 0, 1)

            # 余弦插值：上身瘦更多(0.7)，下身保留(1.0)
            local_slim = 1.0 - (1.0 - overall_slim) * (0.6 + 0.4 * np.cos(np.pi * t_body))

            new_x = body_cx + (xx[in_body] - body_cx) * local_slim
            # 只在身体区域内应用
            body_region_mask = h_weight[in_body]
            map_x[in_body] = map_x[in_body] * (1 - body_region_mask) + new_x * body_region_mask

    # ================================================================
    # 4. 腿部瘦身 (TPS 风格)
    # ================================================================
    if leg_slim != 1.0 and (ankle_y - hip_y) > 10:
        leg_top = hip_y
        leg_bottom = ankle_y + 20
        in_leg = (yy >= leg_top) & (yy <= leg_bottom)
        if in_leg.any():
            t_leg = np.clip((yy[in_leg] - leg_top) / (leg_bottom - leg_top), 0, 1)
            # 余弦插值：髋部瘦(0.7)，脚踝不动(1.0)
            local_leg = 1.0 - (1.0 - leg_slim) * (0.5 + 0.5 * np.cos(np.pi * t_leg))

            new_x = body_cx + (xx[in_leg] - body_cx) * local_leg
            map_x[in_leg] = map_x[in_leg] * (1 - h_weight[in_leg]) + new_x * h_weight[in_leg]

    return map_x.astype(np.float32), map_y.astype(np.float32)


def apply_warp(frame, map_x, map_y):
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT_101)


def create_body_mask(kps_arr, img_h, img_w, margin=1.3):
    """根据关键点创建身体凸包遮罩

    Args:
        kps_arr: [17, 3] 关键点数组
        img_h, img_w: 图像尺寸
        margin: 身体半宽扩展系数
    Returns:
        body_mask: [img_h, img_w] float32, 0-1 范围
    """
    # 可见点
    vis = kps_arr[:, 2] > 0.3
    if vis.sum() < 4:
        return np.ones((img_h, img_w), dtype=np.float32)

    # 身体中心线
    shoulder_mid_y = (kps_arr[5][1] + kps_arr[6][1]) / 2 * img_h
    hip_mid_y = (kps_arr[11][1] + kps_arr[12][1]) / 2 * img_h
    ankle_mid_y = (kps_arr[15][1] + kps_arr[16][1]) / 2 * img_h

    # 身体各部位半宽（像素）
    shoulder_w = abs(kps_arr[6][0] - kps_arr[5][0]) / 2 * img_w * margin
    hip_w = abs(kps_arr[12][0] - kps_arr[11][0]) / 2 * img_w * margin

    # 创建身体轮廓 mask
    mask = np.zeros((img_h, img_w), dtype=np.float32)

    # 用多边形近似身体轮廓
    body_pts = []

    # 肩部宽度
    left_shoulder = (int(kps_arr[5][0] * img_w - shoulder_w * 0.3),
                     int(shoulder_mid_y - (hip_mid_y - shoulder_mid_y) * 0.1))
    right_shoulder = (int(kps_arr[6][0] * img_w + shoulder_w * 0.3),
                      int(shoulder_mid_y - (hip_mid_y - shoulder_mid_y) * 0.1))

    # 髋部宽度
    left_hip = (int(kps_arr[11][0] * img_w - hip_w * 0.5),
                int(hip_mid_y))
    right_hip = (int(kps_arr[12][0] * img_w + hip_w * 0.5),
                 int(hip_mid_y))

    # 腿部延伸到底部
    left_ankle = (int(kps_arr[15][0] * img_w - hip_w * 0.4),
                  int(ankle_mid_y))
    right_ankle = (int(kps_arr[16][0] * img_w + hip_w * 0.4),
                   int(ankle_mid_y))

    # 构建身体多边形（左右对称+平滑）
    pts = np.array([
        [left_shoulder[0], left_shoulder[1]],
        [right_shoulder[0], right_shoulder[1]],
        [right_hip[0], right_hip[1]],
        [right_ankle[0], right_ankle[1]],
        [left_ankle[0], left_ankle[1]],
        [left_hip[0], left_hip[1]],
    ], dtype=np.int32)

    cv2.fillPoly(mask, [pts], 1.0)

    # 高斯模糊边缘，过渡更自然
    mask = cv2.GaussianBlur(mask, (31, 31), 15)
    return mask


def create_tps_slim_map(kps, img_h, img_w, config):
    """TPS风格自然身体瘦身

    使用 TPS (Thin Plate Spline) 风格的平滑变形：
    - 身体区域内向收缩
    - 边缘平滑过渡，无硬边界
    - 躯干/腰部/大腿分层瘦身
    """
    kps_arr = np.array(kps)

    waist_slim = config.get("waist_slim", 1.0)
    overall_slim = config.get("overall_slim", 1.0)
    leg_slim = config.get("leg_slim", 1.0)

    # 坐标
    shoulder_y = (kps_arr[5][1] + kps_arr[6][1]) / 2 * img_h
    waist_center_y = (kps_arr[11][1] + kps_arr[12][1]) / 2 * img_h + (kps_arr[5][1] + kps_arr[6][1]) / 2 * img_h * 0.3
    waist_center_y = waist_center_y / 1.5
    hip_y = (kps_arr[11][1] + kps_arr[12][1]) / 2 * img_h
    knee_y = (kps_arr[13][1] + kps_arr[14][1]) / 2 * img_h
    ankle_y = (kps_arr[15][1] + kps_arr[16][1]) / 2 * img_h

    body_cx = (kps_arr[11][0] + kps_arr[12][0] + kps_arr[5][0] + kps_arr[6][0]) / 4 * img_w

    # 复用 meshgrid
    key = (img_h, img_w)
    if key not in _mesh_cache:
        _mesh_cache[key] = (np.mgrid[0:img_h, 0:img_w][0].astype(np.float64),
                            np.mgrid[0:img_h, 0:img_w][1].astype(np.float64))
    yy, xx = _mesh_cache[key]

    # 初始化映射
    map_y = yy.copy().astype(np.float32)
    map_x = xx.copy().astype(np.float32)

    # 距离身体中心
    dist_to_center = np.abs(xx - body_cx)

    # 创建身体 mask（只在这个区域内变形）
    body_mask = create_body_mask(kps_arr, img_h, img_w, margin=1.4)

    # TPS 风格多层瘦身
    # 1. 腰部收窄（最重要）
    if waist_slim != 1.0:
        # 腰部区域的高斯权重
        waist_sigma_y = abs(hip_y - shoulder_y) * 0.35
        waist_weight = np.exp(-0.5 * ((yy - waist_center_y) / waist_sigma_y) ** 2)

        # TPS 风格: 用余弦插值让边缘过渡更平滑
        slim_factor = waist_slim
        # 中心最强，线性衰减到边缘
        local_slim = slim_factor + (1 - slim_factor) * (1 - waist_weight)

        # 水平收缩：中心最强
        new_x = body_cx + (xx - body_cx) * local_slim
        map_x = map_x * (1 - body_mask * waist_weight) + new_x * body_mask * waist_weight

    # 2. 整体瘦身（从上到下，强度渐变）
    if overall_slim != 1.0:
        # 上身到下身强度渐变
        body_top = min(shoulder_y, kps_arr[0][1] * img_h)
        body_bottom = max(ankle_y, knee_y)

        # 全身强度映射（0=头，1=脚）
        t_body = np.clip((yy - body_top) / (body_bottom - body_top + 1e-6), 0, 1)

        # 余弦插值：上身瘦更多，下身保留
        slim_strength = overall_slim
        local_overall = 1.0 - (1.0 - slim_strength) * (0.5 + 0.5 * np.cos(np.pi * t_body))

        new_x = body_cx + (xx - body_cx) * local_overall
        map_x = map_x * (1 - body_mask * 0.7) + new_x * body_mask * 0.7

    # 3. 大腿瘦身（膝盖以上）
    if leg_slim != 1.0 and (knee_y - hip_y) > 20:
        thigh_sigma_y = abs(knee_y - hip_y) * 0.4
        thigh_center_y = hip_y + abs(knee_y - hip_y) * 0.3

        thigh_weight = np.exp(-0.5 * ((yy - thigh_center_y) / thigh_sigma_y) ** 2)

        # 大腿区域只作用在 x 方向
        leg_x_weight = np.maximum(0, (xx - body_cx + img_w * 0.3) / img_w * 2)
        leg_x_weight = np.minimum(1, leg_x_weight)

        new_x = body_cx + (xx - body_cx) / leg_slim
        map_x = map_x * (1 - body_mask * thigh_weight * leg_x_weight * 0.5) + \
                new_x * body_mask * thigh_weight * leg_x_weight * 0.5

    # 4. Poisson 风格边缘融合
    # 在身体mask边缘做轻微高斯模糊，减少硬边界
    if body_mask is not None:
        # 边缘区域的 alpha 混合
        edge_mask = np.abs(body_mask - 0.5) * 2  # 边缘处为1，中心为0
        edge_blend = gaussian_filter(edge_mask, sigma=5)

        # 混合原始坐标和变形坐标
        map_x = map_x * (1 - edge_blend * 0.1) + xx * edge_blend * 0.1

    return map_x.astype(np.float32), map_y.astype(np.float32)

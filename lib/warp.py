"""网格变形引擎 v3: 正确的身体比例调整

核心原理:
- 腿拉长 = 髋关节上移(压缩躯干)，脚踝不动，腿占更多空间
- 腰收窄 = 从更宽的输入范围采样，腰看起来更窄
- 整体瘦身 = 全身横向收缩
"""

import cv2
import numpy as np

# 模块级 meshgrid 缓存: (img_h, img_w) -> (yy, xx)
_mesh_cache = {}


def create_displacement_map(kps, img_h, img_w, config):
    kps_arr = np.array(kps)

    leg_lengthen = config.get("leg_lengthen", 1.0)
    waist_slim = config.get("waist_slim", 1.0)
    overall_slim = config.get("overall_slim", 1.0)
    leg_slim = config.get("leg_slim", 1.0)
    chest_enlarge = config.get("chest_enlarge", 1.0)
    neck_lengthen = config.get("neck_lengthen", 1.0)

    # 关键 y 坐标 (像素)
    nose_y = kps_arr[0][1] * img_h
    shoulder_y = (kps_arr[11][1] + kps_arr[12][1]) / 2 * img_h
    hip_y = (kps_arr[23][1] + kps_arr[24][1]) / 2 * img_h
    ankle_y = (kps_arr[27][1] + kps_arr[28][1]) / 2 * img_h

    # 身体中心 x
    body_cx = (kps_arr[11][0] + kps_arr[12][0] + kps_arr[23][0] + kps_arr[24][0]) / 4 * img_w
    shoulder_w = abs(kps_arr[12][0] - kps_arr[11][0]) * img_w
    hip_w = abs(kps_arr[24][0] - kps_arr[23][0]) * img_w

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
    # 3. 腰部收窄
    # ================================================================
    if waist_slim != 1.0:
        waist_center = (shoulder_y * 0.35 + hip_y * 0.65)
        waist_sigma_y = abs(hip_y - shoulder_y) * 0.25

        y_weight = np.exp(-0.5 * ((yy - waist_center) / waist_sigma_y) ** 2)
        combined = y_weight * h_weight

        new_x = body_cx + (xx - body_cx) / waist_slim
        map_x = map_x * (1 - combined) + new_x * combined

    # ================================================================
    # 3. 整体瘦身
    # ================================================================
    if overall_slim != 1.0:
        in_body = (yy >= nose_y) & (yy <= ankle_y)
        if in_body.any():
            new_x = body_cx + (xx[in_body] - body_cx) / overall_slim
            map_x[in_body] = map_x[in_body] * (1 - h_weight[in_body]) + new_x * h_weight[in_body]

    # ================================================================
    # 4. 腿部瘦身
    # ================================================================
    if leg_slim != 1.0 and (ankle_y - hip_y) > 10:
        in_leg = (yy >= hip_y) & (yy <= ankle_y + 20)
        if in_leg.any():
            new_x = body_cx + (xx[in_leg] - body_cx) / leg_slim
            map_x[in_leg] = map_x[in_leg] * (1 - h_weight[in_leg]) + new_x * h_weight[in_leg]

    return map_x.astype(np.float32), map_y.astype(np.float32)


def apply_warp(frame, map_x, map_y):
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT_101)

"""人脸变形核心函数：V-face下巴塑形、大眼效果、磨皮"""
import cv2
import numpy as np


# ============================================================
# MediaPipe FaceMesh 关键点索引（与 research 一致）
# ============================================================
# 下巴轮廓 (Jawline) — V-face用
JAWLINE = list(range(0, 17))  # 0=下巴中心, 1-8=左下颌, 9-16=右下颌

# 左眼 6 点
LEFT_EYE = list(range(33, 39))
# 右眼 6 点
RIGHT_EYE = list(range(42, 48))

# 鼻尖
NOSE_TIP = 4


# ============================================================
# V-face: 下巴塑形
# ============================================================
def create_vface_displacement_map(face_kps, img_h, img_w, strength=0.7):
    """创建V-face下巴塑形位移图

    Args:
        face_kps: 468个关键点，normalized (0~1)，每点 [x, y, z]
        img_h, img_w: 图像尺寸
        strength: 0.0~0.75，越大下巴越尖

    Returns:
        map_x, map_y: cv2.remap 用的位移图 (float32)
    """
    if strength <= 0:
        map_x = np.tile(np.arange(img_w, dtype=np.float32), (img_h, 1))
        map_y = np.tile(np.arange(img_h, dtype=np.float32).reshape(-1, 1), (1, img_w))
        return map_x, map_y

    kps = np.array(face_kps)
    jawline = kps[JAWLINE]  # (17, 3)
    jawline[..., 0] *= img_w
    jawline[..., 1] *= img_h

    # 面部中心: x=中线, y=下颌最低点
    face_cx = img_w / 2
    chin_y = jawline[0, 1]
    jawline_center = jawline[0].copy()

    # 裁切后的图像，y范围大约是头部中下部
    # 创建 meshgrid
    yy, xx = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij')
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    # 计算每个像素到面部中线的水平距离
    dist_to_center = np.abs(xx - face_cx)

    # 权重: 下巴附近强，往上衰减
    # 用余弦权重: 下巴最强，往上逐渐减弱到0
    chin_y = jawline[0, 1]
    top_y = jawline[8, 1]  # 额头顶部y (下巴往上)
    y_range = chin_y - top_y

    def cos_weight(y):
        """下巴附近权重=1，往上逐渐到0"""
        t = (chin_y - y) / (y_range + 1e-6)
        t = np.clip(t, 0, 1)
        return 0.5 + 0.5 * np.cos(t * np.pi)

    y_weight = cos_weight(yy)
    # 水平衰减: 越靠近中线越强
    x_decay = 1.0 - dist_to_center / (img_w / 2) * 0.5
    x_decay = np.clip(x_decay, 0, 1)
    combined = y_weight * x_decay

    # 位移: 往中线推，strength=0.75时最大位移约15-20px
    dx = (face_cx - xx) * strength * combined * 0.4
    dy = np.zeros_like(dx)

    map_x = xx + dx
    map_y = yy + dy

    return map_x, map_y


# ============================================================
# 大眼效果
# ============================================================
def create_eye_enlarge_displacement_map(face_kps, img_h, img_w, enlarge=1.3):
    """创建眼睛放大位移图（局部）

    Args:
        face_kps: 468个关键点，normalized (0~1)
        img_h, img_w: 图像尺寸
        enlarge: 1.0~1.4，1.3=放大30%

    Returns:
        map_x, map_y: cv2.remap 用的位移图
    """
    if enlarge <= 1.0:
        map_x = np.tile(np.arange(img_w, dtype=np.float32), (img_h, 1))
        map_y = np.tile(np.arange(img_h, dtype=np.float32).reshape(-1, 1), (1, img_w))
        return map_x, map_y

    kps = np.array(face_kps)

    map_x = np.tile(np.arange(img_w, dtype=np.float32), (img_h, 1))
    map_y = np.tile(np.arange(img_h, dtype=np.float32).reshape(-1, 1), (1, img_w))

    for eye_idx in [LEFT_EYE, RIGHT_EYE]:
        eye_pts = kps[eye_idx]  # (6, 3)
        eye_pts[..., 0] *= img_w
        eye_pts[..., 1] *= img_h

        # 眼睛中心
        eye_cx = np.mean(eye_pts[:, 0])
        eye_cy = np.mean(eye_pts[:, 1])

        # 眼睛宽度
        eye_w = np.max(eye_pts[:, 0]) - np.min(eye_pts[:, 0])
        eye_h = np.max(eye_pts[:, 1]) - np.min(eye_pts[:, 1])

        # sigma = 眼睛宽度的1.5倍
        sigma = max(eye_w, eye_h) * 1.5

        # 计算每个像素到眼睛中心的距离
        yy, xx = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij')
        dist = np.sqrt((xx - eye_cx) ** 2 + (yy - eye_cy) ** 2)

        # 高斯权重
        weight = np.exp(-0.5 * (dist / sigma) ** 2)

        # 位移: 水平+垂直同时放大(防斗鸡眼)
        dx = (xx - eye_cx) * (enlarge - 1) * weight * 0.8
        dy = (yy - eye_cy) * (enlarge - 1) * weight * 0.4  # 垂直放大减半，避免不自然

        map_x += dx
        map_y += dy

    return map_x, map_y


# ============================================================
# 磨皮效果 (双边滤波)
# ============================================================
def apply_skin_smooth(frame, face_kps, strength=0.5, d=7, sigmaColor=15, sigmaSpace=15):
    """对皮肤区域应用双边滤波保边平滑

    Args:
        frame: BGR图像
        face_kps: 468个关键点，normalized (0~1)
        strength: 0.0~1.0，混合强度
        d: 双边滤波直径
        sigmaColor: 颜色空间标准差
        sigmaSpace: 坐标空间标准差

    Returns:
        磨皮后的图像 (uint8)
    """
    if strength <= 0:
        return frame

    kps = np.array(face_kps)
    h, w = frame.shape[:2]

    # 人脸 bounding box (归一化→像素)
    face_pts = kps[:17]  # 人脸外轮廓
    xs = face_pts[:, 0] * w
    ys = face_pts[:, 1] * h
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # 扩展padding
    pad_x = (x_max - x_min) * 0.3
    pad_y = (y_max - y_min) * 0.2
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)

    x_min, x_max = int(x_min), int(x_max)
    y_min, y_max = int(y_min), int(y_max)

    if x_max - x_min < 10 or y_max - y_min < 10:
        return frame

    # 提取人脸区域
    roi = frame[y_min:y_max, x_min:x_max]

    # 双边滤波
    smooth_roi = cv2.bilateralFilter(roi, d, sigmaColor, sigmaSpace)

    # 混合
    result = frame.copy()
    result[y_min:y_max, x_min:x_max] = (
        roi * (1 - strength) + smooth_roi * strength
    ).astype(np.uint8)

    return result


# ============================================================
# 整合: 逐帧处理人脸变形
# ============================================================
def process_frame_face_warp(frame, face_kps, config):
    """对人脸帧应用所有变形: V-face + 大眼 + 磨皮

    Args:
        frame: BGR图像
        face_kps: 468个关键点 (normalized 0~1, 每点 [x,y,z])
        config: {
            'v_face_strength': 0.0~0.75,
            'eye_enlarge': 1.0~1.4,
            'skin_smooth_strength': 0.0~1.0,
        }

    Returns:
        变形后的图像
    """
    h, w = frame.shape[:2]

    result = frame.copy()

    # 1. V-face — 应用位移图
    v_strength = config.get('v_face_strength', 0.0)
    if v_strength > 0:
        map_x, map_y = create_vface_displacement_map(face_kps, h, w, v_strength)
        result = cv2.remap(result, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 2. 大眼 — 应用位移图
    eye_enlarge = config.get('eye_enlarge', 1.0)
    if eye_enlarge > 1.0:
        map_x, map_y = create_eye_enlarge_displacement_map(face_kps, h, w, eye_enlarge)
        result = cv2.remap(result, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 3. 磨皮
    smooth_strength = config.get('skin_smooth_strength', 0.0)
    if smooth_strength > 0:
        result = apply_skin_smooth(result, face_kps, smooth_strength)

    return result

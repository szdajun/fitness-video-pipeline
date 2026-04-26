"""MediaPipe FaceMesh 检测器包装

提供人脸 landmarks 检测，主要用于美颜阶段：
- 眼部精确定位（瞳孔、眼眶）
- 面部中心和平面角度
- 眉毛、嘴唇等细部位置

使用 mp.solutions.face_mesh 经典 API，无需下载额外模型。
"""

import numpy as np
import cv2
import mediapipe as mp


class FaceMeshDetector:
    """MediaPipe FaceMesh 检测器，按帧调用"""

    # MediaPipe FaceMesh landmark 索引（468个landmarks）
    # 关键 landmarks:
    LANDMARK_LEFT_EYE = 33       # 左眼中心
    LANDMARK_RIGHT_EYE = 263      # 右眼中心
    LANDMARK_LEFT_PUPIL = 468    # 左瞳孔
    LANDMARK_RIGHT_PUPIL = 473   # 右瞳孔
    LANDMARK_NOSE_TIP = 1       # 鼻尖
    LANDMARK_LEFT_MOUTH = 61     # 左嘴角
    LANDMARK_RIGHT_MOUTH = 291   # 右嘴角
    LANDMARK_UPPER_LIP = 13     # 上唇中心
    LANDMARK_FOREHEAD = 10       # 额头中心
    LANDMARK_LEFT_EYEBROW = 107  # 左眉中
    LANDMARK_RIGHT_EYEBROW = 336 # 右眉中

    def __init__(self, max_faces=1, refine_landmarks=True):
        """初始化 FaceMesh 检测器

        Args:
            max_faces: 最大检测人数
            refine_landmarks: 是否精细检测（包含瞳孔和虹膜）
        """
        self.base_options = None
        self.refine = refine_landmarks
        # 初始化时不创建 detector，在第一次 detect 时延迟创建
        self._detector = None

    def _get_detector(self):
        """延迟创建 detector"""
        if self._detector is None:
            self._detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=self.refine,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        return self._detector

    def detect(self, frame_bgr):
        """检测一帧中的人脸 landmarks

        Args:
            frame_bgr: BGR 格式图像 (numpy array)

        Returns:
            dict: {
                'left_eye': (x, y) 像素坐标,
                'right_eye': (x, y) 像素坐标,
                'left_pupil': (x, y),
                'right_pupil': (x, y),
                'nose_tip': (x, y),
                'face_center': (x, y),
                'eye_distance': 双眼中心距离(px),
                'all': [[x, y, z, visibility], ...] 所有478个 landmarks
            }
            如果未检测到人脸，返回 None
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        with self._get_detector() as detector:
            results = detector.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        lms = landmarks  # 简写

        def get_xy(idx):
            """将归一化坐标转为像素坐标"""
            lm = lms[idx]
            return (int(lm.x * w), int(lm.y * h))

        def get_norm(idx):
            """返回归一化坐标"""
            lm = lms[idx]
            return (lm.x, lm.y, lm.z)

        # 双眼中心
        l_eye = get_xy(self.LANDMARK_LEFT_EYE)
        r_eye = get_xy(self.LANDMARK_RIGHT_EYE)
        eye_dist = np.sqrt((r_eye[0] - l_eye[0])**2 + (r_eye[1] - l_eye[1])**2)

        # 瞳孔（如果 refine_landmarks=True）
        l_pupil = get_xy(self.LANDMARK_LEFT_PUPIL) if self.refine else l_eye
        r_pupil = get_xy(self.LANDMARK_RIGHT_PUPIL) if self.refine else r_eye

        # 鼻尖
        nose = get_xy(self.LANDMARK_NOSE_TIP)

        # 面部中心 = 两眼中心的中点
        face_cx = (l_eye[0] + r_eye[0]) // 2
        face_cy = (l_eye[1] + r_eye[1]) // 2

        # 收集所有 landmarks 为 numpy 数组 [N, 4]
        all_landmarks = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
            dtype=np.float32
        )

        return {
            'left_eye': l_eye,
            'right_eye': r_eye,
            'left_pupil': l_pupil,
            'right_pupil': r_pupil,
            'nose_tip': nose,
            'face_center': (face_cx, face_cy),
            'eye_distance': eye_dist,
            'all': all_landmarks,
            'img_w': w,
            'img_h': h,
        }

    def detect_batch(self, frames):
        """批量检测（逐帧调用，不并行）

        Args:
            frames: list of BGR images

        Returns:
            list of detection results (or None if no face)
        """
        return [self.detect(f) for f in frames]


class FaceMeshTracker:
    """跨帧追踪人脸，提供平滑的 landmarks 序列

    用于视频处理，保持连续性。
    """

    def __init__(self, smooth_window=3):
        """初始化追踪器

        Args:
            smooth_window: 平滑窗口大小（帧数）
        """
        self.smooth_window = smooth_window
        self.history = []
        self.detector = FaceMeshDetector(refine_landmarks=True)

    def process_frame(self, frame_bgr):
        """处理一帧，返回平滑后的 landmarks

        Args:
            frame_bgr: BGR 图像

        Returns:
            平滑后的检测结果 dict（与 FaceMeshDetector.detect 格式相同）
            如果连续 N 帧未检测到，返回 None
        """
        det = self.detector.detect(frame_bgr)

        if det is None:
            # 未检测到人脸，尝试从历史恢复
            if self.history:
                # 返回最近一次平滑结果（略有衰减）
                last = self.history[-1]
                return last
            return None

        # 加入历史
        self.history.append(det)
        if len(self.history) > self.smooth_window:
            self.history.pop(0)

        # EMA 平滑 landmarks 位置
        smoothed = self._smooth_landmarks(self.history)
        return smoothed

    def _smooth_landmarks(self, history):
        """对历史序列做 EMA 平滑

        Args:
            history: list of detection dicts

        Returns:
            平滑后的 detection dict
        """
        n = len(history)
        if n == 0:
            return None

        last = history[-1].copy()
        last['left_eye'] = self._smooth_point(history, 'left_eye')
        last['right_eye'] = self._smooth_point(history, 'right_eye')
        last['left_pupil'] = self._smooth_point(history, 'left_pupil')
        last['right_pupil'] = self._smooth_point(history, 'right_pupil')
        last['nose_tip'] = self._smooth_point(history, 'nose_tip')
        last['face_center'] = self._smooth_point(history, 'face_center')

        # 重新计算 eye_distance
        dx = last['right_eye'][0] - last['left_eye'][0]
        dy = last['right_eye'][1] - last['left_eye'][1]
        last['eye_distance'] = np.sqrt(dx**2 + dy**2)

        return last

    def _smooth_point(self, history, key):
        """对某个关键点做 EMA 平滑"""
        points = [h[key] for h in history]
        # 线性加权：越近权重越大
        weights = np.linspace(0.5, 1.0, len(points))
        weights /= weights.sum()
        cx = sum(p[0] * w for p, w in zip(points, weights))
        cy = sum(p[1] * w for p, w in zip(points, weights))
        return (int(cx), int(cy))

    def reset(self):
        """重置追踪器状态"""
        self.history = []


if __name__ == "__main__":
    # 简单测试
    import os

    detector = FaceMeshDetector()

    # 测试合成图片
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (200, 150), (440, 350), (200, 180, 160), -1)
    cv2.circle(test_img, (290, 210), 12, (80, 60, 40), -1)
    cv2.circle(test_img, (350, 210), 12, (80, 60, 40), -1)

    result = detector.detect(test_img)
    if result:
        print(f"Eye distance: {result['eye_distance']:.1f}px")
        print(f"Left eye: {result['left_eye']}, Right eye: {result['right_eye']}")
        print(f"Nose: {result['nose_tip']}")
        print(f"Face center: {result['face_center']}")
    else:
        print("No face detected (synthetic image, expected)")

    print("FaceMesh module OK")
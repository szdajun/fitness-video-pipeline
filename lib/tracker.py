"""人物跟踪: 领操人物识别 + 卡尔曼滤波平滑"""

import numpy as np
from filterpy.kalman import KalmanFilter


def identify_lead_person(all_people_kps, frame_width, frame_height):
    """从多人检测结果中识别领操人物

    策略: 选择身体面积最大的人（离镜头最近，最突出）
    """
    if not all_people_kps or len(all_people_kps) == 0:
        return 0

    if len(all_people_kps) == 1:
        return 0

    best_idx = 0
    best_score = -1

    for i, kps in enumerate(all_people_kps):
        kps = np.array(kps)
        vis_mask = kps[:, 2] > 0.5
        if vis_mask.sum() < 5:
            continue

        left_shoulder = kps[11]
        right_shoulder = kps[12]
        nose = kps[0]
        left_ankle = kps[27]
        right_ankle = kps[28]

        shoulder_w = abs(right_shoulder[0] - left_shoulder[0])
        body_h = abs((left_ankle[1] + right_ankle[1]) / 2 - nose[1])

        visibility = vis_mask.sum() / len(kps)
        score = shoulder_w * body_h * visibility

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


class LeadPersonSmoother:
    """领操人物切换平滑器: 防止抖动

    只有当新候选人连续出现 CONSECUTIVE 帧后才真正切换，
    避免单帧检测抖动导致的画面跳动。
    """

    CONSECUTIVE = 5

    def __init__(self):
        self.current_lead = 0
        self.candidate = 0
        self.candidate_count = 0

    def update(self, detected_lead: int) -> int:
        """输入当前帧检测的 lead index，返回平滑后的 lead index"""
        if detected_lead == self.candidate:
            self.candidate_count += 1
            if self.candidate_count >= self.CONSECUTIVE:
                if self.current_lead != self.candidate:
                    self.current_lead = self.candidate
                    self.candidate_count = 0
        else:
            self.candidate = detected_lead
            self.candidate_count = 1
        return self.current_lead


class SmoothTracker:
    """卡尔曼滤波平滑跟踪器"""

    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        self.kf.P *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[0, 0] = process_noise
        self.kf.Q[1, 1] = process_noise
        self.kf.R[0, 0] = measurement_noise
        self.kf.R[1, 1] = measurement_noise

        self.initialized = False

    def update(self, x, y):
        """输入观测值，返回平滑后的 (x, y)"""
        if not self.initialized:
            self.kf.x = np.array([x, y, 0, 0])
            self.initialized = True
            return x, y

        self.kf.predict()
        self.kf.update([x, y])

        return self.kf.x[0], self.kf.x[1]

"""YOLOv8-pose 姿态检测 — 替代 MediaPipe BlazePose

COCO 17关键点 → BlazePose 33关键点映射:
  COCO:  0鼻子 1左眼 2右眼 3左耳 4右耳
         5左肩 6右肩 7左肘 8右肘 9左腕 10右腕
         11左髋 12右髋 13左膝 14右膝 15左踝 16右踝
  BlazePose 33点中对应关系:
    0(nose)       → 0
    5(l_sholder)  → 11
    6(r_shoulder) → 12
    11(l_hip)     → 23
    12(r_hip)     → 24
    15(l_ankle)   → 27
    16(r_ankle)   → 28
    其余(脸/手/脚) → 以肩髋中心估算或填(0,0,0)
"""

import numpy as np

# COCO 17 → BlazePose 33 映射
# BlazePose indices that matter for body size / lead detection:
# 0(nose), 11(l_shoulder), 12(r_shoulder), 23(l_hip), 24(r_hip), 27(l_ankle), 28(r_ankle)
COCO_TO_BLAZEPOSE = {
    0: 0,    # nose
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    11: 23,  # left_hip
    12: 24,  # right_hip
    15: 27,  # left_ankle
    16: 28,  # right_ankle
    # COCO doesn't have face/hands/feet - those stay at (0,0,0)
}


def coco17_to_blaze33(coco_kps: np.ndarray) -> np.ndarray:
    """将 COCO [17, 3] 转换为 BlazePose [33, 3]

    coco_kps: shape (17, 3) — [x, y, visibility]
    返回: shape (33, 3)
    """
    blaze = np.zeros((33, 3), dtype=np.float32)

    for coco_idx, blaze_idx in COCO_TO_BLAZEPOSE.items():
        blaze[blaze_idx] = coco_kps[coco_idx]

    # 估算脸部长度和手部/脚部关键点（pipeline 不严格依赖这些）
    # 鼻→左肩中点 作为颈部顶端（粗略）
    neck_top = (blaze[11] + blaze[12]) / 2
    blaze[1] = blaze[0]  # 眉心用鼻子代替

    # 手腕/肘部填肩点附近（pipeline不用）
    blaze[16] = blaze[11]  # left_elbow
    blaze[17] = blaze[12]  # right_elbow
    blaze[18] = blaze[11]  # left_wrist
    blaze[19] = blaze[12]  # right_wrist
    blaze[20] = blaze[23]  # left_foot_index / ankle
    blaze[21] = blaze[24]  # right_foot_index / ankle

    # 可见性: 如果原始点 visibility>0.3 则认为该点有效
    for coco_idx, blaze_idx in COCO_TO_BLAZEPOSE.items():
        blaze[blaze_idx][2] = 1.0 if coco_kps[coco_idx][2] > 0.3 else 0.0

    return blaze


def coco_to_blaze_batch(coco_batch: np.ndarray) -> list:
    """批量转换: coco_batch shape (N_people, 17, 3) → list of (33, 3)"""
    return [coco17_to_blaze33(person) for person in coco_batch]

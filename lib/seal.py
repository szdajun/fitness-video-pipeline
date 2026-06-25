"""汉印水印叠加 — stub (原文件丢失，保留接口兼容)"""

import cv2
import numpy as np


def overlay_seal(frame: np.ndarray, text: str = "", pos: str = "top-left",
                 size: int = 130, margin: int = 30, alpha: float = 0.70,
                 **kwargs) -> np.ndarray:
    """叠加汉印到帧上（当前为存根实现，直接返回原帧）"""
    return frame

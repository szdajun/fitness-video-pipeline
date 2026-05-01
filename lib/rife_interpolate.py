"""OpenCV 光流帧插值（替代原 RIFE 空壳）

原设计为 RIFE 模型接口，因模型权重过大改用 OpenCV Farneback 光流。
无需额外依赖，在运动平滑场景（健身视频）效果优于简单混合。

配置 (rife 段):
  enabled: false     # 默认关闭
  target_fps: 60
  gpu: true          # 无实际作用，保留兼容
  half_precision: true
"""

import cv2
import numpy as np
from pathlib import Path

# Farneback 光流参数
_FARNEBACK_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15,
                         iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


class RIFEInterpolator:
    """帧插值器 — OpenCV Farneback 光流实现"""

    def __init__(self, gpu=True, half=True):
        self.gpu = gpu
        self.half = half
        self._model = None

    @staticmethod
    def _has_gpu() -> bool:
        return False  # 光流实现不依赖 GPU

    def is_available(self) -> bool:
        """始终可用（纯 OpenCV，无额外依赖）"""
        return True

    def load(self):
        """初始化（兼容原接口）"""
        self._model = True

    @staticmethod
    def _interpolate_frames(frame0: np.ndarray, frame1: np.ndarray,
                            t: float = 0.5) -> np.ndarray:
        """使用 Farneback 双向光流插值中间帧

        Args:
            frame0: 前一帧 (BGR)
            frame1: 后一帧 (BGR)
            t: 插值位置 (0~1), 0.5=正中间

        Returns:
            插值帧 (BGR)
        """
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        h, w = frame0.shape[:2]

        try:
            # 前向光流 I0 → I1
            flow_fwd = cv2.calcOpticalFlowFarneback(gray0, gray1, None,
                                                     **_FARNEBACK_PARAMS)
            # 后向光流 I1 → I0
            flow_bwd = cv2.calcOpticalFlowFarneback(gray1, gray0, None,
                                                     **_FARNEBACK_PARAMS)

            # 生成像素坐标网格
            x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                               np.arange(h, dtype=np.float32))

            # 前向 warp：I0 沿流向 I1 方向移动 t
            map_fwd_x = x + flow_fwd[:, :, 0] * t
            map_fwd_y = y + flow_fwd[:, :, 1] * t
            warped0 = cv2.remap(frame0, map_fwd_x, map_fwd_y,
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            # 后向 warp：I1 沿流向 I0 方向移动 (1-t)
            map_bwd_x = x + flow_bwd[:, :, 0] * (1.0 - t)
            map_bwd_y = y + flow_bwd[:, :, 1] * (1.0 - t)
            warped1 = cv2.remap(frame1, map_bwd_x, map_bwd_y,
                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            return cv2.addWeighted(warped0, 1.0 - t, warped1, t, 0)
        except cv2.error:
            # 光流失败 → 退化到线性混合
            return cv2.addWeighted(frame0, 1.0 - t, frame1, t, 0)

    def interpolate_video(self, input_path: str, output_dir: str,
                          src_fps: float, target_fps: float,
                          max_frames: int = 0) -> int:
        """对视频帧做插值，输出 PNG 序列

        Args:
            input_path: 输入视频路径
            output_dir: 输出 PNG 目录
            src_fps: 源帧率 (如 30)
            target_fps: 目标帧率 (如 60)
            max_frames: 最大处理帧数 (0=全部)

        Returns:
            输出帧总数
        """
        self.load()
        cap = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames > 0:
            total = min(total, max_frames)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 帧率比 = 每对原帧之间插入 (ratio-1) 个中间帧
        ratio = target_fps / src_fps
        insert_count = max(1, round(ratio) - 1)

        # 读第一帧
        ret, prev = cap.read()
        if not ret:
            cap.release()
            return 0

        cv2.imwrite(str(out_dir / "f_000000.png"), prev)
        out_idx = 1

        for fi in range(1, total):
            ret, curr = cap.read()
            if not ret:
                break

            # 插入中间帧
            for j in range(1, insert_count + 1):
                t = j / (insert_count + 1)
                mid = self._interpolate_frames(prev, curr, t)
                cv2.imwrite(str(out_dir / f"f_{out_idx:06d}.png"), mid)
                out_idx += 1

            # 写当前帧
            cv2.imwrite(str(out_dir / f"f_{out_idx:06d}.png"), curr)
            out_idx += 1
            prev = curr

            if fi % 200 == 0:
                print(f"    插值进度: {fi}/{total}")

        cap.release()
        return out_idx

    def interpolate_pairs(self, frames, t=0.5):
        """批量插值帧对

        Args:
            frames: [(f0, f1), ...] 帧对列表
            t: 插值位置 (0~1)

        Returns:
            [mid_frame, ...]
        """
        return [self._interpolate_frames(f0, f1, t) for f0, f1 in frames]

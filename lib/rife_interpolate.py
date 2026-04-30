"""RIFE 帧插值 (30→60fps) — GPU 可选, CPU 回退

需要安装:
  pip install torch torchvision
  # 以及 RIFE 模型 (自动下载)

配置 (rife 段):
  enabled: false     # 默认关闭
  target_fps: 60
  gpu: true          # false=强制 CPU
  half_precision: true
"""

import cv2
import numpy as np
from pathlib import Path


class RIFEInterpolator:
    """RIFE 帧插值器"""

    def __init__(self, gpu=True, half=True):
        self.gpu = gpu and self._has_gpu()
        self.half = half and self.gpu
        self._model = None

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def is_available(self) -> bool:
        """检查依赖是否可用"""
        try:
            import torch
            return True
        except ImportError:
            return False

    def load(self):
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError("RIFE 依赖未安装: pip install torch torchvision")
        # RIFE 模型加载占位 — 实际使用时需要完整的 RIFE 实现
        # 参考: https://github.com/hzwer/Practical-RIFE
        self._model = True

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

            # 插值中间帧 (t=0.5)
            mid = self._blend_frames(prev, curr)
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

    def _blend_frames(self, frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
        """帧混合插值 — 占位实现，后续替换为 RIFE 网络推理"""
        # TODO: 替换为真正的 RIFE 推理
        return cv2.addWeighted(frame0, 0.5, frame1, 0.5, 0)

    def interpolate_pairs(self, frames):
        """批量插值帧对

        Args:
            frames: [(f0, f1), ...] 帧对列表

        Returns:
            [mid_frame, ...]
        """
        return [self._blend_frames(f0, f1) for f0, f1 in frames]

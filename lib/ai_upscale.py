"""Real-ESRGAN AI 超分封装 (GPU 可选，CPU 回退)

需要安装:
  pip install realesrgan basicsr

配置 (output 段):
  upscale_mode: realesrgan   # 或 lanczos(默认)
  realesrgan_model: realesrgan-x4plus
  realesrgan_scale: 2
  realesrgan_tile: 256
  realesrgan_gpu: true       # false=强制 CPU
"""

import numpy as np
import cv2


class AIUpscaler:
    """Real-ESRGAN 超分器，CPU/GPU 双路径"""

    def __init__(self, model_name="realesrgan-x4plus", scale=2,
                 tile=256, gpu=True):
        self.model_name = model_name
        self.scale = scale
        self.tile = tile
        self.gpu = gpu and self._has_gpu()
        self._model = None

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def is_available(self) -> bool:
        try:
            import realesrgan
            return True
        except ImportError:
            return False

    def load(self):
        if self._model is not None:
            return
        if not self.is_available():
            raise RuntimeError("Real-ESRGAN 未安装: pip install realesrgan basicsr")
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        bg_model = None
        device = "cuda" if self.gpu else "cpu"
        self._model = RealESRGANer(
            scale=4,
            model_path=None,  # 自动下载
            model=model,
            bg_model=bg_model,
            tile=self.tile,
            tile_pad=10,
            pre_pad=0,
            half=self.gpu,
            device=device,
        )

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """超分单帧，返回 uint8 BGR"""
        if self._model is None:
            self.load()
        output, _ = self._model.enhance(frame, outscale=self.scale)
        return output

    @staticmethod
    def need_upscale(in_w: int, in_h: int, out_w: int, out_h: int) -> bool:
        """检查是否真的需要超分"""
        return (out_w and out_h) and (in_w < out_w or in_h < out_h)

    @staticmethod
    def preprocess(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """确保超分后尺寸精确匹配，避免非对齐警告"""
        h, w = frame.shape[:2]
        if w != target_w or h != target_h:
            frame = cv2.resize(frame, (target_w, target_h),
                               interpolation=cv2.INTER_LANCZOS4)
        return frame

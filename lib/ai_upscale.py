"""Real-ESRGAN AI 超分封装 (GPU 可选，CPU 回退)

Python 版: pip install realesrgan basicsr (慢, 适合单图)
ncnn 版: tools/realesrgan-ncnn-vulkan/realesrgan-ncnn-vulkan.exe (快, 适合视频)
"""

import numpy as np
import cv2
import subprocess, shutil, tempfile, os
from pathlib import Path

NCNN_EXE = Path(__file__).parent.parent / "tools" / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe"


def upscale_video_ncnn(input_video: str, output_video: str,
                       scale: int = 2, gpu_id: int = 2,
                       model: str = "realesrgan-x4plus") -> bool:
    """用 realesrgan-ncnn-vulkan 超分整个视频 (fast, Vulkan GPU)

    Returns: True on success
    """
    if not NCNN_EXE.exists():
        return False

    # Use F: drive for temp (avoid filling C:)
    tmpdir = Path(tempfile.mkdtemp(prefix="ncnn_video_",
                 dir="F:/wkspace/fitness-video-pipeline/_temp"))
    frames_in = tmpdir / "in"
    frames_out = tmpdir / "out"
    frames_in.mkdir()
    frames_out.mkdir()

    try:
        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if total_frames == 0:
            return False

        print(f"    ncnn: {w}x{h} {total_frames}f @ {fps:.1f}fps -> {scale}x")

        # Step 1: Extract frames
        r = subprocess.run([
            ffmpeg, "-y", "-i", input_video,
            "-vsync", "0",
            str(frames_in / "f_%06d.png"),
        ], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"    ncnn extract error: {r.stderr[-200:]}")
            return False

        # Step 2: Run ncnn on entire directory
        r = subprocess.run([
            str(NCNN_EXE),
            "-i", str(frames_in),
            "-o", str(frames_out),
            "-s", str(scale),
            "-g", str(gpu_id),
            "-n", model,
            "-f", "png",
        ], capture_output=True, text=True, encoding="utf-8", errors="replace",
           timeout=7200)
        if r.returncode != 0:
            print(f"    ncnn error: {r.stderr[-300:]}")
            return False

        # Step 3: Encode upscaled frames
        upscaled = list(frames_out.glob("*.png"))
        if not upscaled:
            print("    ncnn: no output frames")
            return False

        first = cv2.imread(str(upscaled[0]))
        out_h, out_w = first.shape[:2]
        print(f"    ncnn: {out_w}x{out_h}, {len(upscaled)} frames upscaled")

        r = subprocess.run([
            ffmpeg, "-y", "-framerate", str(fps),
            "-i", str(frames_out / "f_%06d.png"),
            "-i", input_video,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "16",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-shortest",
            output_video,
        ], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"    ncnn encode error: {r.stderr[-200:]}")
            return False

        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class AIUpscaler:
    """Real-ESRGAN 超分器，CPU/GPU 双路径"""

    _DEFAULT_MODEL = Path(__file__).parent.parent / "weights" / "RealESRGAN_x4plus.pth"

    def __init__(self, model_name="realesrgan-x4plus", scale=2,
                 tile=256, gpu=True, model_path=None):
        self.model_name = model_name
        self.scale = scale
        self.tile = tile
        self.gpu = gpu and self._has_gpu()
        self._model_path = model_path or str(self._DEFAULT_MODEL)
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
        if not Path(self._model_path).exists():
            raise RuntimeError(f"模型文件不存在: {self._model_path}")
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        device = "cuda" if self.gpu else "cpu"
        self._model = RealESRGANer(
            scale=4,
            model_path=self._model_path,
            model=model,
            tile=self.tile,
            tile_pad=10,
            pre_pad=0,
            half=self.gpu,
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

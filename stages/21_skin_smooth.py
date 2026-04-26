"""阶段21: 皮肤区域磨皮（YCrCb肤色检测 + 双边滤波）

改进：
- 使用 YCrCb 颜色空间检测皮肤区域
- 只在皮肤区域应用双边滤波，背景保持清晰
- 可选：仅面部区域磨皮（face_only模式）

用法（在配置中）:
    skin_smooth:
      enabled: true
      strength: 0.0-1.0    # 磨皮强度，默认0.3
      d: 9                 # 双边滤波直径（增大让过渡更平滑）
      sigmaColor: 20      # 颜色空间标准差
      sigmaSpace: 20      # 坐标空间标准差
      downscale: 0.5      # 降采样比例，<1.0加速
      skin_detect: true   # 皮肤检测模式，false则全局磨皮
      face_only: false    # 仅面部区域磨皮（需要额外的人脸检测）
"""

import cv2
import numpy as np
from pathlib import Path

from lib.utils import create_writer


def detect_skin_ycrcb(frame):
    """使用 YCrCb 颜色空间检测皮肤区域

    Returns:
        skin_mask: uint8, 0=非皮肤, 255=皮肤
    """
    h, w = frame.shape[:2]
    skin_mask = np.zeros((h, w), dtype=np.uint8)

    # 转换到 YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # 皮肤颜色范围（经验值）
    y, cr, cb = cv2.split(ycrcb)

    # Cr: 133-173, Cb: 77-127
    skin_mask = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8) * 255

    # 形态学处理：去噪 + 平滑边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # 高斯模糊边缘，让过渡更自然
    skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 3)

    return skin_mask


def apply_skin_smooth(frame, strength=0.5, d=9, sigmaColor=20, sigmaSpace=20,
                       downscale=0.5, skin_detect=True):
    """皮肤感知磨皮

    Args:
        frame: BGR图像
        strength: 0.0~1.0，混合强度。0=原图，1=完全磨皮
        d: 双边滤波直径
        sigmaColor: 颜色空间标准差
        sigmaSpace: 坐标空间标准差
        downscale: 降采样比例
        skin_detect: 是否启用皮肤检测（false则全局磨皮）

    Returns:
        磨皮后的图像 (uint8)
    """
    if strength <= 0:
        return frame
    if strength >= 1.0:
        strength = 1.0

    h, w = frame.shape[:2]

    # 皮肤检测
    if skin_detect:
        skin_mask = detect_skin_ycrcb(frame) / 255.0  # 归一化到 0-1

        # 降采样处理
        if downscale < 1.0:
            small_w, small_h = int(w * downscale), int(h * downscale)
            small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(skin_mask.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_AREA) / 255.0

            # 双边滤波
            smoothed = cv2.bilateralFilter(small, d, sigmaColor, sigmaSpace)

            # 只在皮肤区域混合 (small_mask needs to be 3D for broadcasting)
            result = small * (1 - small_mask[:, :, None] * strength) + smoothed * (small_mask[:, :, None] * strength)
            result = np.clip(result, 0, 255).astype(np.uint8)
            return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # 全分辨率
            smoothed = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
            result = frame * (1 - skin_mask[:, :, None] * strength) + smoothed * (skin_mask[:, :, None] * strength)
            return np.clip(result, 0, 255).astype(np.uint8)
    else:
        # 全局磨皮（传统方式）
        if downscale < 1.0:
            small_w, small_h = int(w * downscale), int(h * downscale)
            small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
            smoothed = cv2.bilateralFilter(small, d, sigmaColor, sigmaSpace)
            result = cv2.addWeighted(small, 1 - strength, smoothed, strength, 0)
            return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            smoothed = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
            return cv2.addWeighted(frame, 1 - strength, smoothed, strength, 0)


class SkinSmoothStage:
    def run(self, ctx):
        cfg = ctx.config.get("skin_smooth", {})
        if not cfg.get("enabled", False):
            ctx.set("skin_smooth_path", ctx.get("beatflash_path") or ctx.get("ken_burns_path"))
            return

        strength = cfg.get("strength", 0.3)
        d = cfg.get("d", 9)
        sigmaColor = cfg.get("sigmaColor", 20)
        sigmaSpace = cfg.get("sigmaSpace", 20)
        downscale = cfg.get("downscale", 0.5)
        skin_detect = cfg.get("skin_detect", True)

        input_path = (ctx.get("beatflash_path") or
                      ctx.get("ken_burns_path") or
                      ctx.get("warped_path") or
                      str(ctx.input_path))  # 横屏 fallback
        if not cv2.VideoCapture(input_path).isOpened():
            print("    跳过: 无输入视频")
            ctx.set("skin_smooth_path", ctx.get("beatflash_path") or ctx.get("ken_burns_path") or str(ctx.input_path))
            return

        if strength <= 0:
            print(f"    磨皮强度为0，跳过")
            ctx.set("skin_smooth_path", input_path)
            return

        print(f"    磨皮: strength={strength}, d={d}, skin_detect={skin_detect}, downscale={downscale}")

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        input_video = cv2.VideoCapture(input_path)
        if not input_video.isOpened():
            print(f"    错误: 无法打开视频 {input_path}")
            return

        temp_path = ctx.output_dir / f"{Path(input_path).stem}_smooth.mp4"
        fw = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = create_writer(str(temp_path), fps, fw, fh)

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = input_video.read()
            if not ret:
                break

            processed = apply_skin_smooth(frame, strength, d, sigmaColor, sigmaSpace,
                                        downscale, skin_detect)
            writer.write(processed)

            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"    {frame_idx}/{max_frames}")

        input_video.release()
        writer.release()

        ctx.set("skin_smooth_path", str(temp_path))
        print(f"    输出: {temp_path.name}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    class Ctx:
        def __init__(self):
            self.config = {
                "skin_smooth": {
                    "enabled": True,
                    "strength": 0.3,
                    "d": 9,
                    "sigmaColor": 20,
                    "sigmaSpace": 20,
                    "skin_detect": True
                }
            }
            self._data = {}
            self.output_dir = Path("output/2026-04-22")
            self.video_info = {"fps": 30, "frames": 100, "process_frames": 100}

        def get(self, k, default=None):
            return self._data.get(k, default)

        def set(self, k, v):
            self._data[k] = v

    ctx = Ctx()
    stage = SkinSmoothStage()
    stage.run(ctx)
    print(f"完成: {ctx.get('skin_smooth_path')}")

"""阶段21: 全局磨皮（双边滤波保边平滑）

对视频帧应用双边滤波，在保边的同时平滑皮肤区域。
不需要人脸检测，直接全局处理，按 strength 混合。
使用 0.5x 降采样加速，处理时间减少 2/3。

用法（在配置中）:
    skin_smooth:
      enabled: true
      strength: 0.0-1.0    # 磨皮强度，默认0.3
      d: 7                 # 双边滤波直径
      sigmaColor: 15      # 颜色空间标准差
      sigmaSpace: 15      # 坐标空间标准差
      downscale: 0.5      # 降采样比例，<1.0加速，默认1.0不降
"""

import cv2
import numpy as np
from pathlib import Path

from lib.utils import create_writer


def apply_skin_smooth(frame, strength=0.5, d=7, sigmaColor=15, sigmaSpace=15, downscale=1.0):
    """对整帧应用双边滤波磨皮

    Args:
        frame: BGR图像
        strength: 0.0~1.0，混合强度。0=原图，1=完全磨皮
        d: 双边滤波直径（越大越平滑但越慢）
        sigmaColor: 颜色空间标准差
        sigmaSpace: 坐标空间标准差
        downscale: 降采样比例，<1.0时先缩小再放大以加速

    Returns:
        磨皮后的图像 (uint8)
    """
    if strength <= 0:
        return frame
    if strength >= 1.0:
        strength = 1.0

    h, w = frame.shape[:2]
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
        d = cfg.get("d", 7)
        sigmaColor = cfg.get("sigmaColor", 15)
        sigmaSpace = cfg.get("sigmaSpace", 15)
        downscale = cfg.get("downscale", 1.0)

        input_path = (ctx.get("beatflash_path") or
                      ctx.get("ken_burns_path") or
                      ctx.get("warped_path") or
                      str(ctx.input_path))  # 横屏 fallback
        # Windows pathlib bug: Path.exists() 返回 False 但 cv2.VideoCapture 能打开
        if not cv2.VideoCapture(input_path).isOpened():
            print("    跳过: 无输入视频")
            ctx.set("skin_smooth_path", ctx.get("beatflash_path") or ctx.get("ken_burns_path") or str(ctx.input_path))
            return

        if strength <= 0:
            print(f"    磨皮强度为0，跳过")
            ctx.set("skin_smooth_path", input_path)
            return

        print(f"    磨皮: strength={strength}, d={d}, downscale={downscale}")

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

            processed = apply_skin_smooth(frame, strength, d, sigmaColor, sigmaSpace, downscale)
            writer.write(processed)

            frame_idx += 1
            if frame_idx % 500 == 0:
                print(f"    {frame_idx}/{max_frames}")

        input_video.release()
        writer.release()

        ctx.set("skin_smooth_path", str(temp_path))
        print(f"    输出: {temp_path.name}")


if __name__ == "__main__":
    # 独立运行测试
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    class Ctx:
        def __init__(self):
            self.config = {
                "skin_smooth": {
                    "enabled": True,
                    "strength": 0.3,
                    "d": 7,
                    "sigmaColor": 15,
                    "sigmaSpace": 15
                }
            }
            self._data = {}
            self.output_dir = Path("output/2026-04-22")
            self.video_info = {"fps": 30, "frames": 100, "process_frames": 100}

        def get(self, k, default=None):
            return self._data.get(k, default)

        def set(self, k, v):
            self._data[k] = v

        def config(self):
            return self.config

    ctx = Ctx()
    stage = SkinSmoothStage()
    stage.run(ctx)
    print(f"完成: {ctx.get('skin_smooth_path')}")

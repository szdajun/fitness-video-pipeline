"""阶段23: 视频降噪（GPU 加速版）

使用 FFmpeg hqdn3d 3D 降噪滤镜，支持 GPU 硬件编码。
相比 OpenCV fastNlMeans CPU 版本快 50-100 倍。

配置:
  denoise_strength: 0~20, 降噪强度（默认3，夜景建议8~15）
  denoise_mode: hqdn3d (默认)
"""

import cv2
from lib.utils import path_exists
import subprocess
import shutil
from pathlib import Path


def _hqdn3d_params(strength: int) -> str:
    """将 denoise_strength 映射到 hqdn3d 参数"""
    if strength <= 3:
        return "2:1:0:0"        # 仅空域降噪，关闭时域避免色块跳动
    elif strength <= 8:
        return "4:3:0:0"
    elif strength <= 15:
        return "6:4:0:0"
    else:
        return "8:6:0:0"


def _probe_nvenc() -> bool:
    """检测 h264_nvenc 编码器是否可用"""
    for ffmpeg in ["ffmpeg", "C:/Users/18091/ffmpeg/ffmpeg.exe"]:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             "-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "1M", "-an", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        if r.returncode == 0:
            return True
    return False


class DenoiseStage:
    def run(self, ctx):
        if ctx.get("denoise_path") and path_exists(ctx.get("denoise_path")):
            print("    已存在，跳过")
            return

        input_path = (
            ctx.get("skin_smooth_path") or
            ctx.get("color_path") or
            ctx.get("skin_tone_filter_path") or
            ctx.get("ken_burns_path") or
            ctx.get("face_warp_path") or
            ctx.get("warped_path") or
            ctx.get("h2v_path") or
            str(ctx.input_path)
        )
        if not input_path or not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("denoise_path", None)
            return

        video_info = ctx.get("video_info")
        fps = video_info["fps"]

        cap_check = cv2.VideoCapture(input_path)
        if not cap_check.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")
        cap_check.release()

        cfg = ctx.config.get("denoise", {})
        strength = cfg.get("denoise_strength", 3)

        if strength <= 0:
            print("    跳过: denoise_strength=0")
            ctx.set("denoise_path", input_path)
            return

        out_path = ctx.output_dir / f"{Path(input_path).stem}_denoise.mp4"
        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        params = _hqdn3d_params(strength)

        # 中间编码用 libx264 CRF 1（无损），避免 NVENC 二次压缩色块放大
        enc_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "1"]
        enc_name = "libx264 (CRF 1)"

        print(f"    降噪: hqdn3d({params}), strength={strength}, encoder={enc_name}")

        cmd = [
            ffmpeg, "-y",
            "-i", input_path,
            "-vf", f"hqdn3d={params}",
            "-pix_fmt", "yuv420p",
            "-an",
        ] + enc_args + [str(out_path)]

        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")

        if result.returncode != 0:
            stderr = result.stderr[-500:]
            print(f"    FFmpeg 降噪失败: {stderr}")
            print(f"    回退: 直接复制输入")
            ctx.set("denoise_path", input_path)
            return

        if not Path(out_path).exists() or not path_exists(str(out_path)):
            print(f"    降噪输出无效，回退输入")
            ctx.set("denoise_path", input_path)
            return

        ctx.set("denoise_path", str(out_path))
        ctx.set("color_path", str(out_path))
        print(f"    输出: {out_path.name}")

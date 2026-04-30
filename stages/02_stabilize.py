"""阶段02: 视频稳定 (FFmpeg vidstab)

使用 FFmpeg 的 vidstab 滤镜稳定视频。
先分析运动向量，再应用稳定。
"""

import subprocess
import shutil
from pathlib import Path

from lib.utils import path_exists


class StabilizeStage:
    def run(self, ctx):
        stages_cfg = ctx.config.get("stages", {})
        if not stages_cfg.get("stabilize", True):
            print("    跳过: stabilize 已禁用")
            ctx.set("stabilized_path", None)
            return

        output_path = ctx.output_dir / f"{ctx.input_path.stem}_stabilized.mp4"
        if output_path.exists():
            print(f"    已存在，跳过")
            ctx.set("stabilized_path", str(output_path))
            return

        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        if not Path(ffmpeg).exists():
            print("    跳过: FFmpeg 未安装")
            ctx.set("stabilized_path", None)
            return

        video_path = str(ctx.input_path)
        vectors_path = ctx.output_dir / f"{ctx.input_path.stem}_vectors.trf"

        # 从配置文件读取 stabilize 参数 (位于 config.stabilize 而非 config.stages.stabilize)
        cfg = ctx.config.get("stabilize", {})
        shakiness = cfg.get("shakiness", 5)
        accuracy = cfg.get("accuracy", 10)
        zoom = cfg.get("zoom", 1)
        smoothing = cfg.get("smoothing", 10)

        print(f"    参数: shakiness={shakiness}, accuracy={accuracy}, smoothing={smoothing}")

        ctx.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 分析运动向量
        print(f"    分析运动向量...")
        cmd1 = [
            ffmpeg, "-y",
            "-i", video_path,
            "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={vectors_path}",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd1, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            err = (result.stderr or "")[-300:]
            print(f"    分析失败: {err}")
            return

        # Step 2: 应用稳定
        # 注意: optzoom=0 禁用自动缩放（防止帧间缩放不一致导致色块脉动）
        #       interpol=3 使用双三次插值（默认双线性插值质量差）
        #       zoom=1 固定无缩放（黑边由后续 ken_burns 裁切处理）
        print("    应用稳定...")
        cmd2 = [
            ffmpeg, "-y",
            "-i", video_path,
            "-vf", f"vidstabtransform=input={vectors_path}:zoom=1:optzoom=0:interpol=3:smoothing={smoothing}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "1",
            "-pix_fmt", "yuv444p",
            str(output_path),
        ]
        result = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            err = (result.stderr or "")[-300:]
            print(f"    稳定失败: {err}")
            return

        # 清理临时向量文件
        if vectors_path.exists():
            vectors_path.unlink()

        ctx.set("stabilized_path", str(output_path))
        print(f"    输出: {output_path.name}")

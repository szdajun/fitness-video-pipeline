"""阶段02: 视频稳定 (FFmpeg vidstab)

使用 FFmpeg 的 vidstab 滤镜稳定视频。
先分析运动向量，再应用稳定。
"""

import subprocess
import shutil
from pathlib import Path


class StabilizeStage:
    def run(self, ctx):
        # 检查全局开关
        stages_cfg = ctx.config.get("stages", {})
        if not stages_cfg.get("stabilize", True):
            print("    跳过: stabilize 已禁用")
            ctx.set("stabilized_path", None)
            return

        # 增量跳过：输出已存在则跳过
        if ctx.get("stabilized_path") and path_exists(ctx.get("stabilized_path")):
            print("    已存在，跳过")
            return

        # 优先用 PATH 中的 ffmpeg，没有则用用户目录下的
        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        if not path_exists(ffmpeg):
            print("    跳过: FFmpeg 未安装")
            return

        video_path = ctx.input_path
        video_info = ctx.get("video_info")
        cfg = ctx.config.get("stages", {}).get("stabilize", {})

        shakiness = cfg.get("shakiness", 5)
        accuracy = cfg.get("accuracy", 10)
        zoom = cfg.get("zoom", 1)
        smoothing = cfg.get("smoothing", 10)

        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        stabilized_path = ctx.output_dir / f"{video_path.stem}_stabilized.mp4"
        vectors_path = ctx.output_dir / f"{video_path.stem}_vectors.trf"

        # Step 1: 分析运动向量
        print(f"    分析运动向量 (shakiness={shakiness})...")
        cmd1 = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-vf", f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={vectors_path}",
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd1, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr or "")[-200:]
            print(f"    分析失败: {err}")
            return

        # Step 2: 应用稳定
        print("    应用稳定...")
        cmd2 = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-vf", f"vidstabtransform=input={vectors_path}:zoom={zoom}:smoothing={smoothing}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            str(stabilized_path),
        ]
        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr or "")[-200:]
            print(f"    稳定失败: {err}")
            return

        # 清理临时文件
        if vectors_path.exists():
            vectors_path.unlink()

        ctx.set("stabilized_path", str(stabilized_path))
        print(f"    输出: {stabilized_path.name}")
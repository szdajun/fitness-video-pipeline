"""阶段00: 源视频压缩去块预处理

用 FFmpeg spp 滤镜去除 DCT 压缩块，在处理链最前端运行。
spp 是专门针对视频编码压缩瑕疵的去块+去振铃滤镜。
"""

import shutil, subprocess
from pathlib import Path
from lib.utils import path_exists


class PreDeblockStage:
    def run(self, ctx):
        if ctx.get("deblocked_path") and path_exists(ctx.get("deblocked_path")):
            print("    已存在，跳过")
            return

        input_path = str(ctx.input_path)
        if not input_path or not path_exists(input_path):
            return

        cfg = ctx.config.get("pre_deblock", {})
        if not cfg.get("enabled", False):
            ctx.set("deblocked_path", input_path)
            return

        # spp 参数: quality(0-6), mode(0=hard,1=soft)
        quality = cfg.get("quality", 4)
        mode = cfg.get("mode", 1)

        out_path = ctx.output_dir / f"{Path(input_path).stem}_deblocked.mp4"
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"

        print(f"    压缩去块: spp(quality={quality}, mode={'soft' if mode else 'hard'})")

        cmd = [
            ffmpeg, "-y", "-i", input_path,
            "-vf", f"spp={quality}:{mode}:1",  # quality:mode:use_bframes
            "-c:v", "libx264", "-preset", "fast", "-crf", "1",
            "-pix_fmt", "yuv420p", "-c:a", "copy", str(out_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"    spp 失败: {result.stderr[-300:]}")
            ctx.set("deblocked_path", input_path)
            return

        ctx.set("deblocked_path", str(out_path))
        ctx.input_path = out_path  # 后续所有阶段自动用清洗后的版本
        print(f"    输出: {out_path.name}")

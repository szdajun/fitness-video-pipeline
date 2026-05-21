"""阶段30: GFPGAN 云增强

在 export 之后运行，对最终视频进行 AI 人脸修复。
支持两种模式：
  1. 本地 GPU (CUDA) — 直接调用 enhance_face.py
  2. 云 GPU (AutoDL) — 通过 SSH 自动上传/处理/下载

配置 (config.yaml):
  cloud_enhance:
    enabled: true
    strength: 0.8
    mode: cloud          # cloud | local
    # cloud 模式需以下配置:
    host: connect.westd.seetacloud.com
    port: 19088
    user: root
    password: "xxx"
    model_path: /root/gfpgan_weights/GFPGANv1.4.pth
"""

import os, sys, subprocess, shutil
from pathlib import Path


class FaceEnhanceStage:
    def run(self, ctx):
        cfg = ctx.config.get("cloud_enhance", {})
        if not cfg.get("enabled", False):
            print("    跳过: cloud_enhance 未启用")
            return

        # 检查是否已有输出
        if ctx.get("face_enhance_path") and Path(ctx.get("face_enhance_path")).exists():
            print("    已存在，跳过")
            return

        # 取输入视频（优先用 export 输出）
        input_path = ctx.get("export_path") or str(ctx.input_path)
        if not input_path or not Path(input_path).exists():
            input_path = str(ctx.input_path)

        stem = Path(input_path).stem
        output_path = str(ctx.output_dir / f"{stem}_enhanced.mp4")
        strength = cfg.get("strength", 0.8)

        print(f"    GFPGAN 人脸增强: strength={strength}")

        mode = cfg.get("mode", "cloud")

        if mode == "local":
            success = self._run_local(input_path, output_path, strength, cfg)
        else:
            success = self._run_cloud(input_path, output_path, strength, cfg)

        if success:
            ctx.set("face_enhance_path", output_path)
            print(f"    输出: {Path(output_path).name}")
        else:
            print("    增强失败，跳过")
            ctx.set("face_enhance_path", input_path)

    def _run_local(self, input_path, output_path, strength, cfg):
        """本地 GPU 模式"""
        import torch
        if not torch.cuda.is_available():
            print("    无 CUDA GPU，跳过本地模式")
            return False

        script = Path(__file__).parent.parent / "cloud_gpu" / "enhance_face.py"
        if not script.exists():
            print(f"    找不到: {script}")
            return False

        model_path = cfg.get("model_path", "")
        cmd = [sys.executable, str(script), input_path, output_path,
               "--strength", str(strength)]
        if model_path:
            cmd += ["--model_path", model_path]

        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(f"    本地增强失败: {r.stderr[-300:]}")
            return False
        return True

    def _run_cloud(self, input_path, output_path, strength, cfg):
        """云 GPU 模式 — 通过 SSH 自动处理"""
        # 尝试导入 auto_enhance
        sys.path.insert(0, str(Path(__file__).parent.parent / "cloud_gpu"))
        try:
            from auto_enhance import auto_enhance
        except ImportError as e:
            print(f"    无法加载 auto_enhance 模块: {e}")
            print("    请安装 paramiko: pip install paramiko")
            return False

        success = auto_enhance(input_path, output_path, strength=strength)
        return success

"""AutoDL 云端 GPU 视频超分脚本
720p → 4K (3840x2160) 使用 Real-ESRGAN

用法:
    python upscale_video.py input.mp4 [output.mp4] [--model_path /path/to/RealESRGAN_x4plus.pth]

依赖: pip install realesrgan opencv-python torch numpy
"""
import cv2, torch, argparse, os, sys, time, shutil, subprocess
import numpy as np
from pathlib import Path

# torchvision 兼容补丁
import torchvision.transforms.functional as _F
import types as _types
_tensor_mod = _types.ModuleType('torchvision.transforms.functional_tensor')
_tensor_mod.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _tensor_mod

# 目标分辨率
TARGET_W, TARGET_H = 3840, 2160  # 4K UHD


def _load_esrgan_model(model_path=None, device='cuda'):
    """加载 Real-ESRGAN 模型"""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if model_path and not Path(model_path).exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 创建 RRDBNet 架构 (RealESRGAN_x4plus)
    model_arch = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=4,
    )

    model = RealESRGANer(
        scale=4,
        model_path=str(model_path) if model_path else None,
        model=model_arch,  # 必须传入架构实例
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device,
    )
    print(f"Real-ESRGAN 已加载 (device={device})")
    return model


def upscale_video(input_path, output_path, model_path=None, device='cuda'):
    """720p → 4K 超分"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"输入: {input_path}")
    print(f"分辨率: {TARGET_W}x{TARGET_H} (4K), FPS: {fps:.2f}, 总帧: {total}")

    # 加载模型
    model = _load_esrgan_model(model_path, device)

    # 用 PNG 序列作为中间存储（避免 VideoWriter 4K 编码问题）
    tmpdir = Path(output_path).parent / f"_upscale_tmp_{Path(output_path).stem}"
    tmpdir.mkdir(exist_ok=True)

    # 临时 4K 视频输出（无音频）
    tmp_video = tmpdir / "upscaled.mp4"

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Real-ESRGAN 增强
        try:
            output, _ = model.enhance(frame, outscale=3.0)  # 720*3=2160p (4K)
        except Exception as e:
            print(f"  帧 {frame_idx} 跳过: {e}")
            output = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

        # 确保尺寸精确
        if output.shape[1] != TARGET_W or output.shape[0] != TARGET_H:
            output = cv2.resize(output, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4)

        # 写 PNG 帧
        cv2.imwrite(str(tmpdir / f"f_{frame_idx:06d}.png"), output)

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed
            eta = (total - frame_idx) / fps_proc if fps_proc > 0 else 0
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)  "
                  f"{fps_proc:.1f}fps, ETA {eta/60:.1f}min")

    cap.release()
    print(f"  渲染完成: {frame_idx} 帧，FFmpeg 编码中...")

    # FFmpeg 编码为 4K 视频
    ffmpeg_bin = shutil.which('ffmpeg') or 'ffmpeg'
    cmd = [ffmpeg_bin, '-y', '-v', 'warning',
           '-framerate', str(fps),
           '-i', str(tmpdir / 'f_%06d.png'),
           '-c:v', 'libx265', '-preset', 'medium', '-crf', '18',
           '-pix_fmt', 'yuv420p',
           '-tag:v', 'hvc1',  # Apple 兼容
           str(tmp_video)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FFmpeg 编码错误: {r.stderr[-300:]}")
        # fallback 到 libx264
        cmd[cmd.index('libx265')] = 'libx264'
        cmd[cmd.index('-tag:v')] = '-preset'
        cmd[cmd.index('hvc1')] = 'fast'
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg 编码失败: {r.stderr[-300:]}")

    # 复制音频
    audio_tmp = tmpdir / "with_audio.mp4"
    cmd = [ffmpeg_bin, '-y', '-i', str(input_path), '-i', str(tmp_video),
           '-c:v', 'copy', '-c:a', 'aac', '-map', '0:a?', '-map', '1:v',
           '-shortest', str(audio_tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        shutil.move(str(audio_tmp), str(output_path))
    else:
        shutil.move(str(tmp_video), str(output_path))

    # 清理
    shutil.rmtree(tmpdir, ignore_errors=True)

    total_time = time.time() - t0
    out_size = Path(output_path).stat().st_size
    print(f"完成! {frame_idx}帧, {total_time/60:.1f}min, {out_size/1024/1024:.1f}MB")
    print(f"输出: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-ESRGAN 视频 4K 超分')
    parser.add_argument('input', help='输入视频路径')
    parser.add_argument('output', nargs='?', default=None, help='输出视频路径')
    parser.add_argument('--model_path', default=None, help='RealESRGAN_x4plus.pth 路径')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    out = args.output or str(Path(args.input).with_stem(Path(args.input).stem + '_4K'))
    upscale_video(args.input, out, args.model_path, device)

"""AutoDL 云端 GPU 人脸修复脚本
用法:
    python enhance_face.py input.mp4 [output.mp4] [--strength 0.8] [--interval 5]

依赖: pip install gfpgan opencv-python torch numpy
"""
import cv2, torch, argparse, os, sys, time, shutil, subprocess
import numpy as np
from pathlib import Path

# torchvision 兼容补丁（新版 basicsr/gfpgan 依赖旧版 functional_tensor）
import torchvision.transforms.functional as _F
import types as _types
_tensor_mod = _types.ModuleType('torchvision.transforms.functional_tensor')
_tensor_mod.rgb_to_grayscale = _F.rgb_to_grayscale
sys.modules['torchvision.transforms.functional_tensor'] = _tensor_mod

# ============================================================
# OpenCV 人脸检测（内置 Haar Cascade，无需额外下载）
# ============================================================
def _detect_faces_opencv(frame, cascade):
    """用 OpenCV Haar Cascade 检测人脸，返回 [(x1,y1,x2,y2,conf)]"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测人脸：适当参数适应视频场景
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5,
        minSize=(60, 60),  # 最小人脸 60x60
    )
    results = []
    for (x, y, w, h) in faces:
        conf = min(1.0, w / 120.0)  # 越大框越可信
        results.append((x, y, x + w, y + h, conf))
    # 按面积从大到小排序
    results.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
    return results


# ============================================================
# 加载 GFPGAN 模型（手动加载，绕过 facexlib 检测）
# ============================================================
def _load_gfpgan_model(model_path=None, device='cuda'):
    """手动加载 GFPGANv1.4，不依赖 facexlib

    Args:
        model_path: 模型文件路径。为 None 时自动搜索常见位置。
    """
    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        print(f"  使用指定模型: {model_path}")
    else:
        # 搜索常见位置
        candidates = [
            Path(__file__).parent / 'gfpgan_weights' / 'GFPGANv1.4.pth',
            Path.home() / 'gfpgan_weights' / 'GFPGANv1.4.pth',
            Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'gfpgan' / 'GFPGANv1.4.pth',
        ]
        model_path = None
        for p in candidates:
            if p.exists():
                model_path = p
                print(f"  使用本地权重: {model_path}")
                break
        if model_path is None:
            raise FileNotFoundError(
                f"找不到模型文件，请在以下位置放置:\n" +
                "\n".join(f"  {p}" for p in candidates) +
                "\n或通过 --model_path 参数指定路径"
            )

    # 构造模型
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    model = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )
    state_dict = torch.load(str(model_path), map_location='cpu')
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)
    print(f"  GFPGAN 模型已加载 (device={device})")
    return model


def _preprocess_face(face_roi, device='cuda'):
    """准备人脸 ROI 给 GFPGAN 推理"""
    # 缩放到 512x512
    face_512 = cv2.resize(face_roi, (512, 512), interpolation=cv2.INTER_CUBIC)
    # RGB, float32, normalize to [-1, 1]
    face_tensor = torch.from_numpy(face_512.astype(np.float32) / 127.5 - 1.0)
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return face_tensor


def _postprocess_face(output_tensor, orig_size):
    """GFPGAN 输出 → numpy image"""
    output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = np.clip((output + 1.0) * 127.5, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    if output.shape[:2] != orig_size:
        output = cv2.resize(output, orig_size, interpolation=cv2.INTER_AREA)
    return output


def enhance_video(input_path, output_path, strength=0.8, interval=5, device='cuda', model_path=None):
    """逐帧检测人脸并用 GFPGAN 修复，仅处理脸部区域提高速度"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"输入: {w}x{h} @ {fps}fps, {total}帧")

    # 初始化 OpenCV 人脸检测器
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if cascade.empty():
        raise RuntimeError("OpenCV Haar Cascade 加载失败")
    print("OpenCV 人脸检测器已加载")

    # 加载 GFPGAN 模型
    model = _load_gfpgan_model(model_path, device)

    # 输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    cached_bbox = None
    frame_idx = 0
    t0 = time.time()

    # 预热 GPU（跑一次 dummy）
    dummy = torch.zeros((1, 3, 512, 512), device=device)
    with torch.no_grad():
        _ = model(dummy)
    print("GPU 预热完成")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测间隔：每 interval 帧检测一次人脸
        detect = (frame_idx % interval == 0) or (cached_bbox is None)

        try:
            if detect:
                bboxes = _detect_faces_opencv(frame, cascade)
                if bboxes:
                    cached_bbox = bboxes[0]
                else:
                    cached_bbox = None

            if cached_bbox is not None:
                x1, y1, x2, y2, _ = [int(v) for v in cached_bbox[:5]]
                # 扩大人脸区域 1.5 倍，确保包含完整脸部
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                face_w = x2 - x1
                face_h = y2 - y1
                expand = 1.5
                half_w = int(face_w * expand / 2)
                half_h = int(face_h * expand / 2)
                x1 = max(0, cx - half_w)
                y1 = max(0, cy - half_h)
                x2 = min(w, cx + half_w)
                y2 = min(h, cy + half_h)

                if x2 > x1 + 20 and y2 > y1 + 20:
                    face_roi = frame[y1:y2, x1:x2]
                    orig_h, orig_w = face_roi.shape[:2]

                    # 预处理 → GFPGAN 推理 → 后处理
                    face_tensor = _preprocess_face(face_roi, device)
                    with torch.no_grad():
                        output = model(face_tensor)
                        if isinstance(output, (tuple, list)):
                            output = output[0]

                    enhanced_face = _postprocess_face(output, (orig_w, orig_h))

                    # 强度混合（strength 控制 GFPGAN 影响程度）
                    blended = cv2.addWeighted(
                        face_roi, 1.0 - strength,
                        enhanced_face, strength, 0
                    )

                    # 高斯边缘混合
                    mask = np.zeros((orig_h, orig_w), dtype=np.float32)
                    inner_margin = max(5, min(orig_h, orig_w) // 12)
                    cv2.rectangle(
                        mask,
                        (inner_margin, inner_margin),
                        (orig_w - inner_margin, orig_h - inner_margin),
                        1.0, -1
                    )
                    blur_ksize = max(3, min(orig_h, orig_w) // 6)
                    if blur_ksize % 2 == 0:
                        blur_ksize += 1
                    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
                    mask_3ch = np.stack([mask] * 3, axis=-1)

                    frame[y1:y2, x1:x2] = (
                        face_roi.astype(np.float32) * (1 - mask_3ch) +
                        blended.astype(np.float32) * mask_3ch
                    ).astype(np.uint8)

        except Exception as e:
            print(f"  帧 {frame_idx} 跳过: {e}")
            pass

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            fps_proc = frame_idx / elapsed
            eta = (total - frame_idx) / fps_proc if fps_proc > 0 else 0
            print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%)  "
                  f"{fps_proc:.1f}fps, ETA {eta/60:.1f}min")

    cap.release()
    writer.release()
    total_time = time.time() - t0
    print(f"处理完成! {frame_idx}帧, 耗时 {total_time/60:.1f}min")

    # 从源视频复制音频到输出
    import subprocess
    audio_tmp = str(Path(output_path).with_suffix('.audio_tmp.mp4'))
    ffmpeg_bin = shutil.which('ffmpeg') or 'ffmpeg'
    cmd = [ffmpeg_bin, '-y', '-i', str(input_path), '-i', str(output_path),
           '-c:v', 'copy', '-c:a', 'aac', '-map', '0:a?', '-map', '1:v',
           '-shortest', str(audio_tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        shutil.move(audio_tmp, str(output_path))
        print(f"音频已复制到输出文件")
    else:
        # 如果源无音频轨道，忽略
        if 'Stream map' in r.stderr and 'No audio' in r.stderr:
            print("源视频无音频轨道")
        else:
            print(f"音频复制失败 (源可能无音频): {r.stderr[-200:]}")
        Path(audio_tmp).unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GFPGAN 视频人脸修复（OpenCV 检测版）')
    parser.add_argument('input', help='输入视频路径')
    parser.add_argument('output', nargs='?', default=None, help='输出视频路径')
    parser.add_argument('--strength', type=float, default=0.8, help='修复强度 0-1')
    parser.add_argument('--interval', type=int, default=5, help='人脸检测帧间隔')
    parser.add_argument('--model_path', default=None, help='GFPGAN 模型文件路径（.pth）')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print("警告: 使用 CPU (极慢)，建议 GPU 实例")

    out = args.output or str(Path(args.input).with_stem(Path(args.input).stem + '_enhanced'))
    enhance_video(args.input, out, args.strength, args.interval, device, args.model_path)

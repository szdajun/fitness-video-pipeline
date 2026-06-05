#!/usr/bin/env python3
"""换脸工具：将目标视频的人脸替换为源教练的人脸

依赖: insightface + inswapper_128.onnx

用法:
  python tools/face_swap.py --source 教练照片.jpg --target 目标视频.mp4 --output 输出.mp4

流程:
  1. 从 source 提取人脸特征(embedding)
  2. 逐帧检测 target 视频中的人脸
  3. 用 inswapper 将 source 人脸换到 target 上
  4. 编码输出视频（含原音频）
"""

import cv2, numpy as np, argparse, os, subprocess, sys, tempfile, shutil
from pathlib import Path

SWAPPER_MODEL = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"


def get_swapper():
    import insightface
    if not os.path.exists(SWAPPER_MODEL):
        raise FileNotFoundError(f"模型未找到: {SWAPPER_MODEL}\n请先下载: python _download_inswapper.py")
    # 12GB 显存，全 GPU 模式
    return insightface.model_zoo.get_model(SWAPPER_MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def get_face_analyser():
    import insightface
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def extract_face_embedding(app, image_path):
    """从源图片提取人脸特征"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    faces = app.get(img)
    if not faces:
        raise ValueError(f"未检测到人脸: {image_path}")
    # 取面积最大的人脸
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    print(f"  源人脸: bbox={best.bbox.astype(int)}, det_score={best.det_score:.2f}")
    return best


def swap_face(swapper, source_face, target_img, app):
    """只换最大的人脸（主教练通常离镜头最近）"""
    faces = app.get(target_img)
    if not faces:
        return target_img
    # 只取面积最大的脸
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    if best.det_score < 0.3:
        return target_img
    try:
        return swapper.get(target_img, best, source_face, paste_back=True)
    except Exception:
        return target_img


def process_video(source_path, target_path, output_path, max_frames=0, every_n=1):
    """逐帧处理视频换脸"""
    print(f"加载模型...")
    app = get_face_analyser()
    swapper = get_swapper()
    source_face = extract_face_embedding(app, source_path)

    cap = cv2.VideoCapture(target_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(3))
    h = int(cap.get(4))

    if max_frames > 0:
        total = min(total, max_frames)

    print(f"处理 {total} 帧 @ {fps:.1f}fps, 每 {every_n} 帧检测一次...")

    # 管道输出到F盘临时文件，不写PNG序列
    tmp_vid = os.path.join(os.path.dirname(output_path), "_tmp_vid.mp4")
    ffmpeg_cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an",
        tmp_vid
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)

    fi = 0
    out_fi = 0
    swap_count = 0
    face_count = 0
    while fi < total:
        ret, frame = cap.read()
        if not ret:
            break

        if fi % every_n == 0:
            faces_before = len(app.get(frame))
            frame = swap_face(swapper, source_face, frame, app)
            faces_after = len(app.get(frame))
            if faces_before > 0:
                face_count += 1
                swap_count += 1

        proc.stdin.write(frame.tobytes())
        out_fi += 1
        fi += 1

        if fi % 100 == 0:
            print(f"  进度: {fi}/{total} ({fi*100//total}%) 人脸:{face_count}帧")

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"  换脸完成: {out_fi} 帧, 混入音频...")

    # 混入原音频（如果 target 无音频流则跳过，直接复制无声视频）
    r = subprocess.run([
        FFMPEG, "-y", "-i", tmp_vid, "-i", target_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",  # ?: 1:a:0 不存在时不报错
        "-shortest",
        output_path
    ], check=False, capture_output=True, timeout=120)
    if r.returncode != 0 or not os.path.exists(output_path):
        # 混音失败（target 无音频流/超时），直接复制无声视频
        print(f"    混音失败 (可能无音频流), 直接复制无声视频")
        r2 = subprocess.run([
            FFMPEG, "-y", "-i", tmp_vid,
            "-c:v", "copy", "-an",
            output_path
        ], check=False, capture_output=True, timeout=60)
        if r2.returncode != 0:
            print(f"    复制视频也失败: {r2.stderr[-200:]}")

    # 只在 output 成功生成后才删 tmp_vid（之前无论成功与否都删导致数据丢失）
    if os.path.exists(output_path):
        os.remove(tmp_vid)
        print(f"  输出: {output_path}")
    else:
        # 保留 tmp_vid 供上层 stage fallback 使用
        print(f"  换脸视频未生成, 保留临时文件: {tmp_vid}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="教练换脸工具")
    parser.add_argument("--source", required=True, help="教练脸部照片路径")
    parser.add_argument("--target", required=True, help="目标视频路径")
    parser.add_argument("--output", required=True, help="输出视频路径")
    parser.add_argument("--max-frames", type=int, default=300, help="最大处理帧数(默认300)")
    parser.add_argument("--every-n", type=int, default=1, help="每N帧检测一次人脸(1=每帧)")
    args = parser.parse_args()

    process_video(args.source, args.target, args.output,
                  max_frames=args.max_frames, every_n=args.every_n)

"""夜景健身视频色彩增强 - 独立脚本

用法:
    python color_grade_video.py "视频路径" --output "输出路径"
    python color_grade_video.py "视频路径" --brightness 15 --contrast 1.15 --warmth -5
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def analyze_brightness(video_path, sample_frames=30):
    """分析视频平均亮度，返回 0~100 的亮度值（越高越亮）"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // sample_frames)

    brightness_sum = 0
    count = 0

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        # 转换到 LAB 亮度通道
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        brightness = lab[:, :, 0].mean()
        brightness_sum += brightness
        count += 1

    cap.release()

    if count == 0:
        return None

    # LAB L 通道范围 0~255，归一化到 0~100
    avg = brightness_sum / count
    return avg / 255.0 * 100


def color_grade_frame(frame, brightness=0, contrast=1.0, saturation=1.0, warmth=0,
                       sharpen=0, temporal_smooth=0, prev_frame=None, clahe=None,
                       gamma=1.0, denoise_h=0):
    """对单帧应用色彩调整"""
    result = frame

    # 1. 亮度
    if brightness != 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness * 2.55, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 2. Gamma校正 — 暗部提亮更自然
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        result = cv2.LUT(result, table)

    # 3. 对比度
    if contrast != 1.0:
        result = cv2.convertScaleAbs(result, alpha=contrast, beta=0)

    # 4. 饱和度
    if saturation != 1.0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 5. 色温 (warm > 0 暖, < 0 冷)
    if warmth != 0:
        temp = result.astype(np.float32)
        temp[:, :, 2] = np.clip(temp[:, :, 2] + warmth * 0.5, 0, 255)   # R+
        temp[:, :, 0] = np.clip(temp[:, :, 0] - warmth * 0.3, 0, 255)   # B-
        result = temp.astype(np.uint8)

    # 6. CLAHE 自适应直方图均衡
    if clahe is not None:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 7. 锐化
    if sharpen > 0:
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.0 + sharpen, blurred, -sharpen, 0)

    # 7b. 降噪（fastNlMeans — 边缘保护降噪）
    if denoise_h > 0:
        result = cv2.fastNlMeansDenoisingColored(result, None, denoise_h, denoise_h, 7, 21)

    # 8. 时间平滑
    if temporal_smooth > 0 and prev_frame is not None:
        result = cv2.addWeighted(result, 1.0 - temporal_smooth,
                                 prev_frame, temporal_smooth, 0)

    return result


def auto_adjust_params(brightness_score):
    """根据亮度分析结果自动计算参数

    brightness_score: 0(黑)~100(白)
    返回 (brightness, contrast)
    """
    if brightness_score < 15:
        # 非常暗：大幅提亮
        brightness = 18
        contrast = 1.20
    elif brightness_score < 25:
        # 较暗：中等提亮
        brightness = 12
        contrast = 1.15
    elif brightness_score < 35:
        # 略暗：轻微提亮
        brightness = 8
        contrast = 1.10
    elif brightness_score < 50:
        # 正常偏暗：微调
        brightness = 5
        contrast = 1.05
    elif brightness_score < 65:
        # 正常：轻度增强
        brightness = 3
        contrast = 1.05
    else:
        # 较亮：保守处理
        brightness = 0
        contrast = 1.0

    return brightness, contrast


def main():
    parser = argparse.ArgumentParser(description="夜景健身视频色彩增强")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", help="输出视频路径（默认加 _color 后缀）")
    parser.add_argument("--brightness", type=int, default=None,
                        help="亮度 -100~100 (默认: 自动)")
    parser.add_argument("--contrast", type=float, default=None,
                        help="对比度 0.5~2.0 (默认: 自动)")
    parser.add_argument("--saturation", type=float, default=1.10,
                        help="饱和度 0.5~2.0 (默认: 1.10)")
    parser.add_argument("--warmth", type=int, default=-5,
                        help="色温 -50~50 (默认: -5，抵消黄灯暖色）")
    parser.add_argument("--clahe", action="store_true", default=True,
                        help="启用CLAHE自适应直方图均衡（默认开启）")
    parser.add_argument("--no-clahe", action="store_true",
                        help="禁用CLAHE")
    parser.add_argument("--sharpen", type=float, default=0.06,
                        help="锐化强度 0~1 (默认: 0.06)")
    parser.add_argument("--temporal-smooth", type=float, default=0.2,
                        help="时序平滑 0~1 (默认: 0.2)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma校正 0.5~2.0 (默认: 1.0，>1暗部提亮更自然)")
    parser.add_argument("--denoise", type=float, default=0,
                        help="降噪强度 0~20 (默认: 0，建议夜间5-10)")
    parser.add_argument("--auto", action="store_true",
                        help="自动分析亮度并调整参数")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在 {input_path}")
        return

    # 分析亮度
    print(f"分析视频亮度...")
    brightness_score = analyze_brightness(input_path)
    if brightness_score is None:
        print(f"错误: 无法读取视频 {input_path}")
        return

    level = "极暗" if brightness_score < 15 else \
            "较暗" if brightness_score < 25 else \
            "略暗" if brightness_score < 35 else \
            "正常" if brightness_score < 55 else \
            "较亮"
    print(f"检测到亮度: {brightness_score:.1f}/100 ({level})")

    # 自动或手动参数
    if args.auto or args.brightness is None or args.contrast is None:
        auto_b, auto_c = auto_adjust_params(brightness_score)
        brightness = args.brightness if args.brightness is not None else auto_b
        contrast = args.contrast if args.contrast is not None else auto_c
        mode = "自动" if args.auto else "半自动"
        print(f"参数({mode}): 亮度={brightness}, 对比度={contrast}")
    else:
        brightness = args.brightness
        contrast = args.contrast

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_color.mp4"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"输入: {input_path.name} ({width}x{height}, {fps:.1f}fps, {total_frames}帧)")
    print(f"亮度: {brightness}, 对比度: {contrast:.2f}, "
          f"饱和度: {args.saturation}, 色温: {args.warmth}, gamma: {args.gamma}, 降噪: {args.denoise}")
    print(f"CLAHE: {not args.no_clahe}, 锐化: {args.sharpen}, 时序平滑: {args.temporal_smooth}")
    print(f"输出: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    clahe = None
    if not args.no_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    prev_frame = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        graded = color_grade_frame(
            frame,
            brightness=brightness,
            contrast=contrast,
            saturation=args.saturation,
            warmth=args.warmth,
            sharpen=args.sharpen,
            temporal_smooth=args.temporal_smooth,
            prev_frame=prev_frame,
            clahe=clahe,
            gamma=args.gamma,
            denoise_h=args.denoise
        )

        writer.write(graded)
        prev_frame = graded.copy()
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"    进度: {pct:.0f}% ({frame_idx}/{total_frames})")

    cap.release()
    writer.release()

    output_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n完成! 输出: {output_path} ({output_size:.1f}MB)")


if __name__ == "__main__":
    main()

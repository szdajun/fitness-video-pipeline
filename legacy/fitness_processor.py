"""健身视频每日处理流水线（混合方案）

流程：
  Step1: FFmpeg 滤镜链（亮度/对比度/饱和度/白平衡/gamma/锐化/格式转换）→ 40fps
  Step2: OpenCV 肤色区域识别 + 中心加权保护（仅皮肤区域降噪+保留原始肤色）→ 30fps
  Step3: FFmpeg H.265 编码 + 合并原始音轨

用法：
    python fitness_processor.py                    # 监控模式
    python fitness_processor.py --once            # 单次处理
    python fitness_processor.py --input "路径"     # 指定素材目录
"""

import argparse, os, sys, time, subprocess, yaml, shutil, uuid
import cv2, numpy as np
from pathlib import Path
from datetime import datetime

# Windows 中文编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "fitness_processor_config.yaml"
PROCESSED_LOG = SCRIPT_DIR / "output" / ".processed_log.txt"


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


config = load_config()
DEFAULT_INPUT_DIR = Path(config.get("input_dir", "C:/Users/18091/Desktop/短视频素材"))
DEFAULT_OUTPUT_DIR = Path(config.get("output_dir", "F:/wkspace/fitness-video-pipeline/output"))
FFMPEG_PATH = config.get("ffmpeg_path", "C:/Users/18091/ffmpeg/ffmpeg.exe")
WATCH_INTERVAL = config.get("watch_interval", 5)
CRF = config.get("crf", 23)
AUDIO_BITRATE = config.get("audio_bitrate", "128k")
AUTO_BRIGHTNESS = config.get("auto_brightness", True)


def get_video_date(video_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=tags", "-of", "json", str(video_path)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, errors="replace")
        if r.returncode == 0:
            import json
            data = json.loads(r.stdout)
            tags = data.get("streams", [{}])[0].get("tags", {})
            for key in ["creation_time", "date", "datetime"]:
                if key in tags:
                    dt = tags[key].replace("T", "_").replace("Z", "").split(".")[0]
                    return dt.replace(":", "").replace("-", "")
    except Exception:
        pass
    try:
        mtime = os.path.getmtime(str(video_path))
        return datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
    except Exception:
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def analyze_brightness(video_path, sample_frames=30):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // sample_frames)
    brightness_sum = count = 0
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        brightness_sum += cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0].mean()
        count += 1
    cap.release()
    return (brightness_sum / count) / 255.0 * 100 if count else None


def auto_adjust_params(score):
    if score < 15: return 18, 1.20
    elif score < 25: return 12, 1.15
    elif score < 35: return 8, 1.10
    elif score < 50: return 5, 1.05
    elif score < 65: return 3, 1.05
    else: return 0, 1.0


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_skin(frame):
    """返回皮肤区域二值 mask（255=皮肤）"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv.astype(np.float32))
    skin = ((h >= 0) & (h <= 18) & (s >= 38) & (s <= 191) & (v >= 51)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    return skin


def make_center_weight(h, w, sigma_factor=0.35):
    """生成以画面中心为峰值的高斯权重图"""
    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    sigma = max(h, w) * sigma_factor
    weight = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return (weight * 255).astype(np.uint8)


# ============ 核心处理函数 ============

def process_video(video_path, input_dir, output_base):
    """混合处理：FFmpeg快速基础处理 + OpenCV精细肤色保护"""
    print(f"\n{'='*50}")
    print(f"处理: {video_path.name}")

    video_date = get_video_date(video_path)
    print(f"日期: {video_date}")

    brightness_score = analyze_brightness(video_path)
    if brightness_score is not None:
        level = "极暗" if brightness_score < 15 else \
                "较暗" if brightness_score < 25 else \
                "略暗" if brightness_score < 35 else \
                "正常" if brightness_score < 55 else "较亮"
        print(f"亮度: {brightness_score:.1f}/100 ({level})")
    else:
        brightness_score = 30
        print(f"亮度: 无法检测，使用默认值")

    if AUTO_BRIGHTNESS:
        brightness, contrast = auto_adjust_params(brightness_score)
    else:
        brightness = config.get("manual_brightness", 5)
        contrast = config.get("manual_contrast", 1.05)
    print(f"参数: brightness={brightness}, contrast={contrast:.2f}")

    date_str = f"{video_date[:4]}-{video_date[4:6]}-{video_date[6:8]}"
    date_dir = ensure_dir(output_base / date_str)
    stem = video_path.stem

    gamma_val = config.get("gamma", 1.1)
    g = gamma_val
    # Gamma曲线: x=输入 level, y=输出 level (都是0~1)
    # y = x^(1/g) 实现 gamma 校正
    def gc(x): return x ** (1.0 / g)
    gamma_curve = (
        f"0/0 "
        f"0.05/{gc(0.05):.4f} "
        f"0.10/{gc(0.10):.4f} "
        f"0.20/{gc(0.20):.4f} "
        f"0.35/{gc(0.35):.4f} "
        f"0.50/{gc(0.50):.4f} "
        f"0.65/{gc(0.65):.4f} "
        f"0.80/{gc(0.80):.4f} "
        f"1/1"
    )

    uid = uuid.uuid4().hex[:8]
    tmp_dir = Path(f"{output_base}/tmp_{uid}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ===== Step 1: FFmpeg 滤镜链（基础色彩校正 + 防抖）=====
    # deshake: 单遍防抖，自动检测并补偿相机抖动，实时生效
    # 亮度/对比度/饱和度/gamma/白平衡/锐化全部由FFmpeg并行处理，40fps级速度
    warmth = config.get("warmth", -5)
    warmth_r_adjust = warmth * 0.5  # R通道补偿
    warmth_b_adjust = -warmth * 0.3  # B通道补偿

    # 滤镜链：deshake → hqdn3d → eq → curves → colorbalance → unsharp → vibrance → vignette → format
    # deshake: 单遍防抖（rx/ry=搜索范围px, edge=边缘处理）
    # vibrance: 智能饱和增强，保护肤色不过饱和
    # vignette: 镜头边缘亮度补偿，angle=PI/3.5 轻度修正
    deshake_str = "deshake=rx=16:ry=16:edge=2,"
    hqdn3d_str = "hqdn3d=2:1.5,"
    vibrance_str = f"vibrance={config.get('vibrance', 0.8):.2f},"
    vignette_str = f"vignette=PI/3.5,"

    ffmpeg_filter = (
        deshake_str + hqdn3d_str +
        f"eq=brightness={brightness*0.01:.4f}:contrast={contrast:.4f}"
        f":saturation={config.get('saturation', 1.10):.4f}"
        f",curves=all='{gamma_curve}'"
        f",colorbalance=rs={warmth_r_adjust*0.01:.4f}:bs={warmth_b_adjust*0.01:.4f}"
        f",unsharp=5:5:1.0:3:3:0.35"
        f",{vibrance_str}{vignette_str}"
        f"format=yuv420p"
    )

    step1_path = tmp_dir / "step1.mp4"
    print(f"  [Step1] FFmpeg滤镜处理中...")
    r = subprocess.run([
        FFMPEG_PATH, "-y", "-i", str(video_path),
        "-vf", ffmpeg_filter,
        "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-an", str(step1_path)
    ], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        print(f"  Step1失败: {r.stderr[-300:]}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    print(f"  Step1完成 ({step1_path.stat().st_size/1024/1024:.1f}MB)")

    # ===== Step 2: OpenCV 肤色区域精细保护 =====
    skin_protect = config.get("skin_protect", True)
    blend_strength = config.get("skin_blend_strength", 0.2)
    denoise_h = config.get("denoise_h", 0)
    smooth_val = config.get("temporal_smooth", 0.2)
    sharpen_val = config.get("sharpen", 0.06)

    cap2 = cv2.VideoCapture(str(step1_path))
    fps = cap2.get(cv2.CAP_PROP_FPS)
    width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [Step2] OpenCV肤色保护: {width}x{height}, {fps:.1f}fps, {total_frames}帧")

    step2_path = tmp_dir / "step2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer2 = cv2.VideoWriter(str(step2_path), fourcc, fps, (width, height))

    # 预计算中心权重图（只需计算一次）
    center_weight_map = make_center_weight(height, width)

    prev_frame = None
    frame_idx = 0

    # 同步读取原始视频（肤色对比用）
    cap_orig = cv2.VideoCapture(str(video_path))
    orig_frame = None

    while True:
        ret2, frame2 = cap2.read()
        if not ret2:
            break
        ret_o, orig_frame = cap_orig.read()
        if not ret_o:
            orig_frame = frame2

        result = frame2.copy()

        # 肤色保护（先做肤色混合，再整体降噪）
        if skin_protect:
            skin_mask = detect_skin(result)
            if skin_mask is not None:
                skin_mask_cw = cv2.bitwise_and(skin_mask, skin_mask, mask=center_weight_map)
                result = np.where(skin_mask_cw[:, :, None] > 0,
                                  (result * (1 - blend_strength) + orig_frame * blend_strength).astype(np.uint8),
                                  result)

        # 降噪（fastNlMeans，边缘保护，对肤色混合后的整帧降噪）
        if denoise_h > 0:
            result = cv2.fastNlMeansDenoisingColored(result, None, denoise_h, denoise_h, 7, 21)

        # 锐化
        if sharpen_val > 0:
            blurred = cv2.GaussianBlur(result, (0, 0), 3)
            result = cv2.addWeighted(result, 1.0 + sharpen_val, blurred, -sharpen_val, 0)

        # 时序平滑
        if prev_frame is not None and smooth_val > 0:
            result = cv2.addWeighted(result, 1.0 - smooth_val, prev_frame, smooth_val, 0)
        prev_frame = result.copy()

        writer2.write(result)
        frame_idx += 1
        if frame_idx % 200 == 0:
            pct = frame_idx / total_frames * 100
            elapsed = frame_idx / fps
            eta = (total_frames - frame_idx) / fps
            print(f"    {pct:.0f}% ({frame_idx}/{total_frames}) | {elapsed:.0f}s ETA {eta:.0f}s")

    cap2.release()
    writer2.release()
    cap_orig.release()
    print(f"  Step2完成 ({step2_path.stat().st_size/1024/1024:.1f}MB)")

    # ===== Step3: H.265编码 + 合并音轨 =====
    print(f"  [Step3] H.265编码 + 合并音轨...")
    final_path = date_dir / f"{stem}_{video_date}_final.mp4"
    encoder = config.get("encoder", "libx265")
    r = subprocess.run([
        FFMPEG_PATH, "-y",
        "-i", str(step2_path),
        "-i", str(video_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", encoder, "-crf", str(CRF), "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-shortest",
        str(final_path)
    ], capture_output=True, text=True, errors="replace")

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if r.returncode != 0:
        print(f"  Step3失败: {r.stderr[-200:]}")
        return None

    print(f"  最终: {final_path.name} ({final_path.stat().st_size/1024/1024:.1f}MB)")

    # 备份原始
    orig_backup = date_dir / f"{stem}_{video_date}_original.mp4"
    shutil.copy2(str(video_path), str(orig_backup))
    print(f"  原始备份: {orig_backup.name}")

    # 删除素材目录原文件
    video_path.unlink()
    print(f"  已删除原文件: {video_path}")

    # 记录
    with open(PROCESSED_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {video_path.name} | {final_path}\n")

    return final_path


# ============ 主程序 ============

def get_processed_names():
    if not PROCESSED_LOG.exists():
        return set()
    with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
        return {line.split("|")[1].strip() for line in f if "|" in line}


def scan_and_process(input_dir, output_base, once=False):
    print(f"素材目录: {input_dir}")
    print(f"输出目录: {output_base}")
    print(f"模式: {'单次' if once else '监控'}")
    sep = "=" * 50
    print(sep)

    while True:
        processed = get_processed_names()
        video_files = sorted(input_dir.glob("*.mp4"))
        new_files = [f for f in video_files if f.name not in processed]

        if new_files:
            print(f"\n发现 {len(new_files)} 个新视频")
            for vf in new_files:
                try:
                    result = process_video(vf, input_dir, output_base)
                    if result:
                        print(f"  [OK] 完成: {result.name}")
                except Exception as e:
                    print(f"  [FAIL] {vf.name} - {e}")
        elif once:
            print("没有新视频需要处理")
            break
        else:
            print(f"\r{datetime.now().strftime('%H:%M:%S')} 等待新视频...", end="", flush=True)

        if once:
            break
        time.sleep(WATCH_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="健身视频处理流水线")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误: 素材目录不存在 {args.input}")
        return

    ensure_dir(args.output)
    scan_and_process(args.input, args.output, once=args.once)


if __name__ == "__main__":
    main()
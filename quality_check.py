"""视频质量预检 — 扫描文件夹，评分排序，推荐最佳1-2条"""
import cv2, numpy as np, sys, os, glob, argparse
from pathlib import Path

MIN_DURATION = 30    # 最短秒数
MAX_DURATION = 600   # 最长秒数
MIN_RES = (720, 480)  # 最低分辨率
MAX_SHAKINESS = 15   # 最大抖动指数


def _shakiness(video_path):
    """粗略抖动检测：连续帧间光流方差"""
    cap = cv2.VideoCapture(str(video_path))
    prev = None
    frame_diffs = []
    count = 0
    while count < 300:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if count % 5 == 0 and prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            frame_diffs.append(float(np.mean(mag)))
        prev = gray
        count += 1
    cap.release()
    if not frame_diffs:
        return 0
    return np.std(frame_diffs) / (np.mean(frame_diffs) + 1e-6)


def _brightness(video_path):
    """采样帧平均亮度"""
    cap = cv2.VideoCapture(str(video_path))
    means = []
    count = 0
    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            means.append(float(np.mean(gray)))
        count += 1
    cap.release()
    return np.mean(means) if means else 0


def _person_quality(video_path):
    """粗略人物检测：用轻量姿态检测看人物清晰度"""
    cap = cv2.VideoCapture(str(video_path))
    scores = []
    # Use built-in HOG detector as lightweight quality proxy
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    count = 0
    while count < 60:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), scale=1.05)
            if len(rects) > 0:
                scores.append(float(np.max(weights)))
        count += 1
    cap.release()
    return np.mean(scores) if scores else 0


def score_video(video_path):
    """综合质量评分 0-100"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frames / fps if fps > 0 else 0

    # --- 硬性过滤 ---
    issues = []
    if duration < MIN_DURATION:
        issues.append(f"太短({duration:.0f}s<{MIN_DURATION}s)")
    if duration > MAX_DURATION:
        issues.append(f"太长({duration:.0f}s>{MAX_DURATION}s)")
    if w < MIN_RES[0] or h < MIN_RES[1]:
        issues.append(f"分辨率低({w}x{h})")
    if issues:
        return {"path": video_path, "score": -1, "issues": issues,
                "duration": duration, "resolution": f"{w}x{h}", "fps": fps}

    # --- 质量评分 ---
    brightness = _brightness(video_path)
    shake = _shakiness(video_path)
    person_q = _person_quality(video_path)

    # 亮度: 太暗(<60)或过曝(>200)扣分
    brightness_score = 100 - abs(brightness - 120) if 60 <= brightness <= 200 else max(0, 100 - abs(brightness - 120) * 2)
    # 抖动: 越低越好
    shake_score = max(0, 100 - shake * 10) if shake < MAX_SHAKINESS else 0
    # 人物质量
    person_score = min(100, person_q * 50)

    total = brightness_score * 0.2 + shake_score * 0.3 + person_score * 0.3 + 20  # base 20 for valid video
    total += min(20, duration / 10)  # longer videos get bonus (up to 20)

    return {
        "path": video_path,
        "score": round(total, 1),
        "issues": [],
        "duration": duration,
        "resolution": f"{w}x{h}",
        "fps": fps,
        "brightness": round(brightness, 1),
        "shake": round(shake, 2),
        "person_q": round(person_q, 2),
    }


def scan(source_dir, limit=3):
    videos = []
    for ext in ["*.mp4", "*.MP4", "*.mov", "*.MOV"]:
        videos.extend(glob.glob(os.path.join(source_dir, ext)))
    videos = sorted(set(videos))
    if not videos:
        print("无视频文件")
        return []

    print(f"预检 {len(videos)} 个视频...\n")

    results = []
    for v in videos:
        r = score_video(v)
        symbol = "X" if r["score"] < 0 else f"{r['score']:.0f}"
        name = os.path.basename(v)
        status = " | ".join(r["issues"]) if r["issues"] else f"亮度={r.get('brightness','?')} 抖动={r.get('shake','?')}"
        print(f"  [{symbol}] {name[:40]} | {r['duration']:.0f}s {r['resolution']} | {status}")
        results.append(r)

    # Sort by score descending, filter out rejected
    valid = [r for r in results if r["score"] > 0]
    valid.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n推荐发布 (Top {min(limit, len(valid))}):")
    for i, r in enumerate(valid[:limit]):
        print(f"  {i+1}. [{r['score']:.0f}分] {os.path.basename(r['path'])} "
              f"({r['duration']:.0f}s, {r['resolution']}, 亮度={r['brightness']}, 抖动={r['shake']})")

    return valid[:limit]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dir", nargs="?", default=r"C:\Users\18091\Desktop\短视频素材")
    p.add_argument("--limit", type=int, default=2, help="推荐数量")
    args = p.parse_args()
    scan(args.dir, args.limit)

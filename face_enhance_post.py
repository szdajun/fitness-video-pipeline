"""GFPGAN 跳帧增强 v2 — 带人脸追踪，不漏帧
用法: python face_enhance_post.py [间隔帧数]
"""
import cv2, numpy as np, time, sys, os, subprocess

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
import torch
torch.set_num_threads(6)

from gfpgan import GFPGANer
import mediapipe as mp

# 参数
INPUT = "F:/wkspace/fitness-video-pipeline/output/2026-04-27/艳青4_final_16x9.mp4"
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUTPUT = INPUT.replace(".mp4", "_facehd.mp4")
print(f"间隔: {INTERVAL}帧 | 线程: 6 | 输出: {OUTPUT}")

# GFPGAN
restorer = GFPGANer(
    model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
)

# MediaPipe（阈值 0.6 过滤广告板误检 + 滞后回退）
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.6)

# 打开视频
cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(3))
h = int(cap.get(4))
print(f"视频: {w}x{h}, {fps:.1f}fps, {total} 帧")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT, fourcc, fps, (w, h))

# ---------- 检测区域限定（只查画面中央，广告板在边缘）----------
_CX_MIN = 0.15  # 水平检测范围 15%-85%
_CX_MAX = 0.85
_CY_MIN = 0.15  # 垂直检测范围 15%-85%
_CY_MAX = 0.85

# ---------- 人脸检测 + 位置追踪防误检 ----------
_prev_face = None
_FACE_HYSTERESIS = 5
_no_face_count = 0
_face_positions = []      # 历史位置，用于追踪一致性
_MAX_POS_JUMP = 0.25      # 相对帧尺寸的最大跳跃比例

# ---------- 增强缓存 ----------
cached_tex = None
cached_face = None
cached_kf_idx = -1
_REACQUIRE_MIN = 30    # 丢脸后需连续 30 帧稳定检测才重建缓存
_reacquire_count = 0

def _face_score(face, fw, fh):
    """综合评分：面积 × 中央偏好。领操人近+居中，得分远超背景人脸"""
    cx, cy, w2, h2 = face
    area = w2 * h2
    # 中央加权：1.0（正中）~ 0.2（四角）
    dx = abs(cx - fw / 2) / (fw / 2)
    dy = abs(cy - fh / 2) / (fh / 2)
    center_w = 1.0 - (dx * 0.5 + dy * 0.3)
    return area * max(center_w, 0.0)

def _has_skin_tone(frame, face):
    """肤色检测：过滤广告板等非人脸误检"""
    cx, cy, w2, h2 = face
    x1 = max(0, cx - w2 // 2)
    y1 = max(0, cy - h2 // 2)
    roi = frame[y1:min(frame.shape[0], y1 + h2), x1:min(frame.shape[1], x1 + w2)]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 15, 50]), np.array([25, 150, 255]))
    ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
    return ratio > 0.15

def _valid_face(face, fw, fh):
    """过滤误检：边缘/太小/比例异常/位置突变"""
    cx, cy, w2, h2 = face
    # 只检测中央区域（视频边缘的广告板/背景人不纳入）
    if not (_CX_MIN * fw <= cx <= _CX_MAX * fw and _CY_MIN * fh <= cy <= _CY_MAX * fh):
        return False
    # 最小尺寸
    if w2 < 60 or h2 < 60:
        return False
    # 最大尺寸
    if w2 > fw * 0.5 or h2 > fh * 0.5:
        return False
    # 宽高比
    aspect = w2 / max(h2, 1)
    if aspect < 0.3 or aspect > 3.0:
        return False
    # 位置突变检测（基于历史中位数）
    global _face_positions
    if len(_face_positions) >= 5:
        med_cx = np.median([p[0] for p in _face_positions])
        med_cy = np.median([p[1] for p in _face_positions])
        dx = abs(cx - med_cx) / fw
        dy = abs(cy - med_cy) / fh
        if dx > _MAX_POS_JUMP or dy > _MAX_POS_JUMP:
            return False
    return True

def detect_face(frame):
    """人脸检测：先裁掉边缘（广告板区域），再送 MediaPipe"""
    global _prev_face, _no_face_count, _face_positions
    fw, fh = frame.shape[1], frame.shape[0]
    # 裁切外 20% 边缘（广告板在边缘，裁掉后 MediaPipe 完全看不到）
    crop_x1, crop_y1 = int(fw * 0.20), int(fh * 0.20)
    crop_x2, crop_y2 = int(fw * 0.80), int(fh * 0.80)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    r = mp_face.process(rgb)
    if r.detections:
        best = None
        best_score = 0
        for det in r.detections:
            b = det.location_data.relative_bounding_box
            cfw, cfh = crop.shape[1], crop.shape[0]
            # 坐标从裁切空间映射回原帧
            face = (int((b.xmin + b.width / 2) * cfw + crop_x1),
                    int((b.ymin + b.height / 2) * cfh + crop_y1),
                    int(b.width * cfw), int(b.height * cfh))
            if not _valid_face(face, fw, fh) or not _has_skin_tone(frame, face):
                continue
            score = _face_score(face, fw, fh)
            if score > best_score:
                best = face
                best_score = score
        if best is not None:
            _prev_face = best
            _no_face_count = 0
            _face_positions.append((best[0], best[1]))
            if len(_face_positions) > 30:
                _face_positions.pop(0)
            return best
    # 检测失败：短期使用上一帧位置（保持追踪连续性）
    if _prev_face is not None and _no_face_count < _FACE_HYSTERESIS:
        _no_face_count += 1
        return _prev_face
    # 长期丢脸 → 清历史，准备重新追踪
    # 注：历史清空后，下一帧只有在中央区域的人脸才会被接受（中央加权自然保证）
    _face_positions = []
    return None

def extract_texture(frame, face):
    """提取人脸纹理"""
    cx, cy, fw, fh = face
    margin = 1.2
    rw, rh = int(fw * (1 + margin)), int(fh * (1 + margin))
    x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
    x2, y2 = min(w, x1 + rw), min(h, y1 + rh)
    roi = frame[y1:y2, x1:x2].copy()

    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    ecx, ecy = (cx - x1), (cy - y1)
    erx, ery = abs(int(fw * 0.5)), abs(int(fh * 0.65))
    cv2.ellipse(mask, (ecx, ecy), (erx, ery), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 5)[..., None] / 255.0
    return roi, mask, (x1, y1, x2, y2), (ecx, ecy)

def paste_face(dst, tex_tup, src_face, curr_face):
    """粘贴增强人脸"""
    tex, mask, roi_box, el_center = tex_tup
    cx_cur, cy_cur = curr_face[0], curr_face[1]
    sx1, sy1, sx2, sy2 = roi_box
    s_cx, s_cy = src_face[0], src_face[1]

    scale = min(curr_face[2] / max(src_face[2], 1),
                curr_face[3] / max(src_face[3], 1), 2.0)

    if abs(scale - 1.0) > 0.03:
        new_size = (int(tex.shape[1] * scale), int(tex.shape[0] * scale))
        t = cv2.resize(tex, new_size, interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(mask, new_size, interpolation=cv2.INTER_LINEAR)
        ec = (int(el_center[0] * scale), int(el_center[1] * scale))
    else:
        t, m, ec = tex, mask, el_center

    th, tw = t.shape[:2]
    dx = cx_cur - int((sx1 + sx2) // 2 * scale) - (ec[0] - tw // 2)
    dy = cy_cur - int((sy1 + sy2) // 2 * scale) - (ec[1] - th // 2)

    x1, y1 = dx, dy
    x2, y2 = x1 + tw, y1 + th

    if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
        return dst

    ox1, oy1 = max(0, x1), max(0, y1)
    ox2, oy2 = min(w, x2), min(h, y2)
    if ox2 <= ox1 or oy2 <= oy1:
        return dst

    tx1, ty1 = ox1 - x1, oy1 - y1
    tx2, ty2 = tx1 + (ox2 - ox1), ty1 + (oy2 - oy1)

    tc = t[ty1:ty2, tx1:tx2]
    mc = m[ty1:ty2, tx1:tx2]
    if mc.ndim == 2:
        mc = mc[..., None]

    out = dst.copy()
    roi = out[oy1:oy2, ox1:ox2]
    out[oy1:oy2, ox1:ox2] = (tc * mc + roi * (1 - mc)).astype(np.uint8)
    return out

# ========== 主循环 ==========
t0 = time.time()
gfpgan_count = 0

for fi in range(total):
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()

    # 人脸检测（带历史回退）
    face = detect_face(frame)

    if face is not None:
        # 丢脸后重捕获保护：需要连续 30 帧稳定检测才重建缓存
        if cached_tex is None:
            _reacquire_count += 1
        else:
            _reacquire_count = _REACQUIRE_MIN + 1  # 已锁定

        can_cache = _reacquire_count >= _REACQUIRE_MIN

        # 关键帧：裁切人脸区域跑 GFPGAN（广告板在裁切范围外）
        if fi % INTERVAL == 0 and can_cache:
            cx, cy, fw2, fh2 = face
            margin = 1.5
            rw = int(fw2 * (1 + margin))
            rh = int(fh2 * (1 + margin))
            x1_c = max(0, cx - rw // 2)
            y1_c = max(0, cy - rh // 2)
            x2_c = min(w, x1_c + rw)
            y2_c = min(h, y1_c + rh)
            face_crop = frame[y1_c:y2_c, x1_c:x2_c].copy()
            _, _, enhanced_crop = restorer.enhance(
                face_crop, has_aligned=False, only_center_face=True, paste_back=True)
            output = frame.copy()
            output[y1_c:y2_c, x1_c:x2_c] = enhanced_crop
            cached_tex = extract_texture(output, face)
            cached_face = face
            cached_kf_idx = fi
            gfpgan_count += 1
            if gfpgan_count % 5 == 0:
                print(f"  [GFPGAN x{gfpgan_count}] 帧 {fi}/{total}", flush=True)
        elif cached_tex is not None:
            # 粘贴前校验：当前人脸离缓存位置太远 → 可能跟丢，跳过
            dx = abs(face[0] - cached_face[0]) / w
            dy = abs(face[1] - cached_face[1]) / h
            if dx < 0.30 and dy < 0.30:
                output = paste_face(output, cached_tex, cached_face, face)
            else:
                cached_tex = None  # 位置跳变，清缓存重新获取
    else:
        cached_tex = None  # 丢脸 → 清缓存
        _reacquire_count = 0  # 重置重捕获计数

    writer.write(output)

    # 进度
    if (fi + 1) % (total // 40) == 0:
        el = time.time() - t0
        eta = (total - fi - 1) * el / (fi + 1)
        print(f"  [{100*(fi+1)//total}%] {fi+1}/{total} G={gfpgan_count} {el:.0f}s ETA {eta:.0f}s", flush=True)

cap.release()
writer.release()
elapsed = time.time() - t0
print(f"\n完成! {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"GFPGAN: {gfpgan_count} 帧 ({100*gfpgan_count/total:.1f}%)")
print(f"输出: {OUTPUT}")

# 从原视频拷贝音频
import shutil
from pathlib import Path
ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
audio_track = OUTPUT.replace(".mp4", "_audio.mp4")
print("合并音频...", end=" ", flush=True)
r = subprocess.run([
    ffmpeg, "-y", "-i", OUTPUT, "-i", INPUT,
    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
    "-map", "0:v:0", "-map", "1:a:0", "-shortest",
    audio_track
], capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode == 0:
    os.replace(audio_track, OUTPUT)
    print("OK")
else:
    print(f"失败: {r.stderr[-100:]}")

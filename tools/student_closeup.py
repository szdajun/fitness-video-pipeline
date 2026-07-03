#!/usr/bin/env python3
"""学员特写短视频生成器 (附加功能, 主管线零改动)

把固定广角群操视频里【某一个特定学员】做成竖版特写短视频.
靠参考照的人脸特征锁定她 → 以她为中心 9:16 数字裁切+放大(模拟推近运镜) → 全长输出.

基础只裁+缩放 (干净); 可选特效: 暖调 / 轻锐化 / 节拍卡点闪 (默认全开, --no-fx 全关).
不跑换脸/磨皮/能量条 (用户明确排除能量条).

用法:
  python tools/student_closeup.py --video source_videos/优秀学员1.mp4 \
      --ref tools/灼华娘子.jpg --name 灼华娘子 \
      --output output/2026-06-29/灼华娘子_closeup_9x16.mp4

依赖: ultralytics(YOLOv8-pose) + insightface(buffalo_l) + ffmpeg
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# 让 import face_swap 能找到项目根
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import face_swap  # 复用 get_face_analyser / extract_face_embedding / _imread_unicode

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FFMPEG = r"C:/Users/18091/ffmpeg/ffmpeg.exe"
FFPROBE = r"C:/Users/18091/ffmpeg/ffprobe.exe"

# BlazePose 33 关键点索引 (face_swap.find_lead_person / get_lead_bbox_from_pose 同一套)
NOSE, L_SHO, R_SHO = 0, 11, 12
L_HIP, R_HIP = 23, 24
L_ANK, R_ANK = 27, 28

# 人脸匹配阈值 (arcface cosine, 越高越像; buffalo_l 小脸会偏低)
FACE_MATCH_THRESH = 0.35


# ============================================================
# 1. Pose 检测 (YOLOv8-pose, 复用主流程模型, 独立缓存)
# ============================================================
def detect_pose(video_path, cache_path, device="cuda:0", model_name="yolov8m-pose"):
    """逐帧检测所有人, 返回 {frame_idx: [person_blaze33, ...]} (归一化坐标).
    person_blaze33 = [[x,y,vis]*33] x,y∈[0,1]. 缓存到 cache_path (JSON)."""
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                d = json.load(f)
            kp = d.get("keypoints", {})
            if kp:
                print(f"  [pose] 缓存命中: {cache_path.name} ({len(kp)} 帧)")
                return {int(k): v for k, v in kp.items()}, d.get("video_info", {})
        except Exception:
            pass

    from ultralytics import YOLO
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {"fps": fps, "width": w, "height": h, "frames": total}
    print(f"  [pose] {w}x{h} @ {fps:.2f}fps {total} 帧, {model_name} on {device}")

    model = YOLO(model_name)
    model.to(device)
    # NOTE: 不手动 model.half() — ultralytics 内部 fuse_conv_and_bn 会 half/float dtype 冲突崩
    # (主流程 01_pose_detect.py 也只 to(device); FP16 由 predict(half=...) 内部处理)

    # COCO-17 → BlazePose-33 关节映射. **必须含臂(肘7/8→13/14, 腕9/10→15/16) +
    # 膝(13/14→25/26)** — 旧版漏这些, 缓存里肘/腕/膝全是 vis=0, 导致 bg_swap
    # `_pose_arm_core_matte` 抓不到胳膊关节, arm bolster 失效 (2026-07-03 修).
    # BlazePose33 索引: 0鼻 11/12肩 13/14肘 15/16腕 23/24髋 25/26膝 27/28踝.
    COCO2BLAZE = {0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
                  11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}

    def to_blaze33(coco):
        b = [[0.0, 0.0, 0.0] for _ in range(33)]
        for ci, bi in COCO2BLAZE.items():
            b[bi] = [float(coco[ci][0]) / w, float(coco[ci][1]) / h,
                     float(min(max(coco[ci][2], 0.0), 1.0))]
        b[1] = b[0][:]                       # 鼻→左眼内 (兜底非0)
        b[31] = b[23][:]; b[32] = b[24][:]   # 髋→足趾 (兜底)
        # 注: 旧 b[16/17/18/19]=肩 的副本已删 — 它们会把肩坐标盖到腕槽 (blaze 16=右腕),
        # 覆盖 COCO 10→blaze 16 的真腕数据, 使腕永远停在肩位置 (脏数据).
        return b

    keypoints = {}
    fi = 0
    while fi < total:
        ret, frame = cap.read()
        if not ret:
            break
        res = model(frame, verbose=False, conf=0.3)[0]
        if res.keypoints is not None and len(res.keypoints) > 0:
            kpts = res.keypoints.data.cpu().numpy()
            people = [to_blaze33(p) for p in kpts]
            keypoints[fi] = people
        else:
            keypoints[fi] = []
        fi += 1
        if fi % 200 == 0:
            print(f"  [pose] {fi}/{total} ({fi*100//total}%)", flush=True)
    cap.release()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"keypoints": {str(k): v for k, v in keypoints.items()},
                   "video_info": info}, f)
    print(f"  [pose] 完成 {len(keypoints)} 帧, 缓存 {cache_path.name}")
    return keypoints, info


# ============================================================
# 2. 认人: 参考照 embedding → 在采样帧上逐人匹配 → 定锚 x 列
# ============================================================
def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def person_face_embedding(app, frame, person, w, h):
    """从 person 的鼻/肩关键点裁脸部 ROI, 上采样到 ≥512px, 跑 insightface.
    返回 (embedding, bbox) 或 (None, None). 小脸靠上采样救回."""
    nose = person[NOSE]
    ls, rs = person[L_SHO], person[R_SHO]
    if nose[2] < 0.3:
        return None, None
    cx, cy = nose[0] * w, nose[1] * h
    if ls[2] > 0.3 and rs[2] > 0.3:
        sh_w = abs(ls[0] - rs[0]) * w
    else:
        sh_w = 100
    size = max(int(sh_w * 1.5), 160)
    half = size // 2
    x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
    x2, y2 = min(w, int(cx + half)), min(h, int(cy + half))
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None, None
    roi = frame[y1:y2, x1:x2].copy()
    # 上采样小脸 (insightface det_size=640 不放大小 ROI)
    if max(roi.shape[:2]) < 512:
        roi = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_CUBIC)
    faces = app.get(roi)
    if not faces:
        return None, None
    # ROI 内取最居中的脸 (避免远处路人)
    rcx = rcy = 256
    best = min(faces, key=lambda f: ((f.bbox[0]+f.bbox[2])/2 - rcx)**2 +
               ((f.bbox[1]+f.bbox[3])/2 - rcy)**2)
    return best.embedding, best.bbox


def identify_target(pose, ref_emb, video_path, w, h):
    """采样每隔 ~1s 一帧, 逐人 face-match, 按相似度加权求锚 x (归一化).
    返回 anchor_x_norm (她所在的画面横坐标)."""
    app = face_swap.get_face_analyser()
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pairs = []  # (center_x_norm, sim, det_score)
    step = max(1, int(cap.get(cv2.CAP_PROP_FPS)))  # ~1s
    sample_frames = list(range(0, total, step))
    print(f"  [认人] 采样 {len(sample_frames)} 帧做人脸匹配 (阈值 {FACE_MATCH_THRESH})")
    best_sim_global = 0.0
    for fi in sample_frames:
        if fi not in pose or not pose[fi]:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        for person in pose[fi]:
            emb, _ = person_face_embedding(app, frame, person, w, h)
            if emb is None:
                continue
            sim = cos_sim(emb, ref_emb)
            best_sim_global = max(best_sim_global, sim)
            if sim >= FACE_MATCH_THRESH:
                # 用肩膀中点 x 当作该人列位置 (比鼻尖稳)
                ls, rs = person[L_SHO], person[R_SHO]
                if ls[2] > 0.3 and rs[2] > 0.3:
                    cx = (ls[0] + rs[0]) / 2
                else:
                    cx = person[NOSE][0]
                pairs.append((cx, sim))
    cap.release()

    if not pairs:
        print(f"  [认人][WARN] 全程人脸匹配失败 (最高 sim={best_sim_global:.3f}), "
              f"降级: 用 pose 里最居中的列当锚")
        # 兜底: 取所有帧所有可见人的 cx 中位数里偏右的 (用户说她中间偏右)
        allcx = []
        for ps in pose.values():
            if not ps:
                continue
            for p in ps:
                if p[NOSE][2] > 0.3:
                    allcx.append(p[NOSE][0])
        anchor = float(np.median(allcx)) if allcx else 0.5
        return anchor, 0.0

    # 加权中位数 (sim 越高权重越大)
    pairs.sort(key=lambda t: t[0])
    weights = np.array([t[1] for t in pairs])
    xs = np.array([t[0] for t in pairs])
    cum = np.cumsum(weights) / weights.sum()
    idx = int(np.searchsorted(cum, 0.5))
    anchor = float(xs[min(idx, len(xs) - 1)])
    print(f"  [认人] 锚定 x={anchor:.3f} (画面宽比例), 匹配 {len(pairs)} 次, "
          f"最高 sim={best_sim_global:.3f}")
    return anchor, best_sim_global


# ============================================================
# 3. 逐帧锁定目标 → 平滑裁切窗口
# ============================================================
def _person_metrics(person, w, h):
    """从 blaze33 取 (cx_px, cy_mid_px, body_h_px). 缺关键点返回 None."""
    nose = person[NOSE]
    ls, rs = person[L_SHO], person[R_SHO]
    if nose[2] < 0.3:
        return None
    # 横向中心: 优先肩中点
    if ls[2] > 0.3 and rs[2] > 0.3:
        cx = (ls[0] + rs[0]) / 2 * w
    else:
        cx = nose[0] * w
    # 纵向: 头(鼻)到脚(踝) 的高度
    ankles = [person[i] for i in (L_ANK, R_ANK) if person[i][2] > 0.3]
    hips = [person[i] for i in (L_HIP, R_HIP) if person[i][2] > 0.3]
    nose_y = nose[1] * h
    if ankles:
        foot_y = np.mean([a[1] for a in ankles]) * h
        body_h = foot_y - nose_y
        cy = (nose_y + foot_y) / 2
    elif hips:
        hip_y = np.mean([p[1] for p in hips]) * h
        body_h = (hip_y - nose_y) * 1.9  # 髋到鼻 ≈ 全身/1.9
        cy = (nose_y + hip_y) / 2
    else:
        body_h = 300
        cy = nose_y
    return float(cx), float(cy), float(body_h)


def per_frame_crops(pose, anchor_x, total, w, h, smooth_window=21,
                    zoom_start=1.8, zoom_end=1.4, cy_bias=0.40):
    """逐帧算 9:16 裁切窗口 (x1,y1,x2,y2 像素).
    锚 x 锁定目标列; 时间中位数平滑 cx/cy/body_h; zoom 随时间从宽→紧 (推近).
    cy_bias: 纵向裁切中心 = 鼻下方 cy_bias×体高 (0.5=身体中点全身; 0.30=胸口, 多留头/举手,
             少留脚下 → 收窄横向 crop_w 切掉身后那排人, 同时保住脸+举手)."""
    raw = {}
    for fi in range(total):
        ps = pose.get(fi, [])
        best = None
        best_dx = 1e9
        for person in ps:
            m = _person_metrics(person, w, h)
            if m is None:
                continue
            # 离锚 x 最近的可见人 = 她 (固定机位, 她列位置不动)
            dx = abs(m[0] / w - anchor_x)
            if dx < best_dx:
                best_dx = dx
                best = m
        raw[fi] = best  # 可能 None

    # 中位数平滑 (定机位下她几乎不动, 平滑去抖)
    def smooth(key_idx):
        vals = []
        for off in range(-smooth_window, smooth_window + 1):
            j = key_idx + off
            if 0 <= j < total and raw[j] is not None:
                vals.append(raw[j][key_idx])
        return float(np.median(vals)) if vals else None

    crops = []
    last_good = None
    for fi in range(total):
        cxs, cys, bhs = [], [], []
        for off in range(-smooth_window, smooth_window + 1):
            j = fi + off
            if 0 <= j < total and raw[j] is not None:
                cxs.append(raw[j][0]); cys.append(raw[j][1]); bhs.append(raw[j][2])
        if cxs:
            cx, cy, bh = float(np.median(cxs)), float(np.median(cys)), float(np.median(bhs))
            last_good = (cx, cy, bh)
        elif last_good is None:
            crops.append(None)
            continue
        else:
            cx, cy, bh = last_good

        # 推近运镜: zoom 系数随时间从 zoom_start→zoom_end
        t_frac = fi / max(1, total - 1)
        zf = zoom_start + (zoom_end - zoom_start) * t_frac
        crop_h = min(h - 2, bh * zf)
        crop_h = max(crop_h, bh * 1.1)  # 下限 1.1×体高 (源里人小~200px, 裁太紧放大致糊)
        crop_w = crop_h * 9.0 / 16.0
        if crop_w > w:
            crop_w = w
            crop_h = crop_w * 16.0 / 9.0

        # 横向居中于她; 纵向上移到胸口(cy_bias)多留头/举手, 少留脚下/背后人
        x1 = cx - crop_w / 2
        nose_y = cy - bh / 2  # cy=身体中点 → 反推鼻位
        cy_center = nose_y + cy_bias * bh
        y1 = cy_center - crop_h / 2
        x1 = max(0, min(x1, w - crop_w))
        y1 = max(0, min(y1, h - crop_h))
        crops.append((int(round(x1)), int(round(y1)),
                      int(round(crop_w)), int(round(crop_h))))
    found = sum(1 for c in crops if c is not None)
    print(f"  [裁切] {found}/{total} 帧有锚 (中位数平滑窗口 ±{smooth_window})")
    return crops


# ============================================================
# 3.5 特效: 暖调 / 轻锐化 / 节拍卡点闪 (除能量条外, 用户要的都加)
#    全 numpy 逐帧, 与主管线零耦合; 任一特效可独立开关.
# ============================================================
def _extract_audio(video_path, out_wav):
    """ffmpeg 提取音频 (mono 22050) 给 librosa 检拍."""
    cmd = [FFMPEG, "-y", "-i", str(video_path),
           "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
           str(out_wav)]
    subprocess.run(cmd, capture_output=True)
    return out_wav.exists()


def _detect_beats(audio_path, fps, total):
    """librosa 检测强拍 (秒→帧号). 复用主管线 stages/17 同套规则 (BPM≥100 隔拍取强拍).
    librosa 没装/失败 → 返回 [] (不闪, 不阻断其他特效)."""
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
        try:
            bpm = float(tempo)
        except (TypeError, ValueError):
            bpm = float(tempo[0]) if hasattr(tempo, "__len__") and len(tempo) else 120.0
        max_sec = total / fps
        beats = sorted({int(round(float(t) * fps)) for t in beat_times
                        if 0.0 <= float(t) < max_sec})
        if 30 <= len(beats) <= total and bpm >= 100:
            beats = beats[::2]  # 强拍: 每 2 拍闪 1 次 (抄 stages/17)
        print(f"  [节拍] tempo={bpm:.1f}BPM, {len(beats)} 个强拍帧")
        return beats
    except Exception as e:
        print(f"  [节拍][WARN] librosa 不可用 ({e}), 跳过节拍闪")
        return []


def _warm_grade(frame, d_r=10, d_b=10, sat=14):
    """暖调: +R -B +饱和 (补偿远景广角偏冷, 增感染力). 全帧常量."""
    b, g, r = cv2.split(frame.astype(np.int16))
    r = np.clip(r + d_r, 0, 255)
    b = np.clip(b - d_b, 0, 255)
    out = cv2.merge([b, g, r]).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) + sat, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _sharpen(frame, amount=1.0):
    """unsharp mask 锐化 (源里人小放大必糊, 补清晰度).
    sigma 要够大 (2.0) 让 blur 真模糊掉细节, unsharp 才有效; sigma=1.2 实测无效."""
    blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(frame, 1 + amount, blur, -amount, 0)


def _flash_strength(fi, beat_set, duration=6, alpha=0.34):
    """该帧节拍闪强度 [0, alpha], 衰减窗口 duration 帧 (抄 stages/17)."""
    for f in range(fi - duration + 1, fi + 1):
        if f in beat_set:
            return alpha * (1.0 - (fi - f) / duration)
    return 0.0


def apply_effects(frame, fi, beat_set, fx):
    """统一入口: warm → sharpen → beat_flash (闪最后叠加, 最显眼)."""
    if fx.get("warm"):
        frame = _warm_grade(frame)
    if fx.get("sharpen"):
        frame = _sharpen(frame, fx.get("sharpen_amt", 0.6))
    if fx.get("flash") and beat_set:
        s = _flash_strength(fi, beat_set)
        if s > 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + s), 0, 255).astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame


# ============================================================
# 4. 渲染: 逐帧裁+放大到 1080×1920, pipe 到 ffmpeg (含原音频)
# ============================================================
def render(video_path, crops, output, info, fx=None, beats=None, out_w=1080, out_h=1920):
    fps = info["fps"]
    beat_set = set(beats) if beats else set()
    fx = fx or {}
    on = [k for k in ("warm", "sharpen", "flash") if fx.get(k)]
    cap = cv2.VideoCapture(str(video_path))
    total = len(crops)
    if on:
        print(f"  [特效] {' + '.join(on)} (强拍帧 {len(beat_set)})")

    # 探 nvenc, 不可用退 libx264
    enc = "h264_nvenc"
    probe = subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                           capture_output=True, text=True)
    if "h264_nvenc" not in (probe.stdout or ""):
        enc = "libx264"
    if enc == "h264_nvenc":
        vcmd = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-b:v", "8M", "-pix_fmt", "yuv420p"]
    else:
        vcmd = ["-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p"]
    print(f"  [渲染] 编码器 {enc}, 输出 {out_w}x{out_h} @ {fps:.2f}fps")

    cmd = [FFMPEG, "-y",
           "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24", "-r", f"{fps:.3f}",
           "-i", "pipe:0",
           "-i", str(video_path),
           "-map", "0:v:0", "-map", "1:a:0?",
           *vcmd, "-c:a", "aac", "-b:a", "160k", "-shortest",
           "-movflags", "+faststart", str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    fi = 0
    last_crop = None
    while fi < total:
        ret, frame = cap.read()
        if not ret:
            break
        c = crops[fi] if fi < len(crops) and crops[fi] else last_crop
        if c is None:
            fi += 1
            continue
        last_crop = c
        x1, y1, cw, ch = c
        crop = frame[y1:y1 + ch, x1:x1 + cw]
        out = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        if fx:
            out = apply_effects(out, fi, beat_set, fx)
        try:
            proc.stdin.write(out.tobytes())
        except (BrokenPipeError, OSError):
            print(f"  [渲染] pipe 断在第 {fi} 帧")
            break
        fi += 1
        if fi % 300 == 0:
            print(f"  [渲染] {fi}/{total} ({fi*100//total}%)", flush=True)
    cap.release()
    try:
        proc.stdin.close()
    except Exception:
        pass
    rc = proc.wait()
    if rc != 0 or not output.exists():
        print(f"  [渲染][FAIL] ffmpeg rc={rc}")
        return False
    print(f"  [渲染] 完成: {output}")
    return True


# ============================================================
# 5. 文案生成 (视频干净, 文案单出 .txt)
# ============================================================
def write_copy(name, output_mp4):
    txt = output_mp4.with_suffix(".文案.txt")
    title = f"【{name}】户外燃脂操 | 跟练打卡 | 细柳营健身"
    desc = (
        f"{name} 带练 · 户外有氧健身操\n"
        f"\n"
        f"跟着节奏动起来，一身暴汗，越跳越年轻。\n"
        f"零基础友好，不伤膝盖，居家/户外都能练。\n"
        f"\n"
        f"关注 @细柳营健身，每天一段跟练，陪你瘦下来。\n"
        f"点赞 · 收藏 · 转发，给 {name} 鼓鼓劲！\n"
    )
    tags = "#有氧健身操 #暴汗燃脂 #跟练 #户外健身 #细柳营健身 #燃脂操 #瘦全身"
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"=== 标题 ===\n{title}\n\n")
        f.write(f"=== 描述 ===\n{desc}\n\n")
        f.write(f"=== 标签 ===\n{tags}\n")
    print(f"  [文案] {txt}")
    return txt


# ============================================================
# debug: 把裁切窗口画到采样帧上拼成检查图
# ============================================================
def debug_contact(video_path, crops, anchor_x, out_png, n=8, out_w=1080, out_h=1920):
    cap = cv2.VideoCapture(str(video_path))
    total = len(crops)
    times = [int(total * (i + 0.5) / n) for i in range(n)]
    thumbs = []
    for fi in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        c = crops[fi] if fi < len(crops) else None
        if c is None:
            continue
        x1, y1, cw, ch = c
        # 在原图上画裁切框
        vis = frame.copy()
        cv2.rectangle(vis, (x1, y1), (x1 + cw, y1 + ch), (0, 255, 255), 4)
        # 锚线
        cv2.line(vis, (int(anchor_x * vis.shape[1]), 0),
                 (int(anchor_x * vis.shape[1]), vis.shape[0]), (0, 0, 255), 2)
        cv2.rectangle(vis, (0, 0), (160, 44), (0, 0, 0), -1)
        cv2.putText(vis, f"f={fi}", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2)
        # 缩略图 = 裁切结果 (实际输出)
        crop = frame[y1:y1 + ch, x1:x1 + cw]
        out = cv2.resize(crop, (out_w // 3, out_h // 3), interpolation=cv2.INTER_LANCZOS4)
        # 上下拼: 带框原图 + 裁切结果
        sm_vis = cv2.resize(vis, (out.shape[1], int(vis.shape[0] * out.shape[1] / vis.shape[1])))
        thumbs.append(np.vstack([sm_vis, out]))
    cap.release()
    if not thumbs:
        return None
    # 统一宽 → 拼成 2 列
    mh = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
        if t.shape[0] < mh:
            t = np.vstack([t, np.zeros((mh - t.shape[0], t.shape[1], 3), dtype=np.uint8)])
        padded.append(t)
    while len(padded) % 2:
        padded.append(np.zeros_like(padded[0]))
    rows = [np.hstack(padded[i:i + 2]) for i in range(0, len(padded), 2)]
    grid = np.vstack(rows)
    cv2.imencode(".png", grid)[1].tofile(str(out_png))
    print(f"  [debug] 检查图: {out_png} (黄框=裁切区, 红线=锚x列)")
    return out_png


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="学员特写短视频 (竖版9:16, 主管线零改动)")
    ap.add_argument("--video", required=True, help="源视频")
    ap.add_argument("--ref", required=True, help="目标学员参考照 (认人用)")
    ap.add_argument("--name", default="学员", help="花名 (用于文案/输出名)")
    ap.add_argument("--output", required=True, help="输出 9:16 mp4 路径")
    ap.add_argument("--zoom-start", type=float, default=1.8, help="起始zoom×体高 (建立镜头, 全身+周围)")
    ap.add_argument("--zoom-end", type=float, default=1.4, help="结束zoom×体高 (推近, 主体更大)")
    ap.add_argument("--cy-bias", type=float, default=0.40,
                    help="纵向裁切中心=鼻下方×体高 (0.5=身体中点全身; 0.30=胸口特写)")
    ap.add_argument("--no-warm", action="store_true", help="关闭暖调调色")
    ap.add_argument("--no-sharpen", action="store_true", help="关闭轻锐化")
    ap.add_argument("--no-flash", action="store_true", help="关闭节拍卡点闪")
    ap.add_argument("--no-fx", action="store_true", help="全关特效 (干净版)")
    ap.add_argument("--sharpen-amt", type=float, default=1.0, help="锐化强度 0.5~1.5")
    ap.add_argument("--no-render", action="store_true", help="只跑认人+出检查图, 不渲染")
    args = ap.parse_args()

    video = Path(args.video)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path("_temp")
    tmp.mkdir(exist_ok=True)
    cache = tmp / f"{video.stem}_closeup_keypoints.json"

    print(f"=== {args.name} 特写短视频 ===")
    print(f"源: {video.name}  参考照: {Path(args.ref).name}")

    # 1) pose
    print("[1/5] Pose 检测...")
    pose, info = detect_pose(str(video), cache)
    w, h = info["width"], info["height"]
    total = info["frames"]

    # 2) 认人
    print("[2/5] 参考照人脸特征 + 锁定目标列...")
    app = face_swap.get_face_analyser()
    ref_face = face_swap.extract_face_embedding(app, args.ref)
    ref_emb = ref_face.embedding
    anchor_x, best_sim = identify_target(pose, ref_emb, str(video), w, h)

    # 3) 裁切窗口
    print("[3/5] 计算 9:16 裁切窗口 (推近运镜)...")
    crops = per_frame_crops(pose, anchor_x, total, w, h,
                            zoom_start=args.zoom_start, zoom_end=args.zoom_end,
                            cy_bias=args.cy_bias)

    # 4) 检查图 (永远生成, 方便核对认人/取景)
    dbg = tmp / f"{video.stem}_closeup_debug.png"
    debug_contact(str(video), crops, anchor_x, dbg)

    if args.no_render:
        print("[no-render] 只出了检查图, 看一眼再渲染")
        return

    # 5) 特效 + 渲染 + 文案
    fx = {} if args.no_fx else {
        "warm": not args.no_warm,
        "sharpen": not args.no_sharpen,
        "flash": not args.no_flash,
        "sharpen_amt": args.sharpen_amt,
    }
    beats = []
    if fx.get("flash"):
        awav = tmp / f"{video.stem}_closeup_audio.wav"
        if _extract_audio(str(video), awav):
            beats = _detect_beats(awav, info["fps"], total)
    print(f"[4/5] 渲染竖版视频 (warm={fx.get('warm')} sharpen={fx.get('sharpen')} flash={fx.get('flash')})...")
    ok = render(str(video), crops, output, info, fx=fx, beats=beats)
    if not ok:
        print("[FAIL] 渲染失败, 看 _temp 检查图诊断")
        return
    print("[5/5] 生成文案...")
    write_copy(args.name, output)
    print(f"\n=== 完成 ===\n视频: {output}\n检查图: {dbg}")
    print(f"认人 sim: {best_sim:.3f}  锚x: {anchor_x:.3f}")


if __name__ == "__main__":
    main()

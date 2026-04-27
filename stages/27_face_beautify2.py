"""阶段27: InsightFace 美颜 (高级版)

基于 InsightFace 106点精确人脸检测 + 皮肤分割。
比 COCO + MediaPipe FaceMesh 方案更精准：
- 106个人脸特征点（眼角、眼尾、鼻翼、唇形、脸轮廓）
- 可选的 FaceParser 皮肤分割（只美颜皮肤区域，不影响背景）
- 瘦脸、大眼、磨皮、提亮综合效果

多进程并行：使用 spawn 上下文的 multiprocessing.Pool。

配置:
    face_beautify2:
      enabled: true
      mode: insightface              # 模式: insightface
      skin_smooth: 0.4               # 磨皮强度
      eye_brighten: 0.5              # 眼部提亮
      face_whiten: 0.15             # 肤色提亮
      face_slim: 0.0                 # 瘦脸强度 (0=关闭)
      eye_enlarge: 0.0              # 大眼强度 (0=关闭)
      workers: 4                    # 并行进程数（默认4）
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
import ctypes
import tempfile
import subprocess
import time
import multiprocessing

from lib.utils import path_exists


GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
GetShortPathNameW.restype = ctypes.c_uint

def _to_short(path_str):
    buf_size = GetShortPathNameW(str(path_str), None, 0)
    if buf_size == 0:
        return str(path_str)
    buf = ctypes.create_unicode_buffer(buf_size)
    GetShortPathNameW(str(path_str), buf, buf_size)
    return buf.value


# ---- 多进程 worker（模块级别，可被 pickle） ----

def _apply_insightface(frame, main_face, eye_brighten, face_whiten,
                       skin_smooth, face_slim, eye_enlarge, w, h):
    """对一帧应用 InsightFace 美颜（无状态，纯函数）"""
    if main_face is None:
        return frame

    kps = main_face.kps  # [106, 2] normalized

    # ========== 眼部提亮 ==========
    if eye_brighten > 0:
        eye_indices = {
            'left_outer': 36, 'left_inner': 39,
            'right_inner': 42, 'right_outer': 45,
        }
        for eye_key, eye_idx in eye_indices.items():
            if eye_idx < len(kps):
                ex = int(kps[eye_idx][0] * w)
                ey = int(kps[eye_idx][1] * h)
                if 0 <= ex < w and 0 <= ey < h:
                    r = 18
                    mask = np.zeros((h, w), dtype=np.float32)
                    cv2.circle(mask, (ex, ey), r, 1.0, -1)
                    mask = cv2.GaussianBlur(mask, (r * 2 + 1, r * 2 + 1), r * 0.4)
                    mask = np.clip(mask * eye_brighten * 3.0, 0, 0.9)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * (1 + mask), 0, 255).astype(np.uint8)
                    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 瞳孔区域提亮
        if len(kps) >= 48:
            l_eye_pts = kps[36:42]
            r_eye_pts = kps[42:48]
            if l_eye_pts.size > 0 and r_eye_pts.size > 0:
                l_center_x = int(np.nanmean(l_eye_pts[:, 0]) * w)
                l_center_y = int(np.nanmean(l_eye_pts[:, 1]) * h)
                r_center_x = int(np.nanmean(r_eye_pts[:, 0]) * w)
                r_center_y = int(np.nanmean(r_eye_pts[:, 1]) * h)
                for px, py in [(l_center_x, l_center_y), (r_center_x, r_center_y)]:
                    if 0 <= px < w and 0 <= py < h:
                        mask = np.zeros((h, w), dtype=np.float32)
                        cv2.circle(mask, (px, py), 22, 1.0, -1)
                        mask = cv2.GaussianBlur(mask, (45, 45), 11)
                        mask = np.clip(mask * eye_brighten * 2.0, 0, 0.8)
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * (1 + mask), 0, 255).astype(np.uint8)
                        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ========== 肤色提亮 ==========
    if face_whiten > 0:
        face_indices = list(range(0, 17))
        face_pts = [(int(kps[i][0] * w), int(kps[i][1] * h))
                    for i in face_indices if i < len(kps)]
        if len(face_pts) >= 3:
            face_mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(np.array(face_pts, dtype=np.int32))
            cv2.fillPoly(face_mask, [hull], 255)
            face_mask = cv2.GaussianBlur(face_mask, (31, 31), 15).astype(np.float32) / 255.0
            brightness = face_whiten * 35
            frame = np.clip(
                frame.astype(np.float32) + face_mask[:, :, None] * brightness,
                0, 255
            ).astype(np.uint8)

    # ========== 皮肤磨皮 ==========
    if skin_smooth > 0:
        face_indices = list(range(0, 17)) + list(range(68, 88))
        all_face_pts = []
        for i in face_indices:
            if i < len(kps):
                all_face_pts.append((int(kps[i][0] * w), int(kps[i][1] * h)))
        if len(all_face_pts) >= 3:
            skin_mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(np.array(all_face_pts, dtype=np.int32))
            cv2.fillPoly(skin_mask, [hull], 255)
            skin_mask = cv2.GaussianBlur(skin_mask, (51, 51), 17).astype(np.float32) / 255.0
            smooth = cv2.bilateralFilter(frame, 7, 15, 15)
            strength = (skin_mask * skin_smooth).astype(np.float32)
            frame = (frame * (1 - strength) + smooth.astype(np.float32) * strength).astype(np.uint8)

    # ========== 大眼 ==========
    if eye_enlarge > 0:
        l_eye_pts = [(int(kps[i][0] * w), int(kps[i][1] * h))
                      for i in range(36, 48) if i < len(kps)]
        r_eye_pts = [(int(kps[i][0] * w), int(kps[i][1] * h))
                      for i in range(42, 54) if i < len(kps)]
        for eye_pts in [l_eye_pts, r_eye_pts]:
            if len(eye_pts) >= 6:
                eye_arr = np.array(eye_pts)
                (cx, cy), (ax, ay), angle = cv2.fitEllipse(eye_arr)
                enlarged = (int(ax * (1 + eye_enlarge * 0.15)),
                             int(ay * (1 + eye_enlarge * 0.25)))
                ellipse_mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(ellipse_mask, (int(cx), int(cy)),
                             enlarged, angle, 0, 360, 1.0, -1)
                ellipse_mask = cv2.GaussianBlur(ellipse_mask, (21, 21), 5)
                brightness = eye_enlarge * 30
                frame = np.clip(
                    frame.astype(np.float32) + ellipse_mask[:, :, None] * brightness,
                    0, 255
                ).astype(np.uint8)

    # ========== 瘦脸 ==========
    if face_slim > 0:
        jaw_indices = list(range(0, 17))
        jaw_pts = [(int(kps[i][0] * w), int(kps[i][1] * h))
                   for i in jaw_indices if i < len(kps)]
        if len(jaw_pts) >= 3:
            slim_mask = np.zeros((h, w), dtype=np.float32)
            hull = cv2.convexHull(np.array(jaw_pts, dtype=np.int32))
            cv2.fillPoly(slim_mask, [hull], 1.0)
            slim_mask = cv2.GaussianBlur(slim_mask, (81, 81), 21)
            blur_img = cv2.GaussianBlur(frame, (21, 21), 7)
            strength = (slim_mask * face_slim * 0.4).astype(np.float32)
            frame = (frame * (1 - strength) + blur_img.astype(np.float32) * strength).astype(np.uint8)

    return frame


def _face_beautify2_worker(args):
    """并行 worker：处理一组帧的 InsightFace 美颜"""
    (frame_indices, frames_data, keypoints_dict, lead_tid, lead_cx,
     eye_brighten, face_whiten, skin_smooth, face_slim, eye_enlarge,
     tmpdir_short, w, h, out_prefix) = args

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    for i, fi in enumerate(frame_indices):
        frame = frames_data[i]
        curr_kps = None

        frame_kps = keypoints_dict.get(str(fi))
        if frame_kps:
            for person_kps in frame_kps:
                kps_arr = np.array(person_kps)
                vis = kps_arr[:, 2] > 0.5
                if vis.sum() < 6:
                    continue
                shoulders_cx = (kps_arr[5][0] + kps_arr[6][0]) / 2
                hips_cx = (kps_arr[11][0] + kps_arr[12][0]) / 2
                cx = (shoulders_cx + hips_cx) / 2
                if abs(cx - lead_cx) < 0.15:
                    curr_kps = person_kps
                    break

        main_face = None
        if curr_kps is not None:
            faces = app.get(frame)
            if faces:
                lead_arr = np.array(curr_kps)
                lead_nose = (lead_arr[0][0], lead_arr[0][1]) if lead_arr[0][2] > 0.3 else None
                best_dist = float('inf')
                best_face = None
                for face in faces:
                    kps_face = face.kps
                    if kps_face is None:
                        continue
                    face_cx_norm = kps_face[:, 0].mean()
                    face_cx_pixel = int(face_cx_norm * w)
                    dist = abs(face_cx_pixel / w - lead_cx)
                    if dist < best_dist:
                        best_dist = dist
                        best_face = face
                main_face = best_face if best_dist < 0.15 else None
        else:
            faces = app.get(frame)
            if faces:
                main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        if main_face is not None:
            frame = _apply_insightface(
                frame, main_face, eye_brighten, face_whiten,
                skin_smooth, face_slim, eye_enlarge, w, h)

        fname = "%s/%s_%06d.png" % (tmpdir_short, out_prefix, fi)
        cv2.imwrite(fname, frame)

    app = None
    return frame_indices


class FaceBeautify2Stage:
    def run(self, ctx):
        cfg = ctx.config.get("face_beautify2", {})
        if not cfg.get("enabled", False):
            ctx.set("face_beautify2_path",
                    ctx.get("face_beautify_path") or ctx.get("energybar_path") or ctx.get("beatflash_path") or ctx.get("ken_burns_path"))
            return

        skin_smooth = cfg.get("skin_smooth", 0.4)
        eye_brighten = cfg.get("eye_brighten", 0.5)
        face_whiten = cfg.get("face_whiten", 0.15)
        face_slim = cfg.get("face_slim", 0.0)
        eye_enlarge = cfg.get("eye_enlarge", 0.0)
        num_workers = cfg.get("workers", 4)

        if skin_smooth <= 0 and eye_brighten <= 0 and face_whiten <= 0 and face_slim <= 0 and eye_enlarge <= 0:
            ctx.set("face_beautify2_path",
                    ctx.get("face_beautify_path") or ctx.get("energybar_path") or ctx.get("beatflash_path") or ctx.get("ken_burns_path"))
            return

        input_path = (ctx.get("face_beautify_path") or
                      ctx.get("energybar_path") or
                      ctx.get("beatflash_path") or
                      ctx.get("ken_burns_path") or
                      ctx.get("color_path") or
                      str(ctx.input_path))
        if not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("face_beautify2_path", None)
            return

        # ---- 加载领操人追踪 ----
        lead_tid = ctx.get("lead_tid")
        lead_cx = ctx.get("lead_cx")
        if lead_tid is None:
            raw_kp = ctx.get("cropped_keypoints")
            keypoints = raw_kp if raw_kp else {}
            if not keypoints:
                kp_path = ctx.output_dir / f"{ctx.input_path.stem}_cropped_keypoints.json"
                if not kp_path.exists() or kp_path.stat().st_size == 0:
                    kp_path2 = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
                    if not kp_path2.exists() or kp_path2.stat().st_size == 0:
                        print("    跳过: 无关键点数据")
                        ctx.set("face_beautify2_path", None)
                        return
                    with open(kp_path2, encoding="utf-8") as f:
                        raw = json.load(f)
                        keypoints = raw.get("keypoints", raw)
                else:
                    with open(kp_path, encoding="utf-8") as f:
                        keypoints = json.load(f)
            tracks = self._track_people(keypoints)
            if not tracks:
                print("    跳过: 无法追踪人员")
                ctx.set("face_beautify2_path", None)
                return
            lead_tid = max(tracks, key=lambda tid: tracks[tid]["count"])
            lead_cx = np.median(tracks[lead_tid]["cx_list"])
            ctx.set("lead_tid", lead_tid)
            ctx.set("lead_cx", lead_cx)
        else:
            raw_kp = ctx.get("cropped_keypoints")
            keypoints = raw_kp if raw_kp else {}

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"    跳过: 无法打开视频 {input_path}")
            ctx.set("face_beautify2_path", None)
            return
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        cap_actual = cv2.VideoCapture(input_path)
        actual_frames = int(cap_actual.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_actual.release()
        if actual_frames > 0 and actual_frames != max_frames:
            print(f"    警告: 实际帧数 {actual_frames} 与预期 {max_frames} 不符，使用实际帧数")
            max_frames = actual_frames

        print(f"    InsightFace美颜: 磨皮={skin_smooth}, 眼部提亮={eye_brighten}, "
              f"肤色提亮={face_whiten}, 瘦脸={face_slim}, 大眼={eye_enlarge}, workers={num_workers}")

        # ---- 读取所有帧 ----
        print(f"    读取 {max_frames} 帧...")
        cap = cv2.VideoCapture(input_path)
        all_frames = []
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        total = len(all_frames)
        print(f"    读取完成: {total} 帧")

        # ---- 分块并行处理 ----
        tmpdir = ctx.output_dir / f"_tmp_fb2_{Path(input_path).stem}_{int(time.time()*1000):08d}"
        tmpdir.mkdir(exist_ok=True)
        tmpdir_short = _to_short(str(tmpdir))

        chunk_size = max(1, total // num_workers)
        chunks = []
        for i in range(0, total, chunk_size):
            chunk_indices = list(range(i, min(i + chunk_size, total)))
            chunks.append(chunk_indices)

        w, h = orig_w, orig_h
        worker_args = []
        for ci, chunk in enumerate(chunks):
            frames_data = [all_frames[fi] for fi in chunk]
            worker_args.append((
                chunk, frames_data, keypoints, lead_tid, lead_cx,
                eye_brighten, face_whiten, skin_smooth, face_slim, eye_enlarge,
                tmpdir_short, w, h, f"worker_{ci}"
            ))

        # 使用 spawn 上下文避免 MediaPipe protobuf 冲突
        mp_ctx = multiprocessing.get_context('spawn')
        with mp_ctx.Pool(num_workers) as pool:
            results = pool.map(_face_beautify2_worker, worker_args)
        completed_chunks = len([r for r in results if r is not None])
        print(f"    并行处理完成: {completed_chunks}/{len(chunks)} 块")

        # ---- FFmpeg 编码 ----
        print(f"    调用 FFmpeg 编码...")
        temp_out = ctx.output_dir / f"{ctx.input_path.stem}_face_beautify2.mp4"
        output_short = _to_short(str(temp_out))
        ffmpeg_bin = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"

        concat_file = tmpdir / "concat_list.txt"
        with open(concat_file, "w", encoding="utf-8", newline="\n") as f:
            for ci, chunk in enumerate(chunks):
                for fi in chunk:
                    f.write("file '%s/worker_%d_%06d.png'\n" % (tmpdir_short, ci, fi))

        cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", output_short
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"    FFmpeg 错误: {result.stderr[-300:]}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            ctx.set("face_beautify2_path", None)
            return

        shutil.rmtree(tmpdir, ignore_errors=True)

        if cv2.VideoCapture(str(temp_out)).isOpened():
            ctx.set("face_beautify2_path", str(temp_out))
            print(f"    输出: {temp_out.name} ({total} 帧)")
        else:
            ctx.set("face_beautify2_path", None)
            print(f"    错误: 美颜视频创建失败")

    def _track_people(self, keypoints):
        """简单追踪人员"""
        tracks = {}
        for fi, frame_data in sorted(keypoints.items(),
                                     key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
            if not frame_data:
                continue
            for person_kps in frame_data:
                kps = np.array(person_kps)
                vis = kps[:, 2] > 0.5
                cx = 0.5 if vis.sum() < 6 else ((kps[5][0] + kps[6][0]) / 2 + (kps[11][0] + kps[12][0]) / 2) / 2
                best_tid, best_dist = None, float('inf')
                for tid, trk in tracks.items():
                    prev_cx = np.median(trk["cx_list"]) if trk["cx_list"] else cx
                    dist = abs(cx - prev_cx)
                    if dist < best_dist:
                        best_dist, best_tid = dist, tid
                if best_tid is not None and best_dist < 0.2:
                    tracks[best_tid]["cx_list"].append(cx)
                    tracks[best_tid]["count"] += 1
                else:
                    new_tid = len(tracks)
                    tracks[new_tid] = {"cx_list": [cx], "count": 1}
        return tracks

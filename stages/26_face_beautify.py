"""阶段26: 眼部提亮 + 面部美颜 (MediaPipe FaceMesh 版)

基于 MediaPipe FaceMesh 精确检测 478 个人脸关键点，
包括瞳孔、眼角、眉毛、嘴唇等细部位置。
只针对领操人面部处理，不影响背景和其他人。

用法（在配置中）:
    face_beautify:
      enabled: true
      eye_brighten: 0.5      # 眼部提亮强度 0~1
      face_smooth: 0.35      # 面部额外磨皮强度 0~1
      face_fill_light: 0.2   # 正面补光强度 0~1
      eye_radius: 22         # 眼部提亮区域半径(px)
"""

import cv2
import numpy as np
from pathlib import Path
import json
import shutil
import ctypes
import tempfile
import subprocess

from lib.utils import path_exists
from lib.face_mesh import FaceMeshDetector


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


class FaceBeautifyStage:
    def run(self, ctx):
        cfg = ctx.config.get("face_beautify", {})
        if not cfg.get("enabled", False):
            ctx.set("face_beautify_path", ctx.get("energybar_path") or ctx.get("beatflash_path") or ctx.get("ken_burns_path"))
            return

        eye_brighten = cfg.get("eye_brighten", 0.4)
        face_smooth = cfg.get("face_smooth", 0.3)
        eye_radius = cfg.get("eye_radius", 20)
        face_fill_light = cfg.get("face_fill_light", 0.15)
        if eye_brighten <= 0 and face_smooth <= 0 and face_fill_light <= 0:
            ctx.set("face_beautify_path", ctx.get("energybar_path") or ctx.get("beatflash_path") or ctx.get("ken_burns_path"))
            return

        # face_beautify 接在 energybar 之后，保留能量条
        input_path = (ctx.get("energybar_path") or
                      ctx.get("beatflash_path") or
                      ctx.get("ken_burns_path") or
                      ctx.get("color_path") or
                      str(ctx.input_path))
        if not path_exists(input_path):
            print("    跳过: 无输入视频")
            ctx.set("face_beautify_path", None)
            return

        # 加载领操人追踪信息（来自 pose_detect）
        lead_tid = ctx.get("lead_tid")
        lead_cx = ctx.get("lead_cx")
        if lead_tid is None:
            raw_kp = ctx.get("cropped_keypoints")
            if raw_kp:
                keypoints = raw_kp
            else:
                kp_path = ctx.output_dir / f"{ctx.input_path.stem}_cropped_keypoints.json"
                if not kp_path.exists() or kp_path.stat().st_size == 0:
                    kp_path2 = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
                    if not kp_path2.exists() or kp_path2.stat().st_size == 0:
                        print("    跳过: 无关键点数据")
                        ctx.set("face_beautify_path", None)
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
                ctx.set("face_beautify_path", None)
                return
            lead_tid = max(tracks, key=lambda tid: tracks[tid]["count"])
            lead_cx = np.median(tracks[lead_tid]["cx_list"])
            ctx.set("lead_tid", lead_tid)
            ctx.set("lead_cx", lead_cx)

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"    跳过: 无法打开视频 {input_path}")
            ctx.set("face_beautify_path", None)
            return
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"    美颜: 眼部提亮={eye_brighten}, 面部磨皮={face_smooth}, 补光={face_fill_light}, eye_radius={eye_radius}")

        ffmpeg_bin = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        # 临时目录建在 output_dir 中，避免系统临时目录短路径问题
        tmpdir = ctx.output_dir / f"_tmp_fb_{Path(input_path).stem}_{frame_idx:08d}"
        tmpdir.mkdir(exist_ok=True)
        tmpdir_short = _to_short(str(tmpdir))

        # 创建 FaceMesh 检测器（refine_landmarks=True 以获取瞳孔位置）
        face_mesh = FaceMeshDetector(refine_landmarks=True)
        face_mesh_tracker = self._create_tracker()

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0
        prev_kps = None
        no_face_count = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # 从关键点找领操人位置（用于判断是否处理当前帧）
            frame_kps = keypoints.get(str(frame_idx)) if 'keypoints' in dir() else None
            curr_kps = None
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

            # 如果领操人在画面中，用 FaceMesh 检测精确面部位置
            if curr_kps and prev_kps:
                # 调用 MediaPipe FaceMesh 获取精确眼部和面部 landmarks
                fm_result = face_mesh_tracker.process_frame(frame)

                if fm_result:
                    no_face_count = 0
                    eye_dist = fm_result['eye_distance']

                    # ========== 眼部提亮（瞳孔级精确） ==========
                    if eye_brighten > 0:
                        for pupil_key in ['left_pupil', 'right_pupil']:
                            px, py = fm_result[pupil_key]
                            if 0 <= px < w and 0 <= py < h:
                                mask = np.zeros((h, w), dtype=np.float32)
                                cv2.circle(mask, (px, py), eye_radius, 1.0, -1)
                                mask = cv2.GaussianBlur(mask, (eye_radius * 2 + 1, eye_radius * 2 + 1), eye_radius * 0.5)
                                mask = np.clip(mask * eye_brighten * 2.5, 0, 0.85)

                                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                                hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * (1 + mask), 0, 255).astype(np.uint8)
                                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                        # 眼角提亮：眼睑边缘
                        l_eye = fm_result['left_eye']
                        r_eye = fm_result['right_eye']
                        corner_mask = np.zeros((h, w), dtype=np.float32)
                        # 左眼角
                        cv2.circle(corner_mask, l_eye, int(eye_radius * 0.8), 0.6, -1)
                        # 右眼角
                        cv2.circle(corner_mask, r_eye, int(eye_radius * 0.8), 0.6, -1)
                        corner_mask = cv2.GaussianBlur(corner_mask, (int(eye_radius * 1.6 + 1),) * 2, eye_radius * 0.3)
                        corner_mask *= eye_brighten * 0.5
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * (1 + corner_mask), 0, 255).astype(np.uint8)
                        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    # ========== 面部区域磨皮 ==========
                    if face_smooth > 0:
                        l_eye = fm_result['left_eye']
                        r_eye = fm_result['right_eye']
                        face_cx = fm_result['face_center'][0]
                        face_cy = fm_result['face_center'][1]

                        # 面部区域：以两眼中心和为圆心，半径=眼距×2.5
                        face_r = int(eye_dist * 2.5)
                        face_mask = np.zeros((h, w), dtype=np.float32)
                        cv2.circle(face_mask, (face_cx, face_cy), face_r, 1.0, -1)
                        face_mask = cv2.GaussianBlur(face_mask, (face_r * 2 + 1, face_r * 2 + 1), face_r * 0.4)

                        smooth_img = cv2.bilateralFilter(frame, 7, 15, 15)
                        strength_map = (face_mask * face_smooth).astype(np.float32)
                        frame = (frame * (1 - strength_map) + smooth_img * strength_map).astype(np.uint8)

                    # ========== 正面补光（FaceMesh 精确定位） ==========
                    if face_fill_light > 0:
                        l_eye = fm_result['left_eye']
                        r_eye = fm_result['right_eye']
                        nose = fm_result['nose_tip']
                        face_cx = fm_result['face_center'][0]
                        face_cy = fm_result['face_center'][1]
                        eye_dist = fm_result['eye_distance']

                        # 面部区域：以 face_center 为中心
                        face_r = int(eye_dist * 2.2)
                        fill_mask = np.zeros((h, w), dtype=np.float32)
                        cv2.circle(fill_mask, (face_cx, face_cy), face_r, 1.0, -1)
                        fill_mask = cv2.GaussianBlur(fill_mask, (face_r * 2 + 1, face_r * 2 + 1), face_r * 0.5)

                        brightness = face_fill_light * 40
                        frame_f = frame.astype(np.float32)
                        frame_f = np.clip(frame_f + fill_mask[:, :, None] * brightness, 0, 255)
                        frame = frame_f.astype(np.uint8)

                else:
                    no_face_count += 1
                    if no_face_count > 30:
                        # 连续多帧未检测到人脸，跳过当前帧（保持能量条）
                        pass

            prev_kps = curr_kps

            fname = f"{tmpdir_short}/f_{frame_idx:06d}.png"
            cv2.imwrite(fname, frame)
            frame_idx += 1

            if frame_idx % 500 == 0:
                print(f"    进度: {frame_idx}/{max_frames}")

        cap.release()
        print(f"    写入完成: {frame_idx} 帧, 调用 FFmpeg 编码...")

        temp_out = ctx.output_dir / f"{ctx.input_path.stem}_face_beautify.mp4"
        output_short = _to_short(str(temp_out))
        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(fps),
            "-i", f"{tmpdir_short}/f_%06d.png",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an", output_short
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"    FFmpeg 错误: {result.stderr[-300:]}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            ctx.set("face_beautify_path", None)
            return

        shutil.rmtree(tmpdir, ignore_errors=True)

        if cv2.VideoCapture(str(temp_out)).isOpened():
            ctx.set("face_beautify_path", str(temp_out))
            print(f"    输出: {temp_out.name} ({frame_idx} 帧)")
        else:
            ctx.set("face_beautify_path", None)
            print(f"    错误: 美颜视频创建失败")

    def _create_tracker(self):
        """创建 FaceMesh 追踪器（每帧检测，不用 tracking）"""
        return FaceMeshDetector(refine_landmarks=True)

    def _track_people(self, keypoints):
        """简单追踪人员，返回 {tid: {cx_list, count}}"""
        tracks = {}
        for fi, frame_data in sorted(keypoints.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
            if not frame_data:
                continue
            for person_kps in frame_data:
                kps = np.array(person_kps)
                vis = kps[:, 2] > 0.5
                if vis.sum() < 6:
                    cx = 0.5
                else:
                    shoulders_cx = (kps[5][0] + kps[6][0]) / 2
                    hips_cx = (kps[11][0] + kps[12][0]) / 2
                    cx = (shoulders_cx + hips_cx) / 2

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
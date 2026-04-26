"""阶段26: 眼部提亮 + 面部美颜

基于关键点检测的眼部区域提亮 + 面部区域增强磨皮 + 正面补光。
只针对领操人面部处理，不影响背景和其他人。

用法（在配置中）:
    face_beautify:
      enabled: true
      eye_brighten: 0.4      # 眼部提亮强度 0~1
      face_smooth: 0.3       # 面部额外磨皮强度 0~1
      eye_radius: 20        # 眼部提亮区域半径(px)
      face_fill_light: 0.15 # 正面补光强度 0~1（照亮全脸阴影）
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
            ctx.set("face_beautify_path", ctx.get("beatflash_path") or ctx.get("energybar_path") or ctx.get("ken_burns_path"))
            return

        eye_brighten = cfg.get("eye_brighten", 0.4)
        face_smooth = cfg.get("face_smooth", 0.3)
        eye_radius = cfg.get("eye_radius", 20)
        face_fill_light = cfg.get("face_fill_light", 0.15)  # 正面补光强度
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

        # 加载关键点
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

        # 领操人追踪
        lead_tid = ctx.get("lead_tid")
        lead_cx = ctx.get("lead_cx")
        if lead_tid is None:
            tracks = self._track_people(keypoints)
            if not tracks:
                print("    跳过: 无法追踪人员")
                ctx.set("face_beautify_path", None)
                return
            lead_tid = max(tracks, key=lambda tid: tracks[tid]["count"])
            lead_cx = np.median(tracks[lead_tid]["cx_list"])
            ctx.set("lead_tid", lead_tid)
            ctx.set("lead_cx", lead_cx)

        print(f"    美颜: 眼部提亮={eye_brighten}, 面部磨皮={face_smooth}, 补光={face_fill_light}, eye_radius={eye_radius}")

        ffmpeg_bin = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        tmpdir = Path(tempfile.mkdtemp(prefix="fb_"))
        tmpdir_short = _to_short(str(tmpdir))

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0
        prev_kps = None
        lead_kps_cache = {}

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            frame_kps = keypoints.get(str(frame_idx))
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

            if curr_kps and prev_kps:
                kps_arr = np.array(curr_kps)
                prev_arr = np.array(prev_kps)
                vis_mask = (kps_arr[:, 2] > 0.5) & (prev_arr[:, 2] > 0.5)

                # 眼部提亮 (COCO: 1=左眼, 2=右眼, 0=鼻)
                if eye_brighten > 0:
                    for eye_idx in [1, 2]:  # COCO: 1=left_eye, 2=right_eye
                        if kps_arr[eye_idx][2] > 0.4:
                            ex = int(kps_arr[eye_idx][0] * w)
                            ey = int(kps_arr[eye_idx][1] * h)
                            # 高光圈：中心最亮，向外渐暗
                            mask = np.zeros((h, w), dtype=np.float32)
                            cv2.circle(mask, (ex, ey), eye_radius, 1.0, -1)
                            mask = cv2.GaussianBlur(mask, (eye_radius * 2 + 1, eye_radius * 2 + 1), eye_radius * 0.5)
                            mask = np.clip(mask * eye_brighten * 2, 0, 0.8)

                            # 提亮：addWeighted 在 V 通道上操作更自然
                            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                            hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * (1 + mask), 0, 255).astype(np.uint8)
                            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # 面部区域磨皮：仅对以两眼中心为圆心的区域磨皮
                if face_smooth > 0 and kps_arr[1][2] > 0.4 and kps_arr[2][2] > 0.4:
                    lx = int(kps_arr[0][0] * w)
                    ly = int(kps_arr[0][1] * h)
                    rx = int(kps_arr[1][0] * w)
                    ry = int(kps_arr[1][1] * h)
                    face_cx = (lx + rx) // 2
                    face_cy = (ly + ry) // 2
                    eye_dist = max(1, int(np.sqrt((rx - lx)**2 + (ry - ly)**2)))
                    face_r = int(eye_dist * 2.2)

                    face_mask = np.zeros((h, w), dtype=np.float32)
                    cv2.circle(face_mask, (face_cx, face_cy), face_r, 1.0, -1)
                    face_mask = cv2.GaussianBlur(face_mask, (face_r * 2 + 1, face_r * 2 + 1), face_r * 0.4)

                    # 双边滤波 + 按 mask 加权
                    smooth_img = cv2.bilateralFilter(frame, 7, 15, 15)
                    strength_map = (face_mask * face_smooth).astype(np.float32)
                    frame = (frame * (1 - strength_map) + smooth_img * strength_map).astype(np.uint8)

            # 正面补光：打亮全脸阴影（模拟柔光箱效果）
            if face_fill_light > 0 and curr_kps:
                kps_arr = np.array(curr_kps)
                # 用鼻尖(0)和两眼中心作为面部中心
                if kps_arr[1][2] > 0.3 and kps_arr[2][2] > 0.3:
                    lx = int(kps_arr[1][0] * w)
                    ly = int(kps_arr[1][1] * h)
                    rx = int(kps_arr[2][0] * w)
                    ry = int(kps_arr[2][1] * h)
                    face_cx = (lx + rx) // 2
                    face_cy = (ly + ry) // 2
                    # 鼻子位置做参考，估算面部大小
                    nose_x = int(kps_arr[0][0] * w)
                    nose_y = int(kps_arr[0][1] * h)
                    eye_dist = max(1, int(np.sqrt((rx - lx)**2 + (ry - ly)**2)))
                    face_r = int(eye_dist * 2.0)  # 覆盖整张脸

                    fill_mask = np.zeros((h, w), dtype=np.float32)
                    cv2.circle(fill_mask, (face_cx, face_cy), face_r, 1.0, -1)
                    fill_mask = cv2.GaussianBlur(fill_mask, (face_r * 2 + 1, face_r * 2 + 1), face_r * 0.5)

                    # 正面补光：在 BGR 三个通道都加亮，模拟柔光
                    brightness = face_fill_light * 40  # 最多加40亮度
                    frame_f = frame.astype(np.float32)
                    frame_f = np.clip(frame_f + fill_mask[:, :, None] * brightness, 0, 255)
                    frame = frame_f.astype(np.uint8)

            fname = f"{tmpdir_short}/f_{frame_idx:06d}.png"
            cv2.imwrite(fname, frame)
            prev_kps = curr_kps
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
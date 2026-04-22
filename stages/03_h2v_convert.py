"""阶段03: 横转竖

策略:
- 用身体中心的水平位置识别不同人（位置不随姿势变化）
- 领操人: 综合评分（肩宽×身高×帧数）最高的人
- 领操人单独出现时 → 9:16 特写
- 其他人单独出现或多人 → 3:4 整体场景
"""

import cv2
import numpy as np
import subprocess
import shutil
import json
from pathlib import Path


class H2VConvertStage:
    def run(self, ctx):
        # 增量跳过：h2v 视频已存在则跳过（关键点文件单独保存，下方会加载或生成）
        if ctx.get("h2v_path") and Path(ctx.get("h2v_path")).exists():
            # 尝试加载已保存的关键点
            kp_file = Path(ctx.output_dir) / f"{Path(ctx.input_path).stem}_cropped_keypoints.json"
            if kp_file.exists():
                with open(kp_file) as f:
                    ctx.data["cropped_keypoints"] = json.load(f)
            # 更新 video_info（避免 preview 残留 process_frames 影响后续阶段）
            h2v_path = ctx.get("h2v_path")
            cap = cv2.VideoCapture(h2v_path)
            if cap.isOpened():
                vi = ctx.data.get("video_info", {})
                vi["fps"] = cap.get(cv2.CAP_PROP_FPS)
                vi["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vi["process_frames"] = vi["frames"]
                ctx.data["video_info"] = vi
                cap.release()
            print("    已存在，跳过")
            return

        video_path = Path(ctx.get("stabilized_path")) if ctx.get("stabilized_path") else ctx.input_path
        keypoints = ctx.get("keypoints")
        video_info = ctx.get("video_info")

        if not keypoints:
            raise ValueError("未检测到姿态关键点，请先运行 pose_detect 阶段")

        orig_w = video_info["width"]
        orig_h = video_info["height"]
        fps = video_info["fps"]
        total_frames = video_info["frames"]

        target_ratio = 9.0 / 16.0  # 9:16  portrait

        target_ratio = 9.0 / 16.0  # 9:16  portrait

        # 输出尺寸: 使用最终分辨率 1080x1920
        out_w = 1080
        out_h = 1920

        # 9:16 裁剪参数
        is_portrait = orig_h > orig_w
        if is_portrait:
            crop9_h = out_h
            crop9_w = orig_w
            crop9_x = 0
            crop9_y = 0
        else:
            # 横屏转竖屏：取原视频上部 70%，去掉地面区域
            # 人物在画面中上部，底部区域被排除
            crop9_h = int(orig_h * 0.70)  # 只用上部70%，去掉脚底阴影区域
            crop9_h = crop9_h if crop9_h % 2 == 0 else crop9_h - 1
            crop9_w = int(crop9_h * target_ratio)
            crop9_w = crop9_w if crop9_w % 2 == 0 else crop9_w - 1
            crop9_x = (orig_w - crop9_w) // 2
            crop9_y = 0

        # 3:4 居中裁剪（轻微裁切，约94%宽度）
        crop3_h = orig_h
        crop3_w = int(crop3_h * 3.0 / 4.0)
        crop3_w = crop3_w if crop3_w % 2 == 0 else crop3_w - 1
        crop3_x = (orig_w - crop3_w) // 2

        # ========== Step 1: 追踪各人，综合评分确定领操人 ==========
        print(f"    身份追踪 ({total_frames} 帧)...")
        # 用位置+体型追踪：匹配时用身体中心x距离
        tracks = {}  # track_id -> {"cx_list": [], "body_size_list": [], "count": 0}

        for fi in range(total_frames):
            pose_data = keypoints.get(fi)
            if not pose_data:
                continue

            frame_detections = []
            for pi, person_kps in enumerate(pose_data):
                cx = self._body_center_x(person_kps)
                body_size = self._body_size_score(person_kps)
                if cx is None:
                    cx = 0.5
                frame_detections.append((pi, cx, body_size))

            # 简单最近邻匹配：分配到距离最近的 track
            assigned = set()
            for pi, cx, body_size in frame_detections:
                best_tid = None
                best_dist = float('inf')
                for tid, trk in tracks.items():
                    if tid in assigned:
                        continue
                    prev_cx = np.median(trk["cx_list"]) if trk["cx_list"] else cx
                    dist = abs(cx - prev_cx)
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                # 距离阈值：>0.2 认为是新人
                if best_tid is not None and best_dist < 0.2:
                    tracks[best_tid]["cx_list"].append(cx)
                    tracks[best_tid]["body_size_list"].append(body_size)
                    tracks[best_tid]["count"] += 1
                    assigned.add(best_tid)
                else:
                    # 新建 track
                    new_tid = len(tracks)
                    tracks[new_tid] = {
                        "cx_list": [cx],
                        "body_size_list": [body_size],
                        "count": 1,
                    }
                    assigned.add(new_tid)

        if not tracks:
            tracks = {0: {"cx_list": [0.5], "body_size_list": [1.0], "count": total_frames}}

        # 综合评分: 帧数 × 平均肩宽×身高 (领操人通常在画面中央且体型大)
        lead_tid = max(tracks, key=lambda tid: self._lead_score(tracks[tid]))
        lead_cx = np.median(tracks[lead_tid]["cx_list"])
        lead_size = np.median(tracks[lead_tid]["body_size_list"])
        print(f"    身份数: {len(tracks)}, 领操人: tid={lead_tid}, "
              f"x_center={lead_cx:.3f}, body_size={lead_size:.3f}, "
              f"帧数={tracks[lead_tid]['count']}")

        # ========== Step 2: 逐帧决定场景类型 ==========
        # 策略: 完全用人数判断，不用体型
        # 人数 >= 3 → 全景（多人场景）
        # 人数 <= 2 → 领操人特写（单人/双人close-up）
        lead_sizes = [s for s in tracks[lead_tid]["body_size_list"] if s > 0]
        lead_size_median = np.median(lead_sizes) if lead_sizes else 1.0
        PANO_PERSON_THRESHOLD = 3
        print(f"    逐帧判断（全景人数>={PANO_PERSON_THRESHOLD}）...")
        frame_decisions = []
        lead_frames = other_frames = multi_frames = 0

        for fi in range(total_frames):
            pose_data = keypoints.get(fi)
            num = len(pose_data) if pose_data else 0

            if num == 0:
                decision = "other"
            elif num >= PANO_PERSON_THRESHOLD:
                decision = "multi"
            else:
                decision = "lead"

            frame_decisions.append(decision)
            if decision == "lead":
                lead_frames += 1
            elif decision == "other":
                other_frames += 1
            else:
                multi_frames += 1

        print(f"    场景: 领操人特写={lead_frames}帧, "
              f"其他人员={other_frames}帧, 整体={multi_frames}帧")

        # ========== Step 3: 分段 ==========
        MIN_SEG = max(int(fps * 0.8), 15)  # 降低到0.8秒或15帧，保留更多短全景
        segments = self._merge_segments(frame_decisions, MIN_SEG)
        print(f"    分段: {len(segments)}, 最短{MIN_SEG}帧")
        total_seg_frames = sum(end_f - start_f + 1 for start_f, end_f, _ in segments)
        print(f"    段内帧数合计: {total_seg_frames} / {total_frames}")
        for i, (start_f, end_f, dtype) in enumerate(segments):
            dur = (end_f - start_f + 1) / fps
            print(f"    段{i}: 帧{start_f}-{end_f}, 类型={dtype}, 时长={dur:.1f}秒")

        # ========== Step 4: FFmpeg 分段裁剪 + concat ==========
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"

        seg_info = []
        seg_files = []

        for i, (start_f, end_f, dtype) in enumerate(segments):
            duration = end_f - start_f + 1
            tmp_path = (ctx.output_dir / f"seg_{i}.mp4").resolve()
            seg_files.append(tmp_path)

            if dtype == "lead":
                # 以领操人位置为中心的 9:16 裁剪
                cx_list = []
                for fi in range(start_f, end_f + 1):
                    pose_data = keypoints.get(fi)
                    if pose_data:
                        for pi, person_kps in enumerate(pose_data):
                            cx = self._body_center_x(person_kps)
                            if cx is not None and abs(cx - lead_cx) < 0.15:
                                cx_list.append(cx)
                seg_lead_cx = np.median(cx_list) if cx_list else lead_cx
                crop9_x = int(seg_lead_cx * orig_w - crop9_w / 2)
                crop9_x = max(0, min(crop9_x, orig_w - crop9_w))
                crop9_x = crop9_x if crop9_x % 2 == 0 else crop9_x - 1
                seg_crop_x = crop9_x
                seg_crop_w = crop9_w
                if is_portrait:
                    # 竖屏：直接scale到目标尺寸（不需要crop）
                    vf = f"scale={out_w}:{out_h}"
                else:
                    vf = f"crop={crop9_w}:{crop9_h}:{crop9_x}:0,scale={out_w}:{out_h}"
            else:
                # 全景：裁掉约15%宽度（左7.5%+右7.5%），然后scale填满9:16
                # 这样内容幅面约50%，同时保留大部分原始场景
                seg_crop_x = 0
                seg_crop_w = orig_w
                if is_portrait:
                    vf = f"scale={out_w}:{out_h}"
                else:
                    wide_crop_w = int(orig_w * 0.95)
                    wide_crop_w = wide_crop_w if wide_crop_w % 2 == 0 else wide_crop_w - 1
                    wide_crop_x = (orig_w - wide_crop_w) // 2
                    seg_crop_x = wide_crop_x
                    seg_crop_w = wide_crop_w
                    vf = f"crop={wide_crop_w}:{orig_h}:{wide_crop_x}:0,scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"

            start_sec = start_f / fps
            dur_sec = duration / fps

            cmd = [
                ffmpeg, "-y",
                "-ss", str(start_sec),
                "-i", str(video_path),
                "-t", str(dur_sec),
                "-vf", vf,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-an",
                str(tmp_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
            if result.returncode != 0:
                print(f"    段{i} FFmpeg失败: {result.stderr[-150:]}")
                self._create_black(tmp_path, out_w, out_h, fps, duration)
            seg_info.append((str(tmp_path), duration, seg_crop_x, seg_crop_w, dtype))

        # ========== concat: 标准化所有片段尺寸后合并 ==========
        # 验证每个片段的尺寸，不一致的重新编码到目标尺寸
        std_seg_files = []
        for i, (seg_path, dur, cx, cw, dtype) in enumerate(seg_info):
            cap = cv2.VideoCapture(str(seg_path))
            sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if sw != out_w or sh != out_h:
                if dtype == "multi":
                    # 全景片段：已经是letterbox格式，保持不变
                    std_seg_files.append(seg_path)
                else:
                    # 尺寸不匹配，重新编码
                    std_path = ctx.output_dir / f"_std_{i}.mp4"
                    cmd = [ffmpeg, "-y", "-i", str(seg_path),
                           "-vf", f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2",
                           "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", str(std_path)]
                    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                    if r.returncode != 0:
                        print(f"    段{i} 重新编码失败，使用原片段")
                        std_path = seg_path
                    std_seg_files.append(std_path)
            else:
                std_seg_files.append(seg_path)

        concat_list = (ctx.output_dir / "concat.txt").resolve()
        with open(concat_list, "w", encoding="utf-8", newline="\n") as f:
            for p in std_seg_files:
                f.write(f"file '{Path(p).as_posix()}'\n")

        temp_path = (ctx.output_dir / f"{video_path.stem}_h2v.mp4").resolve()
        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",
            str(temp_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"    concat失败: {result.stderr[-200:]}")
            raise RuntimeError(f"concat失败")

        concat_list.unlink()
        for f in seg_files:
            if f.exists():
                f.unlink()
        # 清理标准化片段
        for f in ctx.output_dir.glob("_std_*.mp4"):
            f.unlink()

        print(f"    输出: {temp_path.name} ({out_w}x{out_h})")
        # 验证输出帧数
        cap_verify = cv2.VideoCapture(str(temp_path))
        actual_out_frames = int(cap_verify.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_verify.release()
        expected_frames = sum(end_f - start_f + 1 for start_f, end_f, _ in segments)
        print(f"    输出帧数验证: cv2读出{actual_out_frames}帧 vs 预期{expected_frames}")
        # 用 ffprobe 双重验证
        ffprobe = shutil.which("ffprobe") or "C:/Users/18091/ffmpeg/ffprobe.exe"
        r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                           "-show_entries", "stream=duration,nb_frames",
                           "-of", "csv=p=0", str(temp_path)],
                          capture_output=True, text=True)
        if r.returncode == 0:
            print(f"    ffprobe验证: {r.stdout.strip()}")

        # ========== Step 5: 保存关键点（使用每段的实际裁剪偏移） ==========
        cropped_keypoints = {}
        for i, (start_f, end_f, dtype) in enumerate(segments):
            _, _, seg_crop_x, seg_crop_w, _ = seg_info[i]
            for fi in range(start_f, end_f + 1):
                pose_data = keypoints.get(fi)
                if pose_data:
                    kps_list = []
                    for person_kps in pose_data:
                        ckps = [
                            [(person_kps[k][0] * orig_w - seg_crop_x) / seg_crop_w,
                             person_kps[k][1],
                             person_kps[k][2]]
                            for k in range(len(person_kps))
                        ]
                        kps_list.append(ckps)
                    cropped_keypoints[fi] = kps_list
                else:
                    cropped_keypoints[fi] = None

        ctx.set("cropped_keypoints", cropped_keypoints)
        ctx.set("h2v_path", str(temp_path))
        ctx.set("h2v_size", (out_w, out_h))

        # 保存到 JSON 文件供后续增量使用
        kp_file = Path(ctx.output_dir) / f"{Path(ctx.input_path).stem}_cropped_keypoints.json"
        with open(kp_file, "w") as f:
            json.dump(cropped_keypoints, f)
        print(f"    关键点已保存: {kp_file.name}")

    def _body_center_x(self, person_kps):
        """计算人体水平中心（归一化）"""
        kps = np.array(person_kps)
        vis = kps[:, 2] > 0.5
        if vis.sum() < 6:
            return None
        shoulders_cx = (kps[11][0] + kps[12][0]) / 2
        hips_cx = (kps[23][0] + kps[24][0]) / 2
        return (shoulders_cx + hips_cx) / 2

    def _body_size_score(self, person_kps):
        """计算人体大小评分（肩宽×身高）"""
        kps = np.array(person_kps)
        vis = kps[:, 2] > 0.5
        if vis.sum() < 8:
            return 0.0
        left_shoulder = kps[11]
        right_shoulder = kps[12]
        nose = kps[0]
        left_ankle = kps[27]
        right_ankle = kps[28]
        shoulder_w = abs(right_shoulder[0] - left_shoulder[0])
        body_h = abs((left_ankle[1] + right_ankle[1]) / 2 - nose[1])
        return shoulder_w * body_h

    def _lead_score(self, track):
        """领操人综合评分: 帧数 × 平均体型大小"""
        frame_count = track["count"]
        avg_size = np.mean(track["body_size_list"]) if track["body_size_list"] else 0.0
        # 领操人通常出现次数多且体型大（离镜头近）
        return frame_count * (avg_size ** 0.5)

    def _merge_segments(self, decisions, min_frames):
        if not decisions:
            return []
        raw = []
        start = 0
        cur = decisions[0]
        for i in range(1, len(decisions)):
            if decisions[i] == cur:
                continue
            raw.append((start, i - 1, cur))
            start = i
            cur = decisions[i]
        raw.append((start, len(decisions) - 1, cur))

        merged = [raw[0]] if raw else []
        for seg in raw[1:]:
            last = merged[-1]
            seg_len = seg[1] - seg[0] + 1
            if seg_len < min_frames:
                merged[-1] = (last[0], seg[1], last[2])
            else:
                merged.append(seg)
        return merged

    def _create_black(self, path, w, h, fps, num_frames):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                fps, (w, h))
        for _ in range(num_frames):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
        writer.release()

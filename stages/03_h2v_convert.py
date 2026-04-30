"""阶段03: 横转竖

策略:
- 用身体中心的水平位置识别不同人（位置不随姿势变化）
- 领操人: 综合评分（肩宽×身高×帧数）最高的人
- 领操人单独出现时 → 9:16 特写
- 其他人单独出现或多人 → 3:4 整体场景
"""

import cv2
from lib.utils import path_exists
from lib.crop_strategy import (
    build_tracks, select_lead_track, classify_frames,
    merge_segments, get_lead_center_in_segment,
    _body_center_x, _body_size_score,
)
import numpy as np
import subprocess
import shutil
import json
from pathlib import Path


class H2VConvertStage:
    def run(self, ctx):
        # 增量跳过：h2v 视频已存在则跳过（关键点文件单独保存，下方会加载或生成）
        if ctx.get("h2v_path") and path_exists(ctx.get("h2v_path")):
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
        tracks = build_tracks(keypoints, total_frames)
        lead_tid, lead_track = select_lead_track(tracks)
        lead_cx = np.median(lead_track["cx_list"])
        lead_size = np.median(lead_track["body_size_list"])
        print(f"    身份数: {len(tracks)}, 领操人: tid={lead_tid}, "
              f"x_center={lead_cx:.3f}, body_size={lead_size:.3f}, "
              f"帧数={lead_track['count']}")

        # ========== Step 2: 逐帧决定场景类型 ==========
        PANO_PERSON_THRESHOLD = 3
        print(f"    逐帧判断（全景人数>={PANO_PERSON_THRESHOLD}）...")
        frame_decisions, stats = classify_frames(keypoints, total_frames, PANO_PERSON_THRESHOLD)
        print(f"    场景: 领操人特写={stats['lead']}帧, "
              f"其他人员={stats['other']}帧, 整体={stats['multi']}帧")

        # ========== Step 3: 分段 ==========
        MIN_SEG = max(int(fps * 0.8), 15)
        segments = merge_segments(frame_decisions, MIN_SEG)
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
                seg_lead_cx = get_lead_center_in_segment(
                    keypoints, segments, lead_tid, lead_cx,
                    start_f, end_f, orig_w, crop9_w)
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
                "-c:v", "libx264", "-preset", "fast", "-crf", "1",
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
                           "-c:v", "libx264", "-preset", "fast", "-crf", "1", "-an", str(std_path)]
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
            "-c:v", "libx264", "-preset", "fast", "-crf", "1",
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
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
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

    def _create_black(self, path, w, h, fps, num_frames):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                fps, (w, h))
        for _ in range(num_frames):
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
        writer.release()

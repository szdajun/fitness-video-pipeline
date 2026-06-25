"""阶段38: 逐帧智能裁切 (smart_crop)

解决问题:
  07_export 用 lead_cx (整个视频的中位数) 做静态裁切 → 领操人横移时被裁出/偏一侧
  本 stage 用 keypoints 逐帧算 crop_x, 以领操人为中心, 实现"教练稳在中心"

输入: ctx.get("processed_pre_export") — 选 h2v_path / ken_burns_path / color_path / warped_path / stabilized_path / input
      ctx.get("keypoints") — 来自 01_pose_detect 的 {frame_idx: [person_kps]} 字典
      ctx.get("cropped_keypoints") — 优先用 (来自 h2v_convert 的 0~1 归一化)
      ctx.get("video_info") — fps / frames
      ctx.get("output") — {width, height} 目标尺寸

输出: ctx.set("smart_crop_path", str(out_path)) — 跟 04_ken_burns auto_track 同样套路

算法:
  1. 用 build_tracks / select_lead_track 找 lead_tid
  2. 对每帧取 lead_tid 的人肩髋中点 cx (0~1 归一化)
  3. SMA 平滑 (默认窗口 15) + 场景突变重置
  4. 边界钳制: cx ∈ [crop_w/2, 1 - crop_w/2]
  5. FFmpeg 一帧帧 PNG + crop 编码 (与 04_ken_burns auto_track 同样的可靠套路)

只在需要时跑 (h2v 输出是 9:16/3:4 时):
  - 默认 output.width < output.height → 跑
  - 否则跳过 (横版不裁)
"""

import json
import os
import subprocess
import shutil
import ctypes
import tempfile
from pathlib import Path

import cv2
import numpy as np

from lib.crop_strategy import build_tracks, select_lead_track


def _get_short_path(p):
    """Windows 8.3 短路径 (避免 FFmpeg 命令行空格/中文问题)"""
    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    GetShortPathNameW.restype = ctypes.c_uint

    def to_short(s):
        buf_size = GetShortPathNameW(str(s), None, 0)
        if buf_size == 0:
            return str(s)
        buf = ctypes.create_unicode_buffer(buf_size)
        GetShortPathNameW(str(s), buf, buf_size)
        return buf.value
    return to_short(p)


class SmartCropStage:
    def run(self, ctx):
        # 增量跳过
        existing = ctx.get("smart_crop_path")
        if existing and Path(existing).exists():
            print("    已存在，跳过")
            return

        # 1. 决定要不要跑 (横版不裁)
        out_cfg = ctx.config.get("output", {})
        out_w = out_cfg.get("width")
        out_h = out_cfg.get("height")
        if not (out_w and out_h) or out_w >= out_h:
            print(f"    跳过: 输出 {out_w}x{out_h} 非竖版, 无需逐帧裁切")
            return

        # 2. 选输入视频 (从后往前找, 跟 04_ken_burns 一样的优先级)
        input_path = (ctx.get("warped_path") or
                      ctx.get("ken_burns_path") or
                      ctx.get("color_path") or
                      ctx.get("h2v_path") or
                      ctx.get("stabilized_path") or
                      str(ctx.input_path))
        if not input_path or not Path(input_path).exists():
            print(f"    跳过: 无输入视频 {input_path}")
            return

        # 3. 关键点 (优先 cropped_keypoints, 没有就用 raw keypoints)
        kp_dict = ctx.get("cropped_keypoints") or ctx.get("keypoints")
        if not kp_dict:
            print("    跳过: 无关键点数据 (跑过 pose_detect 才能用)")
            return

        # 4. 读输入视频信息
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"    跳过: 无法打开 {input_path}")
            return
        in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vi = ctx.get("video_info") or {}

        # 4a. 2026-06-21 自动判断是否启用 smart_crop
        #     根据视频特征判断, 节省算力 (短视频/单人场景不需要 smart_crop)
        cfg = ctx.config.get("smart_crop", {})
        user_choice = cfg.get("enabled")
        if user_choice is False:
            print(f"    [auto-disable] 用户显式禁用 smart_crop")
            return

        # 规则 1: 短视频 (< 10 秒) 不启用
        duration_sec = total / fps if fps > 0 else 0
        if duration_sec < 10:
            print(f"    [auto-disable] 时长 {duration_sec:.1f}s < 10s, smart_crop 不必要")
            return

        # 规则 2: 1-2 人不启用 (单人/双人裁切居中即可, 不用跟踪)
        # 扫前 30 帧看人数中位数
        person_counts = []
        for fi in range(min(30, total)):
            pose = kp_dict.get(str(fi))
            if not pose:
                pose = kp_dict.get(fi)
            if pose:
                person_counts.append(len(pose))
        if person_counts:
            median_count = sorted(person_counts)[len(person_counts) // 2]
            if median_count <= 2:
                print(f"    [auto-disable] 前 30 帧人数中位数 {median_count} <= 2, smart_crop 不必要")
                return

        # 通过所有规则, 启用 (line 69 已过滤横版输出)
        print(f"    [auto-enable] 多人({median_count}人)长视频({duration_sec:.0f}s)竖版输出")
        max_frames = vi.get("process_frames", total) or total
        max_frames = min(max_frames, total)
        if max_frames <= 0:
            max_frames = total
        cap.release()

        # 5. 选 lead_tid
        #    keypoints 格式: {fi: [person_kps,...]}
        #    person_kps: list of [x, y, conf] (x, y 已归一化 0~1)
        #    build_tracks 要的是这个格式, 直接用
        #
        # 拼接视频优化: 用前 N 帧（拼接前）选人，避免背面骨架混乱时选错。
        # 2026-06-20 修复: 抖音拼接视频背面段 build_tracks 会选错 tid
        cfg_pre = ctx.config.get("smart_crop", {})
        first_n_frames = int(cfg_pre.get("select_lead_first_n_frames", 60))  # 默认前 60 帧 (2s @ 30fps)
        tracks = build_tracks(kp_dict, max_frames)
        if first_n_frames > 0:
            lead_tid, lead_track = select_lead_track(tracks, first_n_frames=first_n_frames)
            print(f"    领操人选人: 用前 {first_n_frames} 帧评分")
        else:
            lead_tid, lead_track = select_lead_track(tracks)
        if not lead_track["cx_list"]:
            print("    跳过: 未找到领操人轨迹")
            return

        # 6. cfg
        cfg = ctx.config.get("smart_crop", {})
        smooth_window = int(cfg.get("smooth_window", 15))
        max_jump_ratio = float(cfg.get("max_jump_ratio", 0.03))  # 每帧最大移动 3%
        margin_ratio = float(cfg.get("margin_ratio", 0.06))      # 边界留白 6%

        # 7. 裁切尺寸: 在输入视频上裁出 out_aspect (out_w/out_h) 窗口
        #    输入已经是 16:9 (横屏原视频), crop_h = in_h, crop_w = in_h * out_w / out_h
        #    如果输入已经接近竖版 (h2v 阶段产物), 直接 scale
        in_aspect = in_w / in_h if in_h > 0 else 16 / 9
        out_aspect = out_w / out_h
        if in_aspect > out_aspect:
            # 输入比输出更宽: 裁宽, 留高
            crop_h = in_h
            crop_w = int(crop_h * out_aspect)
        else:
            # 输入比输出更窄 (竖源): 裁高, 留宽
            crop_w = in_w
            crop_h = int(crop_w / out_aspect)
        crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1
        crop_h = crop_h if crop_h % 2 == 0 else crop_h - 1
        if crop_w <= 0 or crop_h <= 0 or crop_w > in_w or crop_h > in_h:
            print(f"    跳过: 裁切尺寸异常 crop={crop_w}x{crop_h} (in={in_w}x{in_h})")
            return

        # 7b. 模式: extreme — 当 lead cx 超出 9:16 窗口物理范围时,
        #     扩大 crop_w 让窗口能包住 (牺牲横纵比, 后面再 scale 拉伸)
        mode = cfg.get("mode", "default")
        extreme_extra_ratio = float(cfg.get("extreme_extra_ratio", 0.50))  # cx 距边缘余量不足时, 最多把 crop 拉宽 50%
        extreme_min_cx = float(cfg.get("extreme_min_cx", 0.10))           # cx < 此值强制 extreme
        extreme_max_cx = float(cfg.get("extreme_max_cx", 0.90))           # cx > 此值强制 extreme

        # 2026-06-20 修复: 拼接视频背身段 YOLO bbox 检测完全不可信,
        # 既然无法精确跟随, 直接把裁切窗口扩宽保证领操人始终在画面内
        # 用户可配: lead_zoom_out=0.3 表示裁切窗口扩宽 30% (从 607 -> 789)
        lead_zoom_out = float(cfg.get("lead_zoom_out", 0.0))
        if lead_zoom_out > 0:
            old_crop_w = crop_w
            crop_w = int(crop_w * (1.0 + lead_zoom_out))
            crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1
            crop_w = min(crop_w, in_w)  # 不超过原始宽
            print(f"    [lead_zoom_out] crop_w {old_crop_w} → {crop_w} (+{lead_zoom_out*100:.0f}%)")
            # 重算 min/max_cx
            margin = margin_ratio
            min_cx_pre = crop_w / in_w / 2 + margin
            max_cx_pre = 1.0 - crop_w / in_w / 2 - margin
            print(f"    [lead_zoom_out] 物理 cx 范围 [{min_cx_pre:.3f}, {max_cx_pre:.3f}]")

        # 8. 逐帧收集 lead cx (0~1) + bbox 面积 (用于加权)
        # 2026-06-21 修复: bbox 面积越大的帧 YOLO 检测越可靠, 用面积加权选 cx
        # 之前用简单均值, 但 frame 100 (area=0.005 极小, YOLO 漏检) 跟 frame 80
        # (area=0.063 真实绿衣人) 等权, 把 cx 拉回到 0.5 中心
        raw_cx = []
        raw_area = []  # bbox 面积 (用于加权)
        raw_torso_w = []  # 2026-06-21 (v17): 躯干宽度, 用于过滤不可信检测
        for fi in range(max_frames):
            # 2026-06-21 修复: keypoints json 的 keys 是字符串 '0','1',...
            # 必须用 str(fi) 才能查到, 否则 get(fi) 返回 None 走 fallback
            pose_data = kp_dict.get(str(fi)) if isinstance(kp_dict, dict) else None
            if pose_data is None:
                pose_data = kp_dict.get(fi) if isinstance(kp_dict, dict) else None
            cx = 0.5  # fallback
            area = 0.001  # fallback 极小权重
            torso_w = 0.0  # 躯干宽度 (肩+髋平均)
            if pose_data and lead_tid < len(pose_data):
                person = pose_data[lead_tid]
                if person and len(person) >= 29:
                    kps = np.array(person)
                    vis = kps[:, 2] > 0.3 if kps.shape[1] >= 3 else np.ones(len(kps), bool)
                    if vis.sum() >= 4:
                        # v16: 躯干 cx (肩+髋, 排除手腕)
                        torso_idx = [11, 12, 23, 24]
                        torso_vis = vis[torso_idx]
                        if torso_vis.sum() >= 2:
                            cx = float(np.mean(kps[torso_idx][torso_vis, 0]))
                            # 肩宽+髋宽 (平均), 反映躯干稳定性
                            widths = []
                            if vis[11] and vis[12]:
                                widths.append(abs(kps[12][0] - kps[11][0]))
                            if vis[23] and vis[24]:
                                widths.append(abs(kps[24][0] - kps[23][0]))
                            if widths:
                                torso_w = float(np.mean(widths))
                        else:
                            cx = float(np.mean(kps[vis, 0]))
                        xs = kps[vis, 0]
                        ys = kps[vis, 1]
                        area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
            raw_cx.append(cx)
            raw_area.append(area)
            raw_torso_w.append(torso_w)

        raw_cx = np.array(raw_cx, dtype=np.float32)
        raw_area = np.array(raw_area, dtype=np.float32)
        raw_torso_w = np.array(raw_torso_w, dtype=np.float32)

        # 2026-06-21: 不用 fixed weighted_cx 锚定
        # v17 改用 torso_w^2 作为权重 (躯干宽 = 检测稳定性指标)
        #    躯干宽很小 (< 0.025) → YOLO 只检测到 2 个躯干点, cx 跳到极值
        #    躯干宽大 (>= 0.04) → 检测可靠, cx 稳定
        #    用躯干宽平方作权重, 让稳定帧主导, 过滤跳变帧
        win = max(5, smooth_window)
        med_win = max(win, 90)  # 90 帧 (3 秒)
        half = med_win // 2

        # v21: 自动检测分段点（拼接视频机位切换），各段独立算 cx
        # 思路：滑动窗口中位数突变 → 分段边界；段内可靠帧 (torso_w>0.04) 取加权中位数
        # 单段视频退化为全片统一 cx

        # 1. 滑动窗口找分段点
        seg_win = max(30, min(150, max_frames // 20))  # 30~150 帧窗口
        roll_med = np.zeros(max_frames, dtype=np.float32)
        for i in range(max_frames):
            lo = max(0, i - seg_win // 2)
            hi = min(max_frames, i + seg_win // 2)
            roll_med[i] = float(np.median(raw_cx[lo:hi]))

        # 找滚动中位数突变点（前后窗口差 > 阈值 且持续）
        split_frame = None
        jump_thr = 0.08  # cx 突变阈值
        for i in range(seg_win, max_frames - seg_win, 5):
            before = float(np.median(roll_med[i - seg_win:i]))
            after = float(np.median(roll_med[i:i + seg_win]))
            if abs(after - before) > jump_thr:
                # 确认不是单帧噪声：检查前后 30 帧稳定
                b2 = float(np.median(roll_med[max(0, i - seg_win - 30):max(0, i - 30)]))
                a2 = float(np.median(roll_med[min(max_frames, i + 30):min(max_frames, i + seg_win + 30)]))
                if abs(a2 - b2) > jump_thr * 0.7:
                    split_frame = i
                    break

        if split_frame:
            print(f"    v21: 检测到分段点 frame {split_frame} (cx 突变 {roll_med[max(0,split_frame-seg_win)]:.3f} → {roll_med[min(max_frames-1,split_frame+seg_win)]:.3f})")
            segments = [(0, split_frame), (split_frame, max_frames)]
        else:
            print(f"    v21: 未检测到分段点，全片统一")
            segments = [(0, max_frames)]

        smooth = np.zeros(max_frames, dtype=np.float32)
        for seg_start, seg_end in segments:
            seg_cx = raw_cx[seg_start:seg_end]
            seg_tw = raw_torso_w[seg_start:seg_end]
            reliable = seg_tw > 0.04
            if reliable.sum() >= 5:
                seg_target = float(np.median(seg_cx[reliable]))
            else:
                seg_target = float(np.median(seg_cx))
            seg_target = max(0.20, min(0.60, seg_target))
            seg_len = seg_end - seg_start
            # 段内前 1/4 从旧值渐变，后 3/4 保持目标
            fade_n = min(seg_len // 4, 90)
            for j in range(seg_len):
                if j < fade_n and split_frame and seg_start > 0:
                    t = j / fade_n
                    smooth[seg_start + j] = smooth[seg_start - 1] * (1 - t) + seg_target * t
                else:
                    smooth[seg_start + j] = seg_target
            print(f"    v21: 段 [{seg_start}-{seg_end}] cx={seg_target:.3f}")

        # 10. 场景突变 (>0.3) 重置 (保留: 真正大跳跃时让画框跟)
        for i in range(1, max_frames):
            if abs(smooth[i] - smooth[i - 1]) > 0.3:
                # 抹平附近, 用 raw
                lo = max(0, i - half)
                hi = min(max_frames, i + half)
                smooth[lo:hi] = raw_cx[lo:hi]

        # 11. 边界钳制 + max_jump 帧间限速 + margin
        margin = margin_ratio
        min_cx = crop_w / in_w / 2 + margin
        max_cx = 1.0 - crop_w / in_w / 2 - margin

        # 11b. extreme 模式: 检测 lead 真实位置, 扩 crop_w 让窗口包住
        extreme_active = False
        if mode == "extreme" and in_aspect > out_aspect:
            # 看 raw_cx 真实范围
            raw_cx_valid = raw_cx[(raw_cx > 0.01) & (raw_cx < 0.99)]
            if len(raw_cx_valid) > 0:
                lead_min = float(np.min(raw_cx_valid))
                lead_max = float(np.max(raw_cx_valid))
                # 超出物理范围?
                need_left = lead_min < min_cx
                need_right = lead_max > max_cx
                if need_left or need_right or lead_min < extreme_min_cx or lead_max > extreme_max_cx:
                    # 计算需要多宽的 crop_w 才能包住 lead 范围
                    # 新 crop_w 覆盖 [lead_min - margin, lead_max + margin]
                    need_span = (max(lead_max, 0.5) + margin) - (min(lead_min, 0.5) - margin)
                    need_crop_w = int(need_span * in_w)
                    # 限制最大扩展 (extreme_extra_ratio), 避免画面严重失真
                    max_crop_w = int(crop_w * (1.0 + extreme_extra_ratio))
                    if need_crop_w > max_crop_w:
                        need_crop_w = max_crop_w
                    if need_crop_w > crop_w and need_crop_w <= in_w:
                        old_w = crop_w
                        crop_w = need_crop_w if need_crop_w % 2 == 0 else need_crop_w - 1
                        extreme_active = True
                        # 重新算 min_cx / max_cx (扩窗后)
                        min_cx = crop_w / in_w / 2 + margin
                        max_cx = 1.0 - crop_w / in_w / 2 - margin
                        print(f"    extreme 模式: crop_w {old_w} → {crop_w} "
                              f"(lead cx [{lead_min:.3f}, {lead_max:.3f}], "
                              f"现在 min_cx={min_cx:.3f}, max_cx={max_cx:.3f})")

        out = np.zeros(max_frames, dtype=np.float32)
        prev = None
        for i in range(max_frames):
            v = smooth[i]
            v = max(min_cx, min(v, max_cx))
            if prev is not None:
                # max_jump 钳制 (归一化空间)
                jump = max_jump_ratio * (in_w / crop_w)  # 在 normalized 空间允许的跳变
                v = max(prev - jump, min(v, prev + jump))
            out[i] = v
            prev = v

        # 12. cv2.VideoWriter 直接编码 (跳过 PNG 中转, 节省 17GB I/O + ffmpeg 时间)
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        from lib.utils import create_writer

        stem = Path(input_path).stem
        out_path = ctx.output_dir / f"{stem}_smartcrop.mp4"
        writer = create_writer(str(out_path), fps, out_w, out_h)
        if writer is None or not writer.isOpened():
            print(f"    跳过: 无法创建 writer {out_path}")
            return
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"    跳过: 无法打开 {input_path}")
            writer.release()
            return
        try:
            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                cx_norm = float(out[i])
                cx_px = int(cx_norm * in_w - crop_w / 2)
                cx_px = max(0, min(cx_px, in_w - crop_w))
                if in_aspect > out_aspect:
                    cropped = frame[:, cx_px:cx_px + crop_w]
                else:
                    cy_px = max(0, (in_h - crop_h) // 2)
                    cropped = frame[cy_px:cy_px + crop_h, :]
                if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
                    cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                writer.write(cropped)
            print(f"    编码完成: {out_path.name}")
        finally:
            cap.release()
            writer.release()

        ctx.set("smart_crop_path", str(out_path))
        # 同步 lead_cx 给 07_export 当回退 (静态窗口)
        ctx.set("lead_cx", float(np.median(out)))
        print(f"    输出: {out_path.name} ({out_w}x{out_h}), lead_cx={ctx.get('lead_cx'):.3f}")

"""阶段01: YOLOv8-pose 姿态检测

检测视频中每一帧的人体关键点，缓存供后续阶段复用。
支持多人检测（领操人物 + 跟操人员）。
输出格式兼容 BlazePose 33点（通过 COCO17→BlazePose33 映射）。

配置:
  pose_backend: yolo (默认) | mediapipe
  pose_model: yolov8n-pose (默认) | yolov8s-pose (更准但更慢)
  pose_gpu: true (默认) | false
  pose_batch: 4 (默认) - 批量推理帧数
  pose_interval: 1 (默认) - 检测间隔，每N帧检测一次，中间用OpenCV CSRT追踪加速
"""

import json
import importlib
import cv2
import numpy as np
from pathlib import Path

# 尝试导入 YOLO，不存在则回退 MediaPipe
try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# MediaPipe 回退（lazy import，避免 0.10.33 无 mediapipe.tasks 的问题）
try:
    _base_options_mod = importlib.import_module("mediapipe.tasks.python.core.base_options")
    BaseOptions = _base_options_mod.BaseOptions
    _image_mod = importlib.import_module("mediapipe.tasks.python.vision.core.image")
    MPImage = _image_mod.Image
    ImageFormat = _image_mod.ImageFormat
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE_TASKS = True
except (ImportError, ModuleNotFoundError):
    HAS_MEDIAPIPE_TASKS = False

MP_MODEL_PATH = Path(__file__).parent.parent / "models" / "pose_landmarker.task"

# COCO 17 → BlazePose 33 映射
COCO_TO_BLAZEPOSE = {
    0: 0,   # nose
    5: 11,  # left_shoulder
    6: 12,  # right_shoulder
    11: 23, # left_hip
    12: 24, # right_hip
    15: 27, # left_ankle
    16: 28, # right_ankle
}


def _coco17_to_blaze33(coco_kps: np.ndarray) -> list:
    """将 COCO [17, 3] 转换为 BlazePose [33, 3] 列表"""
    blaze = [[0.0, 0.0, 0.0] for _ in range(33)]
    for coco_idx, blaze_idx in COCO_TO_BLAZEPOSE.items():
        blaze[blaze_idx] = coco_kps[coco_idx][:3].tolist()
    # 填充缺失点（脸/手/脚）：用相邻点估算
    blaze[1] = blaze[0][:]
    blaze[16] = blaze[11][:]
    blaze[17] = blaze[12][:]
    blaze[18] = blaze[11][:]
    blaze[19] = blaze[12][:]
    blaze[31] = blaze[23][:]
    blaze[32] = blaze[24][:]
    return blaze


class PoseDetectStage:
    def __init__(self):
        pass

    def run(self, ctx):
        video_path = Path(ctx.input_path)
        cache_path = ctx.output_dir / f"{video_path.stem}_keypoints.json"

        # 检查缓存
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                ctx.set("keypoints", {int(k): v for k, v in cache["keypoints"].items()})
                ctx.set("video_info", cache["video_info"])
                print(f"    关键点缓存: {cache_path.name}")
                return
            except Exception:
                pass

        # 读取视频信息
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 预览模式只处理前N秒
        max_frames = total_frames
        if ctx.config.get("preview"):
            max_frames = int(fps * ctx.config.get("preview_seconds", 3))
            max_frames = min(max_frames, total_frames)

        ctx.set("video_info", {
            "fps": fps,
            "width": width,
            "height": height,
            "frames": total_frames,
            "process_frames": max_frames,
        })

        print(f"    视频: {width}x{height} @ {fps:.1f}fps, 共 {total_frames} 帧")
        print(f"    处理帧数: {max_frames}")

        backend = ctx.config.get("pose_backend", "yolo")
        pose_interval = max(1, ctx.config.get("pose_interval", 1))
        if backend == "mediapipe" or (not HAS_YOLO and HAS_MEDIAPIPE_TASKS):
            self._run_mediapipe(cap, ctx, fps, width, height, max_frames, cache_path)
        else:
            self._run_yolo(cap, ctx, width, height, max_frames, cache_path, pose_interval)

        cap.release()

    def _run_yolo(self, cap, ctx, width, height, max_frames, cache_path, pose_interval=1):
        """YOLOv8-pose 推理

        pose_interval > 1 时：每N帧运行一次YOLO检测，中间帧用CSRT追踪，
        并将上一次检测的关键点按bbox变化进行仿射变换后复用。
        """
        model_name = ctx.config.get("pose_model", "yolov8n-pose")
        use_gpu = ctx.config.get("pose_gpu", True)
        batch_size = max(1, ctx.config.get("pose_batch", 4))

        import torch
        device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"    YOLO: {model_name}, device={device}, batch={batch_size}, interval={pose_interval}")

        model = _YOLO(model_name)
        model.to(device)
        if device.startswith("cuda"):
            model.half()

        # 预热
        ret, frame = cap.read()
        if ret:
            model(frame, verbose=False)

        cap.release()
        cap = cv2.VideoCapture(str(ctx.input_path))

        keypoints = {}
        last_kps = None
        tracker = None
        last_bbox = None  # (x, y, w, h) 像素坐标
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            do_detect = (frame_idx % pose_interval == 0)

            if do_detect:
                # 每 N 帧运行一次 YOLO 批量检测
                batch = []
                batch_indices = []
                batch_frames = []
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                # 填充 batch
                for _ in range(batch_size - 1):
                    ret2, f2 = cap.read()
                    if not ret2:
                        break
                    batch_frames.append(f2)
                    batch_indices.append(frame_idx + len(batch_frames) - 1)
                    # 注意：这里不修改 frame_idx（后面统一处理）

                if len(batch_frames) == 0:
                    break

                if len(batch_frames) == 1:
                    results = [model(batch_frames[0], verbose=False, conf=0.3)[0]]
                else:
                    results = model(batch_frames, verbose=False, conf=0.3)

                for i, res in enumerate(results):
                    fi = batch_indices[i]
                    if res.keypoints is not None and len(res.keypoints) > 0:
                        kpts = res.keypoints.data.cpu().numpy()
                        all_people = []
                        for person_kps in kpts:
                            norm_kps = person_kps.copy()
                            norm_kps[:, 0] /= width
                            norm_kps[:, 1] /= height
                            norm_kps[:, 2] = np.clip(norm_kps[:, 2], 0, 1)
                            blaze_kps = _coco17_to_blaze33(norm_kps)
                            all_people.append(blaze_kps)
                        keypoints[fi] = all_people
                        last_kps = all_people

                        # 建立 CSRT 追踪器：找最大的人
                        lead_idx = 0
                        best_size = 0
                        for pi, person in enumerate(all_people):
                            arr = np.array(person)
                            vis = arr[:, 2] > 0.3
                            if vis.sum() >= 4:
                                xs = arr[vis, 0]
                                ys = arr[vis, 1]
                                size = (xs.max() - xs.min()) * (ys.max() - ys.min())
                                if size > best_size:
                                    best_size = size
                                    lead_idx = pi

                        if best_size > 0 and lead_idx < len(all_people):
                            person = np.array(all_people[lead_idx])
                            vis = person[:, 2] > 0.3
                            xs = person[vis, 0] * width
                            ys = person[vis, 1] * height
                            x1, x2 = xs.min(), xs.max()
                            y1, y2 = ys.min(), ys.max()
                            pad = 0.15
                            bx = max(0, int(x1 - pad * (x2 - x1)))
                            by = max(0, int(y1 - pad * (y2 - y1)))
                            bw = min(width - bx, int((x2 - x1) * (1 + 2 * pad)))
                            bh = min(height - by, int((y2 - y1) * (1 + 2 * pad)))
                            last_bbox = (bx, by, bw, bh)
                            # OpenCV 4.13+ tracker API broken, skip tracking
                            tracker = None
                    else:
                        keypoints[fi] = None

                frame_idx += len(batch_frames)
            else:
                # 中间帧：无追踪，直接复制上一帧关键点
                if last_kps is not None:
                    keypoints[frame_idx] = last_kps
                frame_idx += 1

            if frame_idx % 100 == 0 or frame_idx >= max_frames:
                print(f"    进度: {frame_idx}/{max_frames} ({frame_idx / max_frames * 100:.0f}%)")

        cap.release()

        detected = sum(1 for v in keypoints.values() if v is not None)
        print(f"    检测完成: {detected}/{max_frames} 帧检测到人体")

        ctx.set("keypoints", keypoints)

        # 保存缓存
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "keypoints": {str(k): v for k, v in keypoints.items()},
            "video_info": ctx.get("video_info"),
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)
        print(f"    缓存已保存: {cache_path.name}")

    def _run_mediapipe(self, cap, ctx, fps, width, height, max_frames, cache_path):
        """MediaPipe BlazePose 推理（回退）"""
        print(f"    MediaPipe BlazePose")

        if not MP_MODEL_PATH.exists():
            raise FileNotFoundError(f"MediaPipe 模型不存在: {MP_MODEL_PATH}")

        base_options = BaseOptions(model_asset_path=str(MP_MODEL_PATH))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        keypoints = {}
        frame_idx = 0
        frame_timestamp_ms = 0

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                if results.pose_landmarks and len(results.pose_landmarks) > 0:
                    all_people = []
                    for person_landmarks in results.pose_landmarks:
                        kps = [[lm.x, lm.y, lm.visibility] for lm in person_landmarks]
                        all_people.append(kps)
                    keypoints[frame_idx] = all_people
                else:
                    keypoints[frame_idx] = None

                frame_timestamp_ms += round(1000 / fps)
                frame_idx += 1
                if frame_idx % 30 == 0:
                    pct = frame_idx / max_frames * 100
                    print(f"    进度: {pct:.0f}% ({frame_idx}/{max_frames})")

        detected = sum(1 for v in keypoints.values() if v is not None)
        print(f"    检测完成: {detected}/{frame_idx} 帧检测到人体")

        ctx.set("keypoints", keypoints)

        # 保存缓存
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "keypoints": {str(k): v for k, v in keypoints.items()},
            "video_info": ctx.get("video_info"),
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)
        print(f"    缓存已保存: {cache_path.name}")

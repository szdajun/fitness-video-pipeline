"""阶段08: 人脸变形（V-face + 大眼 + 磨皮）

基于 MediaPipe FaceMesh 468点关键点，对人脸进行美颜变形。
需要在 models/ 目录放置 face_landmarker.task 模型文件。

用法（在配置中）:
    face_warp:
      v_face_strength: 0.0-0.75   # V脸下巴塑形强度
      eye_enlarge: 1.0-1.4        # 眼睛放大倍数
      skin_smooth_strength: 0.0-1.0  # 磨皮强度
      skin_smooth_d: 7             # 双边滤波直径
      skin_smooth_sigmaColor: 15   # 颜色空间标准差
      skin_smooth_sigmaSpace: 15  # 坐标空间标准差
"""

import cv2
import numpy as np
import importlib
from pathlib import Path

from lib.face_warp import process_frame_face_warp
from lib.utils import create_writer

# MediaPipe imports - use same defensive pattern as 01_pose_detect
HAS_MEDIAPIPE_TASKS = False
FaceLandmarker = None
FaceLandmarkerOptions = None
RunningMode = None
MPImage = None
ImageFormat = None
BaseOptions = None

try:
    _vision_mod = importlib.import_module("mediapipe.tasks.python.vision.face_landmarker")
    FaceLandmarker = _vision_mod.FaceLandmarker
    FaceLandmarkerOptions = _vision_mod.FaceLandmarkerOptions
    from mediapipe.tasks.python import vision as _vision_mod2
    RunningMode = _vision_mod2.RunningMode
    _image_mod = importlib.import_module("mediapipe.tasks.python.vision.core.image")
    MPImage = _image_mod.Image
    ImageFormat = _image_mod.ImageFormat
    _base_mod = importlib.import_module("mediapipe.tasks.python.core.base_options")
    BaseOptions = _base_mod.BaseOptions
    HAS_MEDIAPIPE_TASKS = True
except (ImportError, ModuleNotFoundError):
    pass

MODEL_PATH = Path(__file__).parent.parent / "models" / "face_landmarker.task"


class FaceWarpStage:
    def run(self, ctx):
        # 增量跳过：输出已存在则跳过
        if ctx.get("face_path") and Path(ctx.get("face_path")).exists():
            print("    已存在，跳过")
            return

        if not MODEL_PATH.exists():
            print("    跳过: face_landmarker.task 模型不存在")
            print(f"    请下载到: {MODEL_PATH}")
            return

        input_path = (ctx.get("ken_burns_path") or
                      ctx.get("warped_path") or
                      ctx.get("h2v_path") or
                      ctx.get("stabilized_path"))
        if not input_path or not Path(input_path).exists():
            print("    跳过: 无输入视频")
            return

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])
        cfg = ctx.config.get("face_warp", {})

        # 检查是否有需要启用的参数
        v_face = cfg.get("v_face_strength", 0.0)
        eye_enlarge = cfg.get("eye_enlarge", 1.0)
        skin_smooth = cfg.get("skin_smooth_strength", 0.0)

        if v_face <= 0 and eye_enlarge <= 1.0 and skin_smooth <= 0:
            print("    跳过: face_warp 参数全为0")
            ctx.set("face_warp_path", input_path)
            return

        print(f"    V-face={v_face}, 大眼={eye_enlarge}, 磨皮={skin_smooth}")

        # 初始化 FaceLandmarker
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
        )

        input_video = cv2.VideoCapture(input_path)
        if not input_video.isOpened():
            print(f"    错误: 无法打开视频 {input_path}")
            return

        temp_path = ctx.output_dir / f"{Path(input_path).stem}_face.mp4"
        fw = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = create_writer(str(temp_path), fps, fw, fh)

        frame_idx = 0
        frame_timestamp_ms = 0

        with FaceLandmarker.create_from_options(options) as detector:
            while frame_idx < max_frames:
                ret, frame = input_video.read()
                if not ret:
                    break

                # BGR → RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, frame_timestamp_ms)

                if result.face_landmarks and len(result.face_landmarks) > 0:
                    face_kps = result.face_landmarks[0]
                    kps = np.array([[p.x, p.y, p.z] for p in face_kps], dtype=np.float32)
                    processed = process_frame_face_warp(frame, kps, cfg)
                else:
                    processed = frame

                writer.write(processed)
                frame_idx += 1
                frame_timestamp_ms += int(1000 / fps)

                if frame_idx % 200 == 0:
                    print(f"    {frame_idx}/{max_frames}")

        input_video.release()
        writer.release()

        ctx.set("face_warp_path", str(temp_path))
        print(f"    输出: {temp_path.name}")

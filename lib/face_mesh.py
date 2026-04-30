"""Face landmark detector wrapper used by face beautify stages.

This module prefers the MediaPipe Tasks FaceLandmarker API because it works
with newer MediaPipe packages that expose `mediapipe.tasks` but no longer
ship the classic `mp.solutions.face_mesh` surface. It falls back to the
legacy FaceMesh API when Tasks is unavailable.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

HAS_MEDIAPIPE_TASKS = False
HAS_MEDIAPIPE_SOLUTIONS = False
FaceLandmarker = None
FaceLandmarkerOptions = None
MPImage = None
ImageFormat = None
BaseOptions = None
mp = None

try:
    import mediapipe as mp  # type: ignore

    HAS_MEDIAPIPE_SOLUTIONS = bool(
        hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
    )
except Exception:
    mp = None

try:
    _vision_mod = importlib.import_module("mediapipe.tasks.python.vision.face_landmarker")
    FaceLandmarker = _vision_mod.FaceLandmarker
    FaceLandmarkerOptions = _vision_mod.FaceLandmarkerOptions
    _base_mod = importlib.import_module("mediapipe.tasks.python.core.base_options")
    BaseOptions = _base_mod.BaseOptions

    if mp is not None and hasattr(mp, "Image") and hasattr(mp, "ImageFormat"):
        MPImage = mp.Image
        ImageFormat = mp.ImageFormat
    else:
        _image_mod = importlib.import_module("mediapipe.tasks.python.vision.core.image")
        MPImage = _image_mod.Image
        ImageFormat = _image_mod.ImageFormat

    HAS_MEDIAPIPE_TASKS = True
except (ImportError, ModuleNotFoundError, AttributeError):
    pass


class FaceMeshDetector:
    """Detect face landmarks for one frame at a time."""

    LANDMARK_LEFT_EYE = 33
    LANDMARK_RIGHT_EYE = 263
    LANDMARK_LEFT_PUPIL = 468
    LANDMARK_RIGHT_PUPIL = 473
    LANDMARK_NOSE_TIP = 1
    LANDMARK_LEFT_MOUTH = 61
    LANDMARK_RIGHT_MOUTH = 291
    LANDMARK_UPPER_LIP = 13
    LANDMARK_FOREHEAD = 10
    LANDMARK_LEFT_EYEBROW = 107
    LANDMARK_RIGHT_EYEBROW = 336

    def __init__(self, max_faces: int = 1, refine_landmarks: bool = True):
        self.max_faces = max_faces
        self.refine = refine_landmarks
        self._detector = None
        self._backend = self._select_backend()

    def _select_backend(self) -> str:
        if HAS_MEDIAPIPE_TASKS and MODEL_PATH.exists():
            return "tasks"
        if HAS_MEDIAPIPE_SOLUTIONS:
            return "solutions"
        raise RuntimeError(
            "No compatible MediaPipe face landmark backend is available. "
            "Install mediapipe and keep models/face_landmarker.task in place."
        )

    def _get_detector(self):
        if self._detector is not None:
            return self._detector

        if self._backend == "tasks":
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                num_faces=self.max_faces,
            )
            self._detector = FaceLandmarker.create_from_options(options)
            return self._detector

        self._detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=self.max_faces,
            refine_landmarks=self.refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return self._detector

    def _detect_landmarks(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detector = self._get_detector()

        if self._backend == "tasks":
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            if not result.face_landmarks:
                return None
            return result.face_landmarks[0]

        result = detector.process(rgb)
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark

    def detect(self, frame_bgr):
        """Return a normalized face landmark payload for one frame."""
        landmarks = self._detect_landmarks(frame_bgr)
        if landmarks is None:
            return None

        h, w = frame_bgr.shape[:2]

        def get_xy(idx):
            lm = landmarks[idx]
            return (int(lm.x * w), int(lm.y * h))

        l_eye = get_xy(self.LANDMARK_LEFT_EYE)
        r_eye = get_xy(self.LANDMARK_RIGHT_EYE)
        eye_dist = np.sqrt((r_eye[0] - l_eye[0]) ** 2 + (r_eye[1] - l_eye[1]) ** 2)

        has_pupil_points = len(landmarks) > self.LANDMARK_RIGHT_PUPIL
        if self.refine and has_pupil_points:
            l_pupil = get_xy(self.LANDMARK_LEFT_PUPIL)
            r_pupil = get_xy(self.LANDMARK_RIGHT_PUPIL)
        else:
            l_pupil = l_eye
            r_pupil = r_eye

        nose = get_xy(self.LANDMARK_NOSE_TIP)
        face_cx = (l_eye[0] + r_eye[0]) // 2
        face_cy = (l_eye[1] + r_eye[1]) // 2

        all_landmarks = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z,
                    float(getattr(lm, "visibility", 1.0)),
                ]
                for lm in landmarks
            ],
            dtype=np.float32,
        )

        return {
            "left_eye": l_eye,
            "right_eye": r_eye,
            "left_pupil": l_pupil,
            "right_pupil": r_pupil,
            "nose_tip": nose,
            "face_center": (face_cx, face_cy),
            "eye_distance": eye_dist,
            "all": all_landmarks,
            "img_w": w,
            "img_h": h,
        }

    def detect_batch(self, frames):
        return [self.detect(frame) for frame in frames]

    def close(self):
        if self._detector is not None and hasattr(self._detector, "close"):
            self._detector.close()
        self._detector = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class FaceMeshTracker:
    """Lightweight temporal smoothing on top of per-frame face landmarks."""

    def __init__(self, smooth_window: int = 3):
        self.smooth_window = smooth_window
        self.history = []
        self.detector = FaceMeshDetector(refine_landmarks=True)

    def process_frame(self, frame_bgr):
        det = self.detector.detect(frame_bgr)

        if det is None:
            if self.history:
                return self.history[-1]
            return None

        self.history.append(det)
        if len(self.history) > self.smooth_window:
            self.history.pop(0)

        return self._smooth_landmarks(self.history)

    def _smooth_landmarks(self, history):
        if not history:
            return None

        last = history[-1].copy()
        last["left_eye"] = self._smooth_point(history, "left_eye")
        last["right_eye"] = self._smooth_point(history, "right_eye")
        last["left_pupil"] = self._smooth_point(history, "left_pupil")
        last["right_pupil"] = self._smooth_point(history, "right_pupil")
        last["nose_tip"] = self._smooth_point(history, "nose_tip")
        last["face_center"] = self._smooth_point(history, "face_center")

        dx = last["right_eye"][0] - last["left_eye"][0]
        dy = last["right_eye"][1] - last["left_eye"][1]
        last["eye_distance"] = np.sqrt(dx**2 + dy**2)
        return last

    def _smooth_point(self, history, key):
        points = [item[key] for item in history]
        weights = np.linspace(0.5, 1.0, len(points))
        weights /= weights.sum()
        cx = sum(point[0] * weight for point, weight in zip(points, weights))
        cy = sum(point[1] * weight for point, weight in zip(points, weights))
        return (int(cx), int(cy))

    def reset(self):
        self.history = []
        self.detector.close()
        self.detector = FaceMeshDetector(refine_landmarks=True)

import cv2, numpy as np

# Test face_beautify
print("=== Testing face_beautify (MediaPipe) ===")
import sys
sys.path.insert(0, 'F:/wkspace/fitness-video-pipeline')
from lib.face_mesh import FaceMeshDetector

tracker = FaceMeshDetector(refine_landmarks=True)
frame = np.zeros((480, 640, 3), dtype=np.uint8)
result = tracker.detect(frame)
print("FaceMeshDetector result:", type(result))
print("face_beautify: SUCCESS" if result is not None else "face_beautify: returns None (no face)")

print("\n=== Testing face_beautify2 (InsightFace) ===")
from insightface.app import FaceAnalysis
app = FaceAnalysis('buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
faces = app.get(frame)
print("InsightFace faces:", faces)
print("face_beautify2: SUCCESS")

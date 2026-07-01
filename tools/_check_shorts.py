"""抽帧验证 concat 后的 shorts"""
import cv2
cap = cv2.VideoCapture(r"F:/wkspace/fitness-video-pipeline/output/2026-06-27/艳青1_final_16x9_1920x1080_yt_shorts.mp4")
fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(3))
h = int(cap.get(4))
print(f"shorts: {fc:.0f}f @ {fps:.2f}fps = {fc/fps:.1f}s, {w}x{h}")
for t in [0.5, 1, 2, 3, 5, 10, 20, 25, 26, 27, 28, 29]:
    if t*fps < fc:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t*fps))
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"F:/wkspace/fitness-video-pipeline/_dbg_final_{int(t*10):03d}.png", frame)
            print(f"  t={t}s saved")
cap.release()

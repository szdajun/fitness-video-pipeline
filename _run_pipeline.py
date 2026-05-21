"""Run pipeline with CRF14 + deblock"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os, shutil
os.chdir("F:/wkspace/fitness-video-pipeline")

print("Starting pipeline...", flush=True)

# Copy keypoints cache to today's output dir
today_dir = "F:/wkspace/fitness-video-pipeline/output/2026-05-03"
src_kp = "F:/wkspace/fitness-video-pipeline/output/2026-05-02/枫林红1_keypoints.json"
dst_kp = os.path.join(today_dir, "枫林红1_keypoints.json")
os.makedirs(today_dir, exist_ok=True)
if os.path.exists(src_kp):
    shutil.copy2(src_kp, dst_kp)
    print(f"Cached keypoints: {dst_kp}", flush=True)

# Run main
from main import main
sys.argv = ["main.py", "process", "C:/Users/18091/Desktop/短视频素材/枫林红1.mp4", "-c", "config.yaml"]
main()

"""Run pipeline with CRF14 + deblock and log to file"""
import sys, os
sys.stdout = open(f"F:/wkspace/fitness-video-pipeline/_deblock_run.log", "w", encoding="utf-8", buffering=1)
sys.stderr = sys.stdout

os.chdir("F:/wkspace/fitness-video-pipeline")
from pipeline.engine import PipelineEngine, PipelineContext
from pipeline.config import load_config
import importlib

cfg = load_config("config.yaml")
cfg["output_dir"] = "F:/wkspace/fitness-video-pipeline/output/2026-05-03"

# Import all stages
from stages import (
    pose_detect, stabilize, ken_burns, skin_smooth, denoise,
    beat_flash, energy_bar, intro_outro, export
)

engine = PipelineEngine(cfg)
engine.add_stage("pose_detect", pose_detect.PoseDetectStage())
engine.add_stage("stabilize", stabilize.StabilizeStage())
engine.add_stage("ken_burns", ken_burns.KenBurnsStage())
engine.add_stage("skin_smooth", skin_smooth.SkinSmoothStage())
engine.add_stage("denoise", denoise.DenoiseStage())
engine.add_stage("beat_flash", beat_flash.BeatFlashStage())
engine.add_stage("energy_bar", energy_bar.EnergyBarStage())
engine.add_stage("intro_outro", intro_outro.IntroOutroStage())
engine.add_stage("export", export.ExportStage())

# Copy keypoints cache
import shutil
src_kp = "F:/wkspace/fitness-video-pipeline/output/2026-05-02/枫林红1_keypoints.json"
dst_kp = "F:/wkspace/fitness-video-pipeline/output/2026-05-03/枫林红1_keypoints.json"
os.makedirs(os.path.dirname(dst_kp), exist_ok=True)
if os.path.exists(src_kp):
    shutil.copy2(src_kp, dst_kp)
    print(f"Copied keypoints cache to {dst_kp}")

input_path = "C:/Users/18091/Desktop/短视频素材/枫林红1.mp4"
ctx = PipelineContext(input_path, cfg)
ctx.output_dir = type(ctx.output_dir)(cfg["output_dir"])

print(f"Output dir: {ctx.output_dir}")
print(f"Config CRF: {cfg.get('output', {}).get('crf')}")
print(f"Config deblock: {cfg.get('output', {}).get('deblock')}")

engine.run(ctx)
print("Done!")

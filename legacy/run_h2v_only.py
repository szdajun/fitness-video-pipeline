"""只运行 h2v_convert 阶段，复用已有 keypoints"""
import json, sys
from pathlib import Path

sys.path.insert(0, ".")
import importlib
mod = importlib.import_module("stages.03_h2v_convert")
H2VConvertStage = mod.H2VConvertStage

class Ctx:
    def __init__(self):
        self.data = {}
        self.output_dir = Path("output")
    def get(self, k, default=None):
        return self.data.get(k, default)
    def set(self, k, v):
        self.data[k] = v

# 加载 keypoints
with open("output/艳青和丽丽_keypoints.json", "r", encoding="utf-8") as f:
    saved = json.load(f)

# keypoints 的 key 是字符串，转为 int
kps = {int(k): v for k, v in saved["keypoints"].items()}
video_info = saved["video_info"]

ctx = Ctx()
ctx.input_path = Path("C:/Users/18091/Desktop/短视频素材/艳青和丽丽.mp4")
ctx.data = {
    "keypoints": kps,
    "video_info": video_info,
    "stabilized_path": None,
}

print("=== 运行 h2v_convert ===")
print(f"视频: {video_info['width']}x{video_info['height']}, {video_info['frames']}帧, fps={video_info['fps']:.2f}")
print()

try:
    H2VConvertStage().run(ctx)
    print(f"\nh2v_path: {ctx.get('h2v_path')}")
    print(f"h2v_size: {ctx.get('h2v_size')}")
except Exception as e:
    print(f"失败: {e}")
    import traceback
    traceback.print_exc()

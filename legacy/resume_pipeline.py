"""直接运行剩余 pipeline 阶段，跳过中间件，直接使用已知路径"""
import sys, os, glob
sys.path.insert(0, '.')

from pathlib import Path

# 找到文件
h2v_files = glob.glob('output/*_h2v.mp4')
h2v_files.sort(key=os.path.getmtime)
new_h2v = h2v_files[-1]  # 最新
print(f"h2v: {repr(new_h2v)}")

warped_files = glob.glob('output/*_h2v_warped.mp4')
warped_files.sort(key=os.path.getmtime)
new_warped = warped_files[-1]
print(f"warped: {repr(new_warped)}")

# 读取 h2v 视频信息
import cv2
cap = cv2.VideoCapture(new_h2v)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"h2v视频: {w}x{h}, {frames}帧, fps={fps}")

# ========== Stage 4: ken_burns ==========
print("\n[Stage 4] ken_burns (smooth mode)...")
from stages.ken_burns import KenBurnsStage

class Ctx:
    def __init__(self):
        self.data = {}
        self.output_dir = Path("output")
        self.config = {"ken_burns": {"mode": "smooth", "zoom_range": [1.0, 1.05]}}
    def get(self, k, default=None):
        return self.data.get(k, default)
    def set(self, k, v):
        self.data[k] = v

ctx = Ctx()
ctx.data = {
    "warped_path": new_warped,
    "h2v_path": new_h2v,
    "h2v_size": (w, h),
    "cropped_keypoints": {},
    "video_info": {"fps": fps, "frames": frames, "process_frames": frames},
}

try:
    KenBurnsStage().run(ctx)
except Exception as e:
    print(f"  失败: {e}")
    import traceback; traceback.print_exc()

ken_burns_path = ctx.get("ken_burns_path") or ctx.get("warped_path")
print(f"ken_burns输出: {ken_burns_path}")

# ========== Stage 5: color_grade ==========
print("\n[Stage 5] color_grade...")
from stages.color_grade import ColorGradeStage
ctx2 = Ctx()
ctx2.data = dict(ctx.data)
ctx2.data["video_info"] = ctx.data["video_info"]
ctx2.config = {"color_grade": {}}
try:
    ColorGradeStage().run(ctx2)
except Exception as e:
    print(f"  失败: {e}")
    import traceback; traceback.print_exc()

color_path = ctx2.get("color_path") or ken_burns_path
print(f"color输出: {color_path}")

# ========== Stage 6: export ==========
print("\n[Stage 6] export...")
from stages.export import ExportStage
ctx3 = Ctx()
ctx3.data = {
    "color_path": color_path,
    "warped_path": new_warped,
    "ken_burns_path": ken_burns_path,
    "h2v_path": new_h2v,
    "video_info": {"fps": fps, "frames": frames},
    "stabilized_path": None,
}
ctx3.config = {"output": {"crf": 23, "width": 1080, "height": 1920}, "preview": False}
ctx3.input_path = Path("C:/Users/18091/Desktop/短视频素材/艳青和丽丽.mp4")
ctx3.output_dir = Path("output")
try:
    ExportStage().run(ctx3)
except Exception as e:
    print(f"  失败: {e}")
    import traceback; traceback.print_exc()

print(f"\n最终输出: {ctx3.get('final_path')}")
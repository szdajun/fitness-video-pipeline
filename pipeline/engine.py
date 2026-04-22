"""流水线编排引擎"""

import time, json
from pathlib import Path
from typing import Dict, Any

from .config import load_config, load_preset, _deep_merge


class PipelineContext:
    """流水线上下文，各阶段通过它共享数据"""

    def __init__(self, input_path: str, config: dict):
        self.input_path = Path(input_path)
        self.config = config
        self.data: Dict[str, Any] = {}
        self.output_dir = Path("output")

    def set(self, key: str, value: Any):
        self.data[key] = value

    def get(self, key: str, default=None) -> Any:
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.data


class PipelineEngine:
    """流水线引擎，按顺序执行各阶段"""

    def __init__(self, config: dict):
        self.config = config
        self.stages = []  # [(name, stage_instance, enabled), ...]

    def add_stage(self, name: str, stage, enabled: bool = True):
        self.stages.append((name, stage, enabled))

    def _scan_existing_outputs(self, ctx: PipelineContext):
        """扫描 output_dir 中已存在的中间文件，建立 ctx.data 映射"""
        import cv2
        video_stem = ctx.input_path.stem
        known_files = {
            "keypoints": f"{video_stem}_keypoints.json",
            "stabilized_path": f"{video_stem}_stabilized.mp4",
            "h2v_path": f"{video_stem}_h2v.mp4",
            "warped_path": f"{video_stem}_h2v_warped.mp4",
            "face_path": f"{video_stem}_h2v_warped_face.mp4",
            "color_path": f"{video_stem}_h2v_kenburns_color.mp4",
            "ken_burns_path": f"{video_stem}_h2v_kenburns.mp4",
            "audio_path": f"{video_stem}_audio.aac",
        }
        found = 0
        for key, fname in known_files.items():
            fpath = ctx.output_dir / fname
            if fpath.exists():
                self._set_path(ctx, key, str(fpath))
                found += 1

        # h2v_path 存在时，自动设置 h2v_size（避免后续 ken_burns 等阶段无法获取）
        h2v_path_val = ctx.get("h2v_path")
        if h2v_path_val and Path(h2v_path_val).exists():
            cap = cv2.VideoCapture(h2v_path_val)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                ctx.set("h2v_size", (w, h))

        # cropped_keypoints JSON 文件（h2v_convert 跳过时需要）
        ckp_file = ctx.output_dir / f"{video_stem}_cropped_keypoints.json"
        if ckp_file.exists():
            try:
                with open(ckp_file) as f:
                    ctx.data["cropped_keypoints"] = json.load(f)
            except Exception:
                pass

        if found > 0:
            print(f"  增量: 发现 {found} 个已有文件，将跳过")
        return found

    def _set_path(self, ctx: PipelineContext, key: str, value: str):
        """安全设置 path（已存在的文件不会覆盖）"""
        existing = ctx.data.get(key)
        if existing is None:
            ctx.set(key, value)

    def run(self, ctx: PipelineContext):
        total_start = time.time()
        executed = []

        # 增量扫描：预填充 ctx.data 中已存在的输出路径
        scan_found = self._scan_existing_outputs(ctx)

        print("=" * 50)
        print("  健身短视频处理流水线")
        print("=" * 50)
        print(f"  输入: {ctx.input_path.name}")
        print(f"  预览: {'是 ({}s)'.format(ctx.config.get('preview_seconds', 3)) if ctx.config.get('preview') else '否'}")
        if scan_found > 0:
            print(f"  增量: {scan_found} 个文件已存在")
        print("=" * 50)

        for name, stage, enabled in self.stages:
            if not enabled:
                print(f"  [跳过] {name}")
                continue

            # 增量检查：stage 可通过 ctx 检查自己是否需要运行
            # 机制：stage.run() 内部判断已有输出则打印 "已存在，跳过" 并返回
            print(f"\n  [运行] {name}...")
            t0 = time.time()
            try:
                stage.run(ctx)
                elapsed = time.time() - t0
                print(f"  [完成] {name} ({elapsed:.1f}s)")
                executed.append((name, elapsed))
            except Exception as e:
                print(f"  [失败] {name}: {e}")
                raise

        total = time.time() - total_start
        print("\n" + "=" * 50)
        print(f"  总耗时: {total:.1f}s")
        for name, elapsed in executed:
            print(f"    {name}: {elapsed:.1f}s")
        print("=" * 50)

"""流水线编排引擎"""

import time, json, cv2
from pathlib import Path
from typing import Dict, Any

from .config import load_config, load_preset, _deep_merge
from . import manifest as manifest_lib


class PipelineContext:
    """流水线上下文，各阶段通过它共享数据"""

    def __init__(self, input_path: str, config: dict, output_dir: str = "output"):
        self.input_path = Path(input_path)
        self.config = config
        self.data: Dict[str, Any] = {}
        self.output_dir = Path(output_dir)

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
        video_stem = ctx.input_path.stem
        known_files = {
            "keypoints": f"{video_stem}_keypoints.json",
            "stabilized_path": f"{video_stem}_stabilized.mp4",
            "h2v_path": f"{video_stem}_h2v.mp4",
            "skin_tone_filter_path": f"{video_stem}_h2v_skin_tone.mp4",
            "watermark_path": f"{video_stem}_h2v_watermark.mp4",
            "blush_path": f"{video_stem}_h2v_blush.mp4",
            "warped_path": f"{video_stem}_h2v_warped.mp4",
            "face_path": f"{video_stem}_h2v_warped_face.mp4",
            "color_path": f"{video_stem}_h2v_kenburns_color.mp4",
            "ken_burns_path": f"{video_stem}_h2v_kenburns.mp4",
            "audio_path": f"{video_stem}_audio.aac",
        }
        # 额外文件名变体（横屏精简模式：stabilize → ken_burns → color_grade → skin_smooth）
        extra_patterns = {
            # 旧模式（有 stabilize）
            "ken_burns_path": f"{video_stem}_stabilized_kenburns_16x9.mp4",
            "color_path": f"{video_stem}_stabilized_kenburns_16x9_color.mp4",
            "skin_smooth_path": f"{video_stem}_stabilized_kenburns_16x9_smooth.mp4",
            "denoise_path": f"{video_stem}_stabilized_kenburns_16x9_smooth_denoise.mp4",
            # 新模式（无 stabilize，smooth 模式保持原生分辨率）
            "ken_burns_path": f"{video_stem}_kenburns.mp4",
            # 通用产出
            "beatflash_path": f"{video_stem}_beatflash.mp4",
            "highlight_path": f"{video_stem}_highlight.mp4",
            "energybar_path": f"{video_stem}_energybar.mp4",
            "face_beautify2_path": f"{video_stem}_face_beautify2.mp4",
            "rife_path": f"{video_stem}_rife.mp4",
        }
        found = 0
        for key, fname in extra_patterns.items():
            if key not in ctx.data:  # 不覆盖已存在的
                fpath = ctx.output_dir / fname
                if fpath.exists():
                    ctx.set(key, str(fpath))
                    found += 1
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
        stage_times = {}

        # Manifest 增量恢复
        m = manifest_lib.load_manifest(ctx)
        if m and manifest_lib.is_manifest_compatible(m, ctx):
            restored = manifest_lib.restore_context_from_manifest(ctx, m)
            if restored:
                print(f"  Manifest: 从 manifest 恢复 {restored} 个 stage 的缓存")
        else:
            # 初始化新 manifest
            ctx._manifest = manifest_lib.init_manifest(ctx)
            # 降级：仍做旧的文件扫描作为补充
            scan_found = self._scan_existing_outputs(ctx)
            if scan_found > 0:
                print(f"  增量: 发现 {scan_found} 个已有文件（Manifest 不兼容）")
        ctx._manifest = getattr(ctx, "_manifest", None)

        print("=" * 50)
        print("  健身短视频处理流水线")
        print("=" * 50)
        print(f"  输入: {ctx.input_path.name}")
        print(f"  预览: {'是 ({}s)'.format(ctx.config.get('preview_seconds', 3)) if ctx.config.get('preview') else '否'}")
        print("=" * 50)

        for name, stage, enabled in self.stages:
            if not enabled:
                print(f"  [跳过] {name}")
                continue

            # 检查是否可从 manifest 恢复（stage 内部已设置了输出路径）
            print(f"\n  [运行] {name}...")
            t0 = time.time()
            try:
                stage.run(ctx)
                elapsed = time.time() - t0
                stage_times[name] = elapsed

                # 更新 manifest 并保存（每阶段完成后即保存，支持崩溃恢复）
                if ctx._manifest is not None:
                    outputs = self._collect_stage_outputs(name, ctx)
                    if outputs:
                        manifest_lib.record_stage_result(ctx._manifest, name, outputs)
                        manifest_lib.save_manifest(ctx, ctx._manifest)

                print(f"  [完成] {name} ({elapsed:.1f}s)")
                executed.append((name, elapsed))
            except Exception as e:
                print(f"  [失败] {name}: {e}")
                raise

        # 保存 manifest
        if ctx._manifest is not None:
            manifest_lib.save_manifest(ctx, ctx._manifest)

        total = time.time() - total_start
        print("\n" + "=" * 50)
        print(f"  总耗时: {total:.1f}s")
        for name, elapsed in executed:
            print(f"    {name}: {elapsed:.1f}s")
        print("=" * 50)

        # 写 run_metrics.json
        self._write_metrics(ctx, stage_times)

    def _collect_stage_outputs(self, name: str, ctx: PipelineContext) -> Dict[str, Any]:
        """收集 stage 的产出路径"""
        outputs = {}

        if name == "pose_detect":
            kp = ctx.get("keypoints")
            vi = ctx.get("video_info")
            kp_path = ctx.get("keypoints_path")
            if kp_path:
                outputs["keypoints_path"] = kp_path
            if vi:
                outputs["video_info"] = vi

        elif name == "h2v_convert":
            p = ctx.get("h2v_path")
            if p:
                outputs["h2v_path"] = p
            ck = ctx.get("cropped_keypoints")
            if ck:
                # 写回 cropped_keypoints_path
                ckp = ctx.output_dir / f"{ctx.input_path.stem}_cropped_keypoints.json"
                try:
                    with open(ckp, "w", encoding="utf-8") as f:
                        json.dump(ck, f)
                    outputs["cropped_keypoints_path"] = str(ckp)
                except Exception:
                    pass
            h2v_size = ctx.get("h2v_size")
            if h2v_size:
                outputs["h2v_size"] = list(h2v_size)

        elif name == "body_warp":
            p = ctx.get("warped_path")
            if p:
                outputs["warped_path"] = p

        elif name == "color_grade":
            p = ctx.get("color_path")
            if p:
                outputs["color_path"] = p

        elif name == "ken_burns":
            p = ctx.get("ken_burns_path")
            if p:
                outputs["ken_burns_path"] = p
            r = ctx.get("ken_burns_ratio")
            if r:
                outputs["ken_burns_ratio"] = r

        elif name == "beat_flash":
            p = ctx.get("beatflash_path")
            if p:
                outputs["beatflash_path"] = p

        elif name == "rife":
            p = ctx.get("rife_path")
            if p:
                outputs["rife_path"] = p

        return outputs

    def _write_metrics(self, ctx: PipelineContext, stage_times: Dict[str, float]):
        """输出 run_metrics.json"""
        import os
        metrics_path = ctx.output_dir / f"{ctx.input_path.stem}_metrics.json"

        vi = ctx.get("video_info", {})
        fps = vi.get("fps", 30)
        expected_frames = vi.get("frames", 0)

        # 计算 output 帧数
        final_path = ctx.get("final_path")
        actual_frames = 0
        if final_path and Path(final_path).exists():
            cap = cv2.VideoCapture(final_path)
            if cap.isOpened():
                actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

        metrics = {
            "video_duration_sec": round(actual_frames / fps, 3) if fps > 0 else 0,
            "output_frame_delta": actual_frames - expected_frames if expected_frames > 0 else 0,
            "stage_times": stage_times,
        }

        # 基本质量指标
        kps = ctx.get("keypoints")
        if kps:
            total_frames_with_keypoints = sum(1 for v in kps.values() if v)
            metrics["pose_detect_rate"] = round(total_frames_with_keypoints / len(kps), 3) if len(kps) > 0 else 0

        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

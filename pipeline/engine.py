"""流水线编排引擎"""
from pipeline.process_stage import ProcessStage

import time, json, cv2
from pathlib import Path
from typing import Dict, Any

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
        self._use_process = config.get("process_isolate", False)

    def add_stage(self, name: str, stage, enabled: bool = True):
        if self._use_process and not isinstance(stage, ProcessStage):
            stage = ProcessStage(stage)
        self.stages.append((name, stage, enabled))

    def _scan_existing_outputs(self, ctx: PipelineContext, disabled_stages: set = None):
        """扫描 output_dir 中已存在的中间文件，建立 ctx.data 映射

        Args:
            disabled_stages: 已禁用的 stage 名称集合，跳过这些 stage 的缓存文件
                             (避免 h2v_convert:false 时复用旧的裁切视频)
        """
        if disabled_stages is None:
            disabled_stages = set()
        video_stem = ctx.input_path.stem
        from lib.utils import path_exists as _pe

        # ctx key → 所属 stage 名 (用于判断是否被禁用)
        _key_to_stage = {
            "keypoints": "pose_detect", "pre_deblock_path": "pre_deblock",
            "stabilized_path": "stabilize", "h2v_path": "h2v_convert",
            "warped_path": "body_warp", "face_path": "face_warp",
            "color_path": "color_grade", "ken_burns_path": "ken_burns",
            "skin_smooth_path": "skin_smooth", "denoise_path": "denoise",
            "audio_path": "audio", "beatflash_path": "beat_flash",
            "highlight_path": "highlight", "energybar_path": "energy_bar",
            "intro_path": "intro_outro", "outro_path": "intro_outro",
            "watermark_path": "watermark", "mascot_path": "face_swap",  # 2026-06-29: mascot 禁用, mascot_path 实际由 face_swap 产出. 映射到 face_swap 才不会被 disabled-skip 漏掉
            "blush_path": "blush", "face_beautify_path": "face_beautify",
            "face_beautify2_path": "face_beautify2", "rife_path": "rife",
            "speedramp_path": "speed_ramp", "danmaku_path": "danmaku",
            "burst_path": "intensity_burst", "pip_path": "pip",
            "bgm_path": "bgm_beat", "coldopen_path": "qin_cold_open",
            "skin_tone_filter_path": "skin_tone_filter",
            "face_enhance_path": "face_enhance",
            "skeleton_path": "skeleton_overlay", "count_path": "person_count",
            "leadbox_path": "lead_box", "ghost_path": "lead_ghost",
            "faceblur_path": "face_blur", "heatmap_path": "motion_heatmap",
            "sync_path": "sync_score",
        }

        # 每个 ctx key 对应一个或多个文件名变体（兼容新旧命名规则）
        existing_patterns = {
            "keypoints": [f"{video_stem}_keypoints.json"],
            "pre_deblock_path": [f"{video_stem}_deblocked.mp4"],
            "stabilized_path": [f"{video_stem}_stabilized.mp4"],
            "h2v_path": [f"{video_stem}_h2v.mp4"],
            "skin_tone_filter_path": [f"{video_stem}_h2v_skin_tone.mp4"],
            "watermark_path": [
                f"{video_stem}_watermark.mp4",
                f"{video_stem}_h2v_watermark.mp4",
                f"{video_stem}_energybar_watermark.mp4",  # 2026-06-29: youtube preset watermark 输出名 (input=energybar), 缺它会每次重跑 ~13min
            ],
            "blush_path": [f"{video_stem}_h2v_blush.mp4"],
            "warped_path": [f"{video_stem}_h2v_warped.mp4"],
            "face_path": [f"{video_stem}_h2v_warped_face.mp4"],
            "color_path": [
                f"{video_stem}_color.mp4",  # 2026-06-29: youtube 等无 h2v/stabilize 的 preset, color_grade 直接输出 _color.mp4
                f"{video_stem}_h2v_kenburns_color.mp4",
                f"{video_stem}_stabilized_kenburns_16x9_color.mp4",
            ],
            "ken_burns_path": [
                f"{video_stem}_kenburns.mp4",
                f"{video_stem}_h2v_kenburns.mp4",
                f"{video_stem}_stabilized_kenburns_16x9.mp4",
            ],
            "audio_path": [f"{video_stem}_audio.aac"],
            "skin_smooth_path": [f"{video_stem}_stabilized_kenburns_16x9_smooth.mp4"],
            "denoise_path": [f"{video_stem}_stabilized_kenburns_16x9_smooth_denoise.mp4"],
            "beatflash_path": [f"{video_stem}_beatflash.mp4"],
            "highlight_path": [f"{video_stem}_highlight.mp4"],
            "energybar_path": [f"{video_stem}_energybar.mp4"],
            "mascot_path": [
                # 2026-06-29: face_swap 跑完 set mascot_path=faceswap. face_swap 跳过时也要能
                # 从 faceswap 接力 (否则 danmaku 读未换脸 fallback → 最终没换脸). 放首位.
                f"{video_stem}_faceswap.mp4",
                f"{video_stem}_mascot.mp4",
                f"{video_stem}_energybar_watermark_mascot.mp4",
            ],
            # 2026-06-29: 注册换脸输出. STAGE_OUTPUT_KEYS["face_swap"]=["face_swap_path"],
            # 若不注册此模式, 预扫描设不了 ctx.face_swap_path → incremental 每次都重跑 face_swap
            # (即便 _faceswap.mp4 已存在), 浪费 ~40min/次. stages/37 输出名固定为 _faceswap.mp4.
            "face_swap_path": [f"{video_stem}_faceswap.mp4"],
            "danmaku_path": [
                f"{video_stem}_danmaku.mp4",
                f"{video_stem}_energybar_watermark_mascot_danmaku.mp4",
            ],
            "burst_path": [
                f"{video_stem}_burst.mp4",
                f"{video_stem}_energybar_watermark_mascot_danmaku_burst.mp4",
            ],
            "filmlook_path": [
                f"{video_stem}_film.mp4",
                f"{video_stem}_energybar_watermark_mascot_danmaku_film.mp4",
            ],
            "pip_path": [
                f"{video_stem}_pip.mp4",
                f"{video_stem}_energybar_watermark_mascot_danmaku_film_pip.mp4",
            ],
            "bgm_path": [
                f"{video_stem}_bgm.mp4",
                f"{video_stem}_energybar_watermark_mascot_danmaku_film_pip_withbgm.mp4",
            ],
            "speedramp_path": [f"{video_stem}_speedramp.mp4"],
            "face_beautify_path": [f"{video_stem}_face_beautify.mp4"],
            "face_beautify2_path": [f"{video_stem}_face_beautify2.mp4"],
            "rife_path": [f"{video_stem}_rife.mp4"],
            "face_enhance_path": [f"{video_stem}_final_16x9_enhanced.mp4"],
            "intro_path": [f"{video_stem}_intro.mp4"],
            "outro_path": [f"{video_stem}_outro.mp4"],
            "coldopen_path": [f"{video_stem}_coldopen.mp4"],
            "skeleton_path": [f"{video_stem}_skeleton.mp4"],
            "count_path": [f"{video_stem}_count.mp4"],
            "leadbox_path": [f"{video_stem}_leadbox.mp4"],
            "ghost_path": [f"{video_stem}_ghost.mp4"],
            "faceblur_path": [f"{video_stem}_faceblur.mp4"],
            "heatmap_path": [f"{video_stem}_heatmap.mp4"],
            "sync_path": [f"{video_stem}_sync.mp4"],
            "final_path": [f"{video_stem}_final_16x9.mp4"],
            "shorts_path": [f"{video_stem}_shorts.mp4", f"{video_stem}_shorts_v2.mp4"],
        }
        found = 0
        for key, fnames in existing_patterns.items():
            if key in ctx.data:
                continue
            # 跳过已被禁用的 stage 的缓存（避免 h2v_convert:false 时复用旧裁切视频）
            stage_name = _key_to_stage.get(key)
            if stage_name and stage_name in disabled_stages:
                continue
            for fname in fnames:
                fpath = ctx.output_dir / fname
                if _pe(str(fpath)):
                    # 关键点需要加载为数据，不是路径字符串
                    if key == "keypoints":
                        try:
                            with open(fpath, encoding="utf-8") as f:
                                raw = json.load(f)
                                ctx.set("keypoints", raw.get("keypoints", raw))
                                ctx.set("keypoints_path", str(fpath))
                            found += 1
                        except Exception:
                            pass
                    else:
                        ctx.set(key, str(fpath))
                        found += 1
                    break

        # h2v_path 存在时，自动设置 h2v_size（避免后续 ken_burns 等阶段无法获取）
        h2v_path_val = ctx.get("h2v_path")
        if h2v_path_val and Path(h2v_path_val).exists():
            cap = cv2.VideoCapture(h2v_path_val)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                ctx.set("h2v_size", (w, h))

        # cropped_keypoints JSON 文件（仅 h2v_convert 启用时加载）
        if "h2v_convert" not in disabled_stages:
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

    def run(self, ctx: PipelineContext):
        total_start = time.time()
        executed = []
        stage_times = {}

        # 确保输出目录在中间阶段写入前已创建
        ctx.output_dir.mkdir(parents=True, exist_ok=True)

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
        # 只扫描已启用的 stage，避免禁用 stage 复用旧缓存
        # （如 h2v_convert: false 时不应复用旧 h2v 裁切视频）
        disabled_stages = {name for name, _, enabled in self.stages if not enabled}
        scan_found = self._scan_existing_outputs(ctx, disabled_stages)
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
                output_keys = self.STAGE_OUTPUT_KEYS.get(name, [])
                has_output = any(ctx.get(k) is not None for k in output_keys)
                if has_output:
                    print(f"  [已有] {name}")
                else:
                    print(f"  [跳过] {name}")
                continue

            # 已启用的 stage：若已有全部产出，跳过（继续模式）
            if enabled:
                output_keys = self.STAGE_OUTPUT_KEYS.get(name, [])
                if output_keys and all(ctx.get(k) is not None for k in output_keys):
                    print(f"  [已有] {name}")
                    continue

            # 检查是否可从 manifest 恢复（stage 内部已设置了输出路径）
            print(f"\n  [运行] {name}...")
            t0 = time.time()
            try:
                stage.run(ctx)
                # 入口校验失败 → 中断所有后续 stage
                if name == "preflight" and ctx.get("preflight_ok") is False:
                    print("  [中断] 入口校验失败，跳过所有后续阶段")
                    break
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

    # 每个 stage 产出到 manifest 的 ctx key 映射
    STAGE_OUTPUT_KEYS = {
        "pose_detect":       ["keypoints_path", "video_info"],
        "pre_deblock":       ["pre_deblock_path"],
        "stabilize":         ["stabilized_path"],
        "h2v_convert":       ["h2v_path", "h2v_size", "cropped_keypoints"],
        "body_warp":         ["warped_path"],
        "ken_burns":         ["ken_burns_path", "ken_burns_ratio"],
        "face_warp":         ["face_path"],
        "color_grade":       ["color_path"],
        "skin_smooth":       ["skin_smooth_path"],
        "skin_tone_filter":  ["skin_tone_filter_path"],
        "denoise":           ["denoise_path"],
        "audio":             ["audio_path"],
        "skeleton_overlay":  ["skeleton_path"],
        "person_count":      ["count_path"],
        "lead_box":          ["leadbox_path"],
        "lead_ghost":        ["ghost_path"],
        "face_blur":         ["faceblur_path"],
        "motion_heatmap":    ["heatmap_path"],
        "sync_score":        ["sync_path"],
        "beat_flash":        ["beatflash_path"],
        "highlight":         ["highlight_path"],
        "energy_bar":        ["energybar_path"],
        "intro_outro":       ["intro_path", "outro_path"],
        "watermark":         ["watermark_path"],
        "mascot":            ["mascot_path"],
        "blush":             ["blush_path"],
        "face_beautify":     ["face_beautify_path"],
        "face_beautify2":    ["face_beautify2_path"],
        "rife":              ["rife_path"],
        "speed_ramp":        ["speedramp_path"],
        "danmaku":           ["danmaku_path"],
        "intensity_burst":   ["burst_path"],
        "film_look":         ["filmlook_path"],
        "pip":               ["pip_path"],
        "bgm_beat":          ["bgm_path"],
        "qin_cold_open":     ["coldopen_path"],
        "export":            ["final_path"],
        "shorts":            ["shorts_path", "douyin_vertical_path"],
        "face_enhance":      ["face_enhance_path"],
        "face_swap":         ["face_swap_path"],
    }

    def _collect_stage_outputs(self, name: str, ctx: PipelineContext) -> Dict[str, Any]:
        """收集 stage 的产出路径"""
        outputs = {}
        keys = self.STAGE_OUTPUT_KEYS.get(name, [])
        for k in keys:
            v = ctx.get(k)
            if v is not None:
                if k == "h2v_size":
                    outputs[k] = list(v) if isinstance(v, tuple) else v
                else:
                    outputs[k] = v

        # 特殊处理：h2v_convert 需要写回 cropped_keypoints.json
        if name == "h2v_convert":
            ck = ctx.get("cropped_keypoints")
            if ck:
                ckp = ctx.output_dir / f"{ctx.input_path.stem}_cropped_keypoints.json"
                try:
                    with open(ckp, "w", encoding="utf-8") as f:
                        json.dump(ck, f)
                    outputs["cropped_keypoints_path"] = str(ckp)
                except Exception:
                    pass

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
        except Exception as e:
            print(f"    警告: 无法保存 metrics: {e}")

# CHANGELOG — 健身视频处理流水线

## [Unreleased]

### Added
- `lib/highlight_protect.py` — 夜景高光三层处理模块
  - `protect_highlights()`: LAB L通道 soft roll-off 压缩
  - `protect_bright_neutral_regions()`: HSV 白衣/低饱和区保护
  - `suppress_large_light_regions()`: 连通域大面积灯区抑制
  - `optimize_night_highlights()`: 统一入口
- `lib/crop_strategy.py` — H2V 裁切策略提取
  - `build_tracks()`, `select_lead_track()`, `classify_frames()`, `classify_frame()`, `merge_segments()`, `get_lead_center_in_segment()`
- `pipeline/manifest.py` — Manifest 增量恢复系统
  - `CACHE_VERSION = 1`
  - `load_manifest()`, `is_manifest_compatible()`, `restore_context_from_manifest()`, `record_stage_result()`, `init_manifest()`, `build_input_fingerprint()`, `compute_config_hash()`
- `lib/quality_metrics.py` — run_metrics.json 输出
  - `compute_pose_detect_rate()`, `compute_avg_person_count()`, `compute_lead_center_jitter()`, `dump_metrics()`
- `presets/night_square_dance.yaml` — 夜景广场健身专用 preset
- `presets/sexy.yaml` — 添加轻量高光参数
- `presets/night_gym.yaml` — 添加完整三层高光参数
- `stages/06_color_grade.py` — 集成三层高光处理（adaptive_contrast 之后、CLAHE 之前）
- `main.py` — `night_square_dance` preset 注册

### Changed
- `pipeline/engine.py` — 集成 Manifest 增量恢复 + metrics 输出 + `output_dir` 参数
- `stages/03_h2v_convert.py` — 导入 `lib/crop_strategy.py` 策略函数，移除冗余 local 方法

### Deprecated
- `fitness_processor.py`, `loop_video.py`, `_build_16x9.py`, `_process_16x9.py` — 已废弃，保留参考

---

## v1.1.9 (2026-04-27)

### Added
- 27-stage 完整 pipeline (`main.py` + `pipeline/engine.py`)
- Manifest 增量恢复（config hash + input fingerprint 校验）
- `lib/quality_metrics.py` run_metrics.json
- `.gitignore`
- `tools/` 目录（废弃脚本整理）

### Fixed
- 帧数自动校正（cv2 读帧误差补偿）
- h2v_convert 增量跳过逻辑
- preview 模式 `process_frames` 残留问题

---

## v1.1.0 — GPT-5.4 文档审核后新增

### Added
- **夜景高光三层处理**（最关键）：LAB L通道 roll-off → HSV 白衣保护 → 连通域灯区抑制
- **night_square_dance** 专用 preset（保守裁切 + 多人在景优先 + 弱 Ken Burns）
- H2V 策略拆分（`lib/crop_strategy.py`）

### Changed
- `stages/06_color_grade.py` 处理顺序：高光三层处理 → CLAHE → 锐化
- 文档统一：`docs/视频生成流程.md` 反映当前 27-stage 架构

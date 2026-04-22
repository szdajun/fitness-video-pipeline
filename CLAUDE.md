# Fitness Video Pipeline

## Project Overview

健身短视频处理流水线。将横拍健身视频转为竖版 (9:16)，自动完成人体比例调整、运镜效果和色彩调色。

## Architecture

基于 **Pipeline + Stage** 架构，通过 `PipelineContext` 在阶段间传递数据。

### Stage 顺序（不可随意调换）

| Stage | 文件 | 功能 |
|-------|------|------|
| 1 | `stages/01_pose_detect.py` | YOLOv8-pose 姿态检测，批量推理 + GPU FP16 |
| 2 | `stages/02_stabilize.py` | FFmpeg vidstab 视频稳定（默认关闭）|
| 3 | `stages/03_h2v_convert.py` | 横转竖裁切，领操人智能跟踪（FFmpeg 一刀切）|
| 4 | `stages/04_ken_burns.py` | Ken Burns 运镜（smooth / dual 双模式）|
| 5 | `stages/05_body_warp.py` | 体型调整（瘦腿/瘦腰/丰满/长腿等）|
| 6 | `stages/06_color_grade.py` | 色彩调色（亮度/对比度/饱和度/色温/CLAHE）|
| 7 | `stages/07_export.py` | 合并音频，H.264 输出 |

### 其他 Stage

| Stage | 文件 | 功能 |
|-------|------|------|
| 9 | `stages/09_audio.py` | 音频处理（响度标准化+背景音乐）|
| 10 | `stages/10_skeleton_overlay.py` | 骨架叠加显示 |
| 11 | `stages/11_person_count.py` | 人数统计 |
| 12 | `stages/12_lead_box.py` | 领操人边框高亮 |
| 13 | `stages/13_lead_ghost.py` | 领操人鬼影叠加 |
| 14 | `stages/14_face_blur.py` | 脸部模糊（隐私保护）|
| 15 | `stages/15_motion_heatmap.py` | 运动热力图 |
| 16 | `stages/16_sync_score.py` | 跟操评分 |
| 17 | `stages/17_beat_flash.py` | 节拍闪烁效果 |
| 18 | `stages/18_highlight.py` | 精华片段标记 |
| 19 | `stages/19_energy_bar.py` | 运动能量条 |

### Key Modules

- `pipeline/engine.py` — 流水线执行引擎
- `pipeline/config.py` — 配置管理（含 load_preset / _deep_merge）
- `lib/tracker.py` — Kalman 滤波人物追踪 + `LeadPersonSmoother`
- `lib/warp.py` — 体型变形位移图生成（meshgrid 模块级缓存）
- `lib/utils.py` — 工具函数（create_writer: avc1 优先 mp4v fallback）
- `lib/yolo_pose.py` — YOLO pose 封装
- `main.py` — CLI 入口（subcommand: process / batch）

### Presets (`presets/`)

| 预设 | 适用场景 |
|------|----------|
| `shorts` | 抖音/快手/Shorts 竖版（低分辨率手持拍摄，默认禁用 stabilize/ken_burns）|
| `youtube` | YouTube 横屏版 |
| `sexy` | 强效体型调整（瘦腰 0.75, 丰满 1.35, 长腿 1.25）|
| `natural` | 自然微调 |
| `dramatic` | 电影感调色 + dual 运镜 |
| `gimbal` | 云台/固定机位（启用 stabilization）|
| `beauty` | 多人场景，智能领操人识别 |
| `night_gym` | 低光环境优化 |
| `clean` | 最小处理 |

## CLI Usage

```bash
# 单视频处理
python main.py process "input.mp4" --preset shorts
python main.py process "input.mp4" --preset shorts --full-video

# 批量处理
python main.py batch -i "input_dir" -o "output_dir" --segment 45

# 单独参数覆盖
python main.py process "input.mp4" --leg-lengthen 1.2 --waist-slim 0.85

# 禁用特定阶段
python main.py process "input.mp4" --no-stabilize --no-ken-burns --preview
```

## Key Implementation Details

### cv2.VideoCapture H.264 Bug

某些 H.264 文件 `CAP_PROP_FRAME_COUNT` 报告错误帧数，但实际读取在约 900 帧处 `cap.read()` 返回 False。已通过 `_run_dual_ffmpeg` 保存 JPEG 序列再用 FFmpeg 编码的方式修复。

### GPU Acceleration

Pose 检测默认使用 GPU + FP16（`model.half()`）。可通过 `--no-pose-gpu` 禁用。

### Keypoints Caching

关键点检测结果缓存到 `*_keypoints.json`，修改检测逻辑后需删除缓存。

### 增量处理

所有中间文件支持增量跳过：若输出已存在则打印"已存在，跳过"。

### 重要配置路径

- `ctx.config.get("stages", {}).get("stabilize", {})` — stabilize 配置（勿用 `ctx.config.get("stabilize")`）
- `ctx.config.get("stages", {}).get("ken_burns", {})` — ken_burns 配置

## Conventions

- Python 3.9+，依赖 OpenCV / ultralytics / FFmpeg / NumPy / PyYAML
- 体型参数范围：slim 0.7~1.0, enlarge/lengthen 1.0~1.4
- Stage 编号前缀 (`01_`, `02_`) 表示执行顺序，不可随意调换
- `lib/` 目录存放跨 stage 共享的底层模块
- `output/` 目录存放处理结果和中间文件

## Files for Distribution

- `README.md` — 项目概述、快速开始、命令参考
- `docs/manual.md` — 完整用户手册
- `presets/README.md` — 预设风格详解
- `requirements.txt` — Python 依赖
- `pyproject.toml` — 项目打包配置

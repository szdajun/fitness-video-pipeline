# 健身短视频处理流水线 — 用户手册

## 目录

1. [安装指南](#1-安装指南)
2. [快速入门](#2-快速入门)
3. [预设风格详解](#3-预设风格详解)
4. [命令行参数](#4-命令行参数)
5. [配置文件格式](#5-配置文件格式)
6. [流水线阶段](#6-流水线阶段)
7. [硬件配置](#7-硬件配置)
8. [常见问题](#8-常见问题)

---

## 1. 安装指南

### 环境要求

- Python 3.9 或更高版本
- FFmpeg（视频编解码和稳定化）
- OpenCV 4.8+
- NVIDIA GPU（可选，用于 YOLO 加速）

### 安装步骤

**Step 1: 克隆项目**

```bash
cd your/project/path
```

**Step 2: 创建虚拟环境（推荐）**

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux
```

**Step 3: 安装 Python 依赖**

```bash
pip install -r requirements.txt
```

**Step 4: 安装 FFmpeg**

- Windows: 从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载，解压到 `C:\Users\18091\ffmpeg\`
- 或使用包管理器: `winget install ffmpeg`

验证安装：
```bash
ffmpeg -version
```

**Step 5:（可选）安装 YOLO GPU 加速**

```bash
pip install ultralytics torch torchvision
```

**Step 6: 下载 YOLO 姿态模型（首次自动下载，也可手动）**

```bash
python download_yolo_model.py
```

---

## 2. 快速入门

### 最简用法

```bash
# 处理单个视频（使用默认配置）
python main.py process "C:/素材/video.mp4"
```

### 使用预设风格

```bash
# 竖版短视频（抖音/快手/YouTube Shorts）
python main.py process "video.mp4" --preset shorts

# 保留原始横版（YouTube）
python main.py process "video.mp4" --preset youtube

# 强效体型美化
python main.py process "video.mp4" --preset sexy
```

### 批量处理

```bash
# 处理整个目录
python main.py batch -i "C:/素材/" -o "C:/输出/"

# 自动切割为45秒片段（适合短视频平台）
python main.py batch -i "C:/素材/" --segment 45
```

### 预览效果

```bash
# 只处理前3秒，快速验证效果
python main.py process "video.mp4" --preset shorts --preview
```

---

## 3. 预设风格详解

### shorts — 竖版短视频（推荐）

适合抖音、快手、YouTube Shorts 等竖屏平台。

```bash
python main.py process "video.mp4" --preset shorts --full-video
```

**特点：**
- 竖版 9:16 (1080x1920)
- 自动识别领操人并居中
- 体型美化（瘦腿/瘦腰）
- 节拍闪烁效果
- 运动能量条叠加
- **stabilize: false**（适合手持拍摄）
- **ken_burns: false**（避免放大抖动）

### youtube — YouTube 横屏版

保留原始 16:9 比例，叠加能量条和节拍效果。

```bash
python main.py process "video.mp4" --preset youtube --full-video
```

**特点：**
- 横版 16:9 (1920x1080)
- 不改变原始构图
- 叠加运动能量条
- 节拍闪烁效果

### sexy — 强效体型美化

适合展示身材，强调视觉效果。

```bash
python main.py process "video.mp4" --preset sexy
```

**体型参数：**
- 腿部拉长: 1.25x
- 腰部塑形: 0.75x
- 胸部丰满: 1.35x
- 头部比例: 1.05x

### natural — 自然微调

轻度体型调整，保持自然感。

### dramatic — 电影感调色

高对比度 + 暖色调，增强视觉冲击。

### gimbal — 云台/固定机位

适合画面稳定的高清素材，启用视频稳定化。

### beauty — 多人场景

智能识别领操人 + 团队全景自动切换。

### night_gym — 晚间健身房

低光环境优化：降噪 + 提亮 + 暖色色温校正。

### clean — 最小处理

仅做横转竖裁切，不添加任何特效。

---

## 4. 命令行参数

### 基础参数

| 参数 | 说明 |
|------|------|
| `input` | 输入视频路径（必需）|
| `-o, --output` | 指定输出路径 |
| `-c, --config` | 自定义配置文件路径 |
| `--preset` | 选择预设风格 |
| `--preview` | 预览模式（处理前3秒）|

### 阶段控制

| 参数 | 说明 |
|------|------|
| `--no-stabilize` | 禁用视频稳定 |
| `--no-body-warp` | 禁用体型调整 |
| `--no-face-warp` | 禁用脸部美化 |
| `--no-color-grade` | 禁用色彩增强 |
| `--no-ken-burns` | 禁用运镜效果 |
| `--no-pose-gpu` | 禁用 GPU 加速 |
| `--full-video` | 生成完整视频（跳过精华片段）|

### 体型参数

| 参数 | 说明 | 范围 |
|------|------|------|
| `--leg-lengthen` | 腿部拉长比例 | 1.0~1.4 |
| `--leg-slim` | 腿部瘦比例 | 0.7~1.0 |
| `--waist-slim` | 腰部瘦比例 | 0.7~1.0 |
| `--overall-slim` | 整体瘦身比例 | 0.7~1.0 |
| `--chest-enlarge` | 胸部放大比例 | 1.0~1.3 |
| `--neck-lengthen` | 脖子拉长比例 | 1.0~1.3 |
| `--head-ratio` | 头身比 | 0.8~1.2 |

### 色彩参数

| 参数 | 说明 | 范围 |
|------|------|------|
| `--brightness` | 亮度 | -100~100 |
| `--contrast` | 对比度 | 0.5~2.0 |
| `--saturation` | 饱和度 | 0.0~2.0 |
| `--warmth` | 色温 | -50~50 |
| `--auto-wb` | 自动白平衡 | 开关 |

### 输出控制

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--crf` | 视频质量 (越小越大) | 26 |
| `--enc-preset` | 编码速度 | fast |
| `--audio-bitrate` | 音频码率 | 96k |
| `--video-fade-out` | 片尾淡出秒数 | 2.0 |
| `--cut` | 裁切重复片段 | 如 `30-60,120-150` |

### 示例命令

```bash
# 竖版 + 自定义体型参数
python main.py process "video.mp4" --preset shorts \
  --leg-lengthen 1.2 --waist-slim 0.85 --chest-enlarge 1.15

# 横版 + 关闭所有特效
python main.py process "video.mp4" --preset youtube \
  --no-stabilize --no-ken-burns --no-color-grade

# 批量处理 + 切割45秒片段
python main.py batch -i "C:/素材/" -o "C:/输出/" --segment 45 --preset shorts

# 自定义色彩
python main.py process "video.mp4" --preset dramatic \
  --brightness 10 --saturation 1.2 --warmth 10
```

---

## 5. 配置文件格式

创建 `my_config.yaml`：

```yaml
# 启用的阶段
stages:
  pose_detect: true
  stabilize: false       # 手持拍摄建议关闭
  h2v_convert: true
  ken_burns: false       # 低分辨率建议关闭
  body_warp: true
  beat_flash: true
  highlight: false
  energy_bar: true
  color_grade: true

# 姿态检测配置
pose_backend: yolo        # yolo | mediapipe
pose_model: yolov8n-pose  # yolov8n-pose(快) | yolov8s-pose(准)
pose_gpu: true            # true=优先用 GPU

# 稳定化配置（stages.stabilize=true 时生效）
stabilize:
  shakiness: 12          # 抖动检测灵敏度 1-15
  accuracy: 15            # 向量检测精度
  zoom: 5                 # 画面放大补偿（px）
  smoothing: 40           # 平滑帧数

# 体型调整
body_warp:
  leg_lengthen: 1.0      # 腿部拉长 1.0-1.4
  leg_slim: 1.0           # 腿部瘦 0.7-1.0
  waist_slim: 1.0         # 腰部瘦 0.7-1.0
  overall_slim: 1.0       # 整体瘦 0.7-1.0
  chest_enlarge: 1.0      # 胸部大 1.0-1.3
  head_ratio: 1.0         # 头身比 0.8-1.2

# Ken Burns 运镜
ken_burns:
  mode: smooth           # smooth(微幅) | dual(景别切换)
  zoom_range: [1.0, 1.03] # smooth 模式缩放范围
  # dual 模式参数
  dual_close_zoom: 1.08
  dual_close_zoom_v: 1.04
  dual_cycle_seconds: 10
  dual_pan_amplitude: 5
  dual_dwell: 0.5

# 能量条
energy_bar:
  width: 16
  margin_right: 20
  height: 400
  smoothing: 0.85

# 色彩增强
color_grade:
  brightness: 0
  contrast: 1.0
  saturation: 1.0
  warmth: 0
  clahe: true

# 输出配置
output:
  width: 1080
  height: 1920
  crf: 26
  preset: fast
  audio_bitrate: 96k
```

使用自定义配置：
```bash
python main.py process "video.mp4" -c my_config.yaml
```

---

## 6. 流水线阶段

### 01_pose_detect — 人体关键点检测

- **输入**: 原始视频
- **输出**: 关键点缓存 JSON
- **技术**: YOLOv8-pose（默认）或 MediaPipe BlazePose
- **加速**: 批量推理 (batch=4) + GPU FP16

### 02_stabilize — 视频稳定化

- **输入**: 原始视频
- **输出**: 稳定化后的视频
- **技术**: FFmpeg vid.stab（双通道：vidstabdetect + vidstabtransform）
- **注意**: 低分辨率手持拍摄建议关闭

### 03_h2v_convert — 横转竖裁切

- **输入**: 稳定化视频
- **输出**: 9:16 竖版裁切视频 (1080x1920)
- **智能**: 自动识别领操人并居中，多人时切换为全景构图
- **技术**: FFmpeg crop + 关键点追踪

### 04_ken_burns — Ken Burns 运镜

- **输入**: 竖版裁切视频
- **输出**: 添加运镜效果的竖版视频
- **模式**:
  - `smooth`: 正弦微幅缩放 (1.0-1.03x)，增加动感
  - `dual`: 全景/特写周期性切换，跟随领操人位置

### 05_body_warp — 体型美化

- **输入**: 竖版视频
- **输出**: 体型调整后的视频
- **技术**: OpenCV 网格变形 + 双边滤波保边

### 06_color_grade — 色彩增强

- **输入**: 竖版视频
- **输出**: 色彩调整后的视频
- **效果**: 亮度/对比度/饱和度/色温 + CLAHE 自适应直方图均衡

### 07_export — 最终导出

- **输入**: 所有处理后的视频 + 原始音频
- **输出**: H.264 编码最终视频
- **处理**: 音频合并 + 视频淡出 + 缩放到目标分辨率

---

## 7. 硬件配置

### GPU 加速（推荐）

安装 NVIDIA GPU 驱动后，YOLO 自动使用 CUDA 加速：

```bash
pip install torch torchvision ultralytics
```

验证 GPU：
```python
import torch
print(torch.cuda.is_available())  # True = GPU 可用
```

### 内存优化

处理大文件时可关闭特效节省内存：
```bash
python main.py process "video.mp4" --no-body-warp --no-color-grade
```

---

## 8. 常见问题

### Q: 竖版视频抖动严重

**A:** 低分辨率手持拍摄（<1080p）不适合 stabilization。建议：
```bash
python main.py process "video.mp4" --preset shorts --no-stabilize --no-ken-burns
```

### Q: 处理速度慢

**A:** 确保 GPU 可用（`torch.cuda.is_available()`），或减少并行数：
```bash
python main.py batch --workers 2
```

### Q: 体型调整无效

**A:** 确保 `body_warp: true` 且 `h2v_convert: true`（体型变形在竖版裁切后进行）：
```bash
python main.py process "video.mp4" --leg-lengthen 1.2 --waist-slim 0.8
```

### Q: 关键点检测失败（黑屏/无检测）

**A:** 清理缓存后重试：
```bash
rm output/*_keypoints.json
python main.py process "video.mp4"
```

### Q: 视频无声

**A:** 检查原始视频是否有音轨。音频自动从原始视频提取合并，无需单独处理。

### Q: 如何只输出特定阶段的结果？

**A:** 目前不支持中间产物输出，可通过修改 `config.yaml` 启用/禁用特定阶段实现。

---

## 技术支持

如遇到问题，请提供：
1. 错误信息（完整 stderr 输出）
2. 视频分辨率和时长
3. 使用的预设和参数
4. `python main.py --version` 输出（如适用）

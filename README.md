# 健身短视频处理流水线

将横拍健身视频自动转换为竖版 (9:16) 短视频，支持体型美化、运镜效果、节拍闪烁、能量条等特效。

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)

## 功能特性

| 功能 | 说明 |
|------|------|
| **横转竖裁切** | 自动识别领操人，9:16 竖版构图 |
| **体型美化** | 瘦腿、腰部塑形、胸部丰满、头身比例调整 |
| **Ken Burns 运镜** | smooth（微幅缩放）或 dual（全景/特写切换） |
| **节拍闪烁** | 跟随音乐节拍的画面闪白效果 |
| **运动能量条** | 实时显示领操人运动强度的垂直能量条 |
| **色彩增强** | 亮度/对比度/饱和度/色温调节 + CLAHE 自适应直方图均衡 |
| **骨架叠加** | 叠加姿态关键点骨架显示（可选）|
| **视频稳定** | FFmpeg vid.stab 陀螺仪稳定（高清素材启用）|
| **音频合并** | 自动保留原始音频 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

还需安装 **FFmpeg**（用于视频编解码和稳定化）：
- Windows: 下载 [ffmpeg](https://www.gstatic.com暗/pkg/win64/)，添加到 PATH
- 或修改 `stages/02_stabilize.py` 中的 `ffmpeg` 路径指向本地安装位置

### 2. 运行处理

```bash
# 单视频处理
python main.py process "输入视频.mp4"

# 使用预设风格
python main.py process "输入视频.mp4" --preset shorts

# 批量处理（处理整个素材目录）
python main.py batch -i "C:/素材目录" -o "C:/输出目录"

# 预览模式（只处理前3秒，快速验证效果）
python main.py process "输入视频.mp4" --preview
```

### 3. 查看输出

输出目录: `output/YYYY-MM-DD/`

## 预设风格

| 预设 | 适用场景 | 说明 |
|------|----------|------|
| `shorts` | 抖音/快手/YouTube Shorts | 竖版 9:16，适合低分辨率手持拍摄 |
| `youtube` | YouTube 横屏版 | 保留原始 16:9，叠加能量条+节拍 |
| `sexy` | 强效体型美化 | 瘦腰+丰满+长腿，适合展示身材 |
| `natural` | 自然微调 | 轻度体型调整，保持自然感 |
| `dramatic` | 电影感调色 | 高对比度+暖色调，增强视觉冲击 |
| `gimbal` | 云台/固定机位拍摄 | 画面稳定，启用 stabilization |
| `beauty` | 多人场景 | 智能识别领操人 + 团队全景切换 |
| `night_gym` | 晚间健身房 | 低光环境优化，降噪+提亮 |
| `clean` | 最小处理 | 仅横转竖裁切，无美颜无运镜 |

## 命令行参数

### 阶段控制

| 参数 | 说明 |
|------|------|
| `--no-stabilize` | 禁用视频稳定 |
| `--no-ken-burns` | 禁用 Ken Burns 运镜 |
| `--no-body-warp` | 禁用体型调整 |
| `--no-color-grade` | 禁用色彩增强 |
| `--no-pose-gpu` | 禁用 GPU 加速（用 CPU）|
| `--full-video` | 生成完整视频（跳过精华片段选取）|
| `--preview` | 预览模式（前3秒）|

### 体型参数

| 参数 | 说明 | 范围 |
|------|------|------|
| `--leg-lengthen` | 腿部拉长 | 1.0~1.4 |
| `--leg-slim` | 腿部瘦身 | 0.7~1.0 |
| `--waist-slim` | 腰部塑形 | 0.7~1.0 |
| `--chest-enlarge` | 胸部丰满 | 1.0~1.3 |
| `--overall-slim` | 整体瘦身 | 0.7~1.0 |
| `--head-ratio` | 头身比调整 | 0.8~1.2 |

### 输出控制

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--crf` | 视频质量 (18-28，越小越大) | 26 |
| `--enc-preset` | 编码速度 | fast |
| `--audio-bitrate` | 音频码率 | 96k |
| `--video-fade-out` | 片尾淡出秒数 | 2.0 |
| `--cut 30-60,120-150` | 裁切重复片段 | - |

### 示例

```bash
# 竖版输出 + 强效瘦身
python main.py process "健身操.mp4" --preset sexy --waist-slim 0.8 --leg-slim 0.85

# 保留原始横版 + 叠加能量条
python main.py process "健身操.mp4" --preset youtube --full-video

# 低分辨率手持拍摄（禁用稳定化，避免放大抖动）
python main.py process "手机拍摄.mp4" --preset shorts --no-stabilize

# 批量处理并切割为45秒片段
python main.py batch -i "素材/" -o "输出/" --segment 45
```

## 配置参考

配置文件为 YAML 格式，可通过 `-c` 指定：

```yaml
stages:
  pose_detect: true
  stabilize: false      # 低分辨率手持拍摄建议关闭
  h2v_convert: true
  ken_burns: false     # 低分辨率建议关闭
  body_warp: true
  beat_flash: true
  energy_bar: true
  color_grade: true

output:
  width: 1080
  height: 1920
  crf: 26
  preset: fast

ken_burns:
  mode: smooth         # smooth=微幅运镜, dual=全景/特写切换
  zoom_range: [1.0, 1.03]

energy_bar:
  width: 16
  margin_right: 20
  height: 400
  smoothing: 0.85
```

## 流水线阶段

```
输入视频
    │
    ▼
01_pose_detect     YOLOv8-pose 人体关键点检测（批量推理）
    │
    ▼
02_stabilize       FFmpeg vid.stab 视频稳定（可选）
    │
    ▼
03_h2v_convert     横转竖裁切，领操人智能跟踪
    │
    ▼
04_ken_burns       Ken Burns 运镜效果（可选）
    │
    ▼
05_body_warp       体型变形美化（可选）
    │
    ▼
06_color_grade     色彩增强
    │
    ▼
07_export          H.264 编码输出 + 音频合并
    │
    ▼
输出视频
```

## 硬件要求

| 组件 | 最低要求 | 推荐 |
|------|----------|------|
| CPU | 4 核 | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| GPU | - | NVIDIA GPU（CUDA）|
| 显存 | - | 4 GB+（YOLO GPU 加速）|
| 存储 | 10 GB | 50 GB+（视频文件大）|

## 文件结构

```
fitness-video-pipeline/
├── main.py              # CLI 入口
├── config.yaml          # 默认配置
├── requirements.txt     # Python 依赖
├── pyproject.toml       # 项目打包配置
├── presets/             # 预设配置
│   ├── shorts.yaml      #   竖版短视频预设
│   ├── youtube.yaml     #   YouTube 横屏预设
│   ├── sexy.yaml        #   强效美化预设
│   ├── natural.yaml      #   自然风格
│   ├── dramatic.yaml     #   电影感
│   ├── gimbal.yaml       #   云台稳定素材
│   ├── beauty.yaml       #   多人场景
│   ├── night_gym.yaml    #   晚间健身房
│   └── clean.yaml        #   最小处理
├── stages/              # 流水线阶段
│   ├── 01_pose_detect.py
│   ├── 02_stabilize.py
│   ├── 03_h2v_convert.py
│   ├── 04_ken_burns.py
│   ├── 05_body_warp.py
│   ├── 06_color_grade.py
│   ├── 07_export.py
│   ├── 09_audio.py
│   ├── 17_beat_flash.py
│   ├── 18_highlight.py
│   ├── 19_energy_bar.py
│   └── ...
├── pipeline/             # 引擎核心
│   ├── engine.py
│   └── config.py
├── lib/                  # 共享库
│   ├── utils.py
│   ├── tracker.py
│   ├── warp.py
│   └── yolo_pose.py
└── output/              # 输出目录（自动创建）
    └── YYYY-MM-DD/
        └── *final*.mp4
```


## 2026-06-27 新增能力

### CLI flag 速查

```bash
# 标准流程 (推荐): 跑 youtube preset → 产出宽屏 + YT Shorts + 抖音
python main.py process input.mp4 --preset youtube

# 不想跑 ShortsStage
python main.py process input.mp4 --preset youtube --no-shorts
python main.py process input.mp4 --preset youtube --no-douyin

# 跑前重置 GPU (解决 onnxruntime CUDNN 碎片化, 长期跑建议加)
python main.py process input.mp4 --preset youtube --reset-gpu

# 修改 Shorts 时长 / 教练名
python main.py process input.mp4 --preset youtube --shorts-duration 60 --shorts-coach 丽丽
```

### 输出文件命名规则（避免传错）

跑 `--preset youtube` 一次性产出 4 个 16:9 / 9:16 文件，**长视频必须传 `final_path`**：

| 模式 | 文件名 | 时长 | 用途 |
|---|---|---|---|
| **final_path** | `*_final_16x9_1920x1080.mp4` | ~105s | **YT 主视频（含片头片尾）** |
| full_16x9 | `*_final_16x9_1920x1080_full_16x9.mp4` | ~96s | 16:9 多格式副本（**不上传**） |
| YT Shorts | `*_final_16x9_1920x1080_yt_shorts.mp4` | 30s | YouTube Shorts |
| 抖音 | `*_final_16x9_1920x1080_douyin.mp4` | ~96s | 抖音（人工传） |

### 换脸必读

`--reset-gpu` 跑前加一个能避免 90% 的 CUDNN 崩溃。如果 face_swap 还是崩，看 `tools/face_swap.py:65` 确认有 `arena_extend_strategy='kSameAsRequested'`。

### YT 上传示例

```python
from lib.upload_utils import upload_pair

upload_pair(
    coach="丽丽",
    long_path=r"D:\shorts_test\2026-06-25\丽丽2_final_16x9_1920x1080.mp4",
    short_path=r"D:\shorts_test\2026-06-25\丽丽2_final_16x9_1920x1080_yt_shorts.mp4",
    record_date="2026-04-20",
    privacy="public",
)
# Manifest 自动写 records/upload_manifest.json
```

## 独立工具 (standalone, 主管线零改动)

除主管线外, `tools/` 下有独立 CLI 工具处理特殊需求:

### bg_swap — 网红视频换背景 + 换脸

把源视频里的人抠出 (RVM 高精度抠像) 合成到新背景 (默认西安时代广场) + 换教练脸. 详见 `docs/BG_SWAP.md`.

```bash
# 健身/操课 (fitness 预设, 含接地感 grounding 0.18)
python tools/bg_swap.py --video 网红.mp4 --bg 时代广场背景.mp4 --coach 丽丽 \
  --output output/bgswap/网红_丽丽_时代广场.mp4 --preset fitness

# 舞蹈 (dance 预设, 视差略增)
python tools/bg_swap.py --video dance.mp4 --bg 时代广场背景.mp4 --coach 丽丽 \
  --output output/bgswap/dance_时代广场.mp4 --preset dance
```

预设 `presets/bgswap_*.yaml`: `fitness` (实测) / `clean` (基线) / `dance` (起步). 加新教练丢 `tools/{coach}.jpg` 即可 (首次自动 GFPGAN 增强).

### prefilter_person — 换背景前清洗 (删人物不完整片段)

换背景前先剪掉人物出画/缺头缺脚片段 (否则抠像残缺 + 贴纸感加重). 用 YOLOv8-pose 逐帧判完整性.

```bash
# 先预览 (只出 timeline, 不渲染)
python tools/prefilter_person.py 网红跳舞1.mp4 --preview
# 渲染清洗后视频 (逐帧精确, 保留音频)
python tools/prefilter_person.py 网红跳舞1.mp4 -o 网红跳舞1_cleaned.mp4 --accurate
```

## 常见问题

**Q: 竖版视频抖动严重？**
A: 低分辨率手持拍摄（<1080p）不建议启用 `--stabilize`，stabilization 会放大抖动。使用 `--preset shorts`（默认已禁用 stabilize）。

**Q: 体型调整效果不明显？**
A: 尝试 `--leg-lengthen 1.2 --waist-slim 0.85` 或使用 `sexy` 预设。

**Q: 处理速度太慢？**
A: 确保安装 `ultralytics` 并使用 NVIDIA GPU（`--no-pose-gpu` 可禁用 GPU 节省显存）。

**Q: 关键点检测失败？**
A: 清理缓存 `rm output/*_keypoints.json` 后重新处理。

## 许可

MIT License

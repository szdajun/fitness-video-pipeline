# 教练精修子流程 — 设计规范与操作手册

> **定位**: 抠人像 → GFPGAN 美颜 → Pipeline 换脸 → SAM2 背景替换  
> **最后更新**: 2026-06-06

---

## 一、需求概述

### 1.1 业务需求

健身短视频需要"教练精修"效果：把教练从原始视频中抠出，用美颜照替换脸部，再将背景替换为专业场景（如时代广场、健身房等），提升视频质感和品牌辨识度。

### 1.2 输入输出

| 阶段 | 输入 | 输出 |
|------|------|------|
| 抠人像 | 教练源视频 (mp4) | `tools/{name}_gfpgan.png` (美颜照) |
| Pipeline 换脸 | 任意源视频 + 美颜照 | `output/*_full_*.mp4` |
| 背景替换 | 换脸视频 + 背景图 | `output/*_bgswap.mp4` |

### 1.3 教练美颜照库存

| 教练 | 文件名 | 大小 |
|------|--------|------|
| 艳青 | `tools/yanqing_face_gfpgan.png` | 327 KB |
| 丽丽 | `tools/lili_gfpgan.png` | 470 KB |
| 建玲 | `tools/jianling_face.jpg` | 904 KB |
| 小红豆 | `tools/xhd_gfpgan.png` | 485 KB |
| 枫林红 | `tools/flh_gfpgan.png` | 406 KB |
| 郭海军 | `tools/haijun_face.jpg` | 255 KB |

---

## 二、架构设计

### 2.1 整体链路

```
┌─ Phase 1: 教练入库 (一次性, 本地 GPU) ───────────────────┐
│                                                          │
│  源视频 → YOLO 人脸检测 → 裁剪最佳脸部帧                   │
│         → GFPGAN 美颜增强 → tools/{name}_gfpgan.png       │
│                                                          │
│  工具: coach_portrait.py                                  │
│  模型: yolov8n-pose.pt + GFPGANv1.4.pth (ComfyUI 目录)    │
│  GPU:  RTX 4070                                           │
│  耗时: ~2 分钟/视频                                       │
└──────────────────────────────────────────────────────────┘

┌─ Phase 2: Pipeline 换脸 (自动, --preset 触发) ──────────┐
│                                                          │
│  source_videos/xxx.mp4                                    │
│       ↓  pose_detect → color_grade → skin_smooth         │
│       ↓  ... → mascot → **face_swap** → danmaku → export │
│                                                          │
│  stages/37_face_swap.py                                   │
│       │  读 COACH_FACE_MAP 获取美颜照                     │
│       │  调 tools/face_swap.py (InsightFace)              │
│       │  改写 mascot_path → 后续 stage 自动用换脸版       │
│       ↓                                                  │
│  output/xxx_full_9x16.mp4  /  xxx_full_16x9.mp4          │
│                                                          │
│  状态: ✅ 已集成, 6 位教练就绪                             │
│  耗时: ~8 分钟 (face_swap 阶段)                           │
└──────────────────────────────────────────────────────────┘

┌─ Phase 3: 背景替换 (--bg-swap 触发) ────────────────────┐
│                                                          │
│  Pipeline 输出 + assets/bg/xxx.jpg (背景图)               │
│       ↓                                                  │
│  tools/bgswap_stable.py (SAM2 + ORB 运镜匹配)             │
│       │  SAM2: 视频前景分割 (三点锚定覆盖全身)            │
│       │  ORB+RANSAC: 背景运动估计 + 稳像                  │
│       │  InsightFace: 换脸 (可选, 无人脸时跳过)           │
│       │  合成: frame×mask + bg×H×(1-mask)                │
│       ↓                                                  │
│  output/xxx_bgswap.mp4                                   │
│                                                          │
│  Python: ComfyUI Python 3.11 (独立环境, SAM2 依赖)        │
│  GPU:  RTX 4070                                          │
│  耗时: ~3 分钟/400 帧                                    │
│  内存: ~5-8 GB RAM                                       │
└──────────────────────────────────────────────────────────┘
```

### 2.2 文件职责

| 文件 | 角色 | 集成状态 |
|------|------|---------|
| `coach_portrait.py` | 抠人像 + GFPGAN 增强 | ⚠️ 独立脚本, 不在 Pipeline |
| `tools/face_swap.py` | InsightFace 换脸核心 | ✅ 被 `stages/37_face_swap.py` 调用 |
| `stages/37_face_swap.py` | Pipeline Stage 封装 | ✅ 注册在 main.py |
| `tools/bgswap_stable.py` | SAM2+ORB 背景替换 | ✅ `main.py --bg-swap` 调用 |
| `tools/sam2_bg_swap.py` | SAM2 背景替换(简化版) | ✅ 作为回退 |
| `tools/bgswap_motion.py` | SAM2 运镜版(另一变体) | ⚠️ 未集成 |
| `main.py::_run_bgswap()` | bgswap CLI 入口 | ✅ 已集成 |
| `assets/bg/` | 背景图库存放 | ✅ 已创建 |

### 2.3 Python 环境分离

| 环境 | Python | 用途 |
|------|--------|------|
| 主 Pipeline | 3.9.13 | 所有 Stage + face_swap |
| ComfyUI | 3.11.x | SAM2 (需要 torch>=2.0) |

`main.py` 的 `_run_bgswap()` 通过 `subprocess.run()` 启动 ComfyUI Python 子进程，两个环境独立运行。

---

## 三、Phase 1 详解：教练入库

### 3.1 抠人像 (`coach_portrait.py`)

```bash
# 单视频
python coach_portrait.py --video "source_videos/xxx.mp4" --coach 教练名

# 批量 (扫描桌面短视频素材目录)
python coach_portrait.py --batch
```

**流程**:
1. YOLOv8-pose 检测视频帧中的人体 (每 3 秒采样一帧)
2. 取人体框上半 1/3 作为脸部区域 → 裁剪
3. 按评分排序, 取 Top 3 张
4. 保存原始裁剪: `coach_portraits/{name}_N_raw.jpg`
5. GFPGAN 增强 → `coach_portraits/{name}_N.jpg`

### 3.2 GFPGAN 美颜增强

**本地 GPU 执行** (RTX 4070):
- 模型: `F:\wkspace\ComfyUI\models\gfpgan\GFPGANv1.4.pth`
- monkey-patch: `torchvision.transforms.functional_tensor` 兼容性修复
- 参数: upscale=2, arch='clean', channel_multiplier=2

**关键兼容性问题**: basicsr 1.0+ 与 torchvision 0.19+ 不兼容 (移除了 `functional_tensor` 模块)。`coach_portrait.py:111-115` 用 monkey-patch 创建虚拟模块解决。

### 3.3 入库步骤

```
1. python coach_portrait.py --video "xxx.mp4" --coach 教练名
2. 选择最佳的一张复制为: tools/{教练名}_gfpgan.png
3. 在 stages/37_face_swap.py 的 COACH_FACE_MAP 加入映射
```

---

## 四、Phase 3 详解：背景替换

### 4.1 SAM2 前景分割

**三点锚定策略** (覆盖全身, 不遗漏):

```python
# 头 (15% 高处, 正标签)
# 胯 (55% 高处, 正标签)  
# 脚底 (h-5, 负标签 = 地面)
points = [[cx, h*0.15], [cx, h*0.55], [cx, h-5]]
labels = [1, 1, 0]  # 正/正/负
```

**为什么用固定锚点而非人脸检测**: 健身视频中人物较小、面部模糊, InsightFace 经常检测不到人脸。固定锚点覆盖全身, 鲁棒性更好。

### 4.2 ORB 运镜匹配

`bgswap_stable.py` 使用 ORB 特征 + RANSAC 单应性估计摄像机运动:

1. SAM2 生成前景 mask → 排除人体区域
2. 背景区域检测 ORB 特征点
3. RANSAC 拟合单应性矩阵 (H)
4. 累积 H 矩阵驱动背景图 → 人景同步不滑

**对比**: `sam2_bg_swap.py` 只做静态背景粘贴, 无运镜补偿, 画面会"滑动"。 `bgswap_stable.py` 是推荐版本。

### 4.3 帧数限制与内存管理

```
帧数 < 400: 全帧加载, 步长=1, 30fps 输出 (流畅)
帧数 > 400: 取样 400 帧, 步长自动计算, 帧率等比降低 (防 OOM)
```

**内存瓶颈**: SAM2 对 JPEG 目录调用 `load_video_frames_from_jpg_images`, 一次性全量加载为 CPU 张量。400 帧 × 1080×1920 ≈ 5GB RAM。

### 4.4 预热与开头保护

SAM2 mask 传播前 10 帧不稳定, `bgswap_stable.py:164-171`:
```python
start_fade = 10
if fi < start_fade:
    # 直接输出原帧, 不换背景 (保头)
    pass
else:
    # 正常背景替换
```

---

## 五、关键问题与解决方案

### 5.1 "只看到身体中间, 没有头没有脚"

**原因**: SAM2 锚点范围太窄 (基于人脸 bbox, 远距镜头人脸 bbox 很小 → 只覆盖腰)。

**解决**: 改用固定锚点: 头(15%高) + 胯(55%高) + 脚底(h-5), 覆盖全身。

### 5.2 "运动节奏很快, 不正常"

**原因**: 帧数采样 (步长 > 1) 但输出帧率不变 → 加速播放。

**解决**: 步长采样时等比降低输出帧率 (`out_fps = fps / step`)。同时提高帧数上限 (400 帧), 减少采样。

### 5.3 "视频开始很短时间看不到人头"

**原因**: SAM2 mask 传播前几帧未收敛, 遮罩不稳定。

**解决**: 前 10 帧不换背景, 直接输出原帧 (`start_fade = 10`)。

### 5.4 "SAM2 内存不足 (12GB OOM)"

**原因**: `load_video_frames_from_jpg_images` 一次性全量加载所有帧为 CPU 张量。

**解决**: 限制帧数上限 400 帧 (`limit = min(400, total)`)。如需全片, 分批处理。

### 5.5 "bgswap 输出是 16:9 裁切版, 人物被切"

**原因**: `_run_bgswap()` 传了 `final_path` (export 已裁切为 16:9)。

**解决**: 改用 `mascot_path` 或源视频 (未裁切), 保持原始 9:16 比例。

### 5.6 "ComfyUI Python 找不到 CUDA"

**原因**: ComfyUI 的 onnxruntime 可能只装了 CPU 版。

**解决**: ComfyUI Python 环境安装 `onnxruntime-gpu`, 或接受 CPU 回退 (速度慢但仍可用)。

### 5.7 "GFPGAN basicsr/torchvision 兼容性"

**原因**: `basicsr` 依赖 `torchvision.transforms.functional_tensor`, torchvision 0.19+ 已移除该模块。

**解决**: `coach_portrait.py` monkey-patch 创建虚拟 `functional_tensor` 模块, 提供 `rgb_to_grayscale` 函数。

---

## 六、避坑指南 (红线)

| # | 红线 | 原因 |
|---|------|------|
| 1 | **bgswap 不要用 `final_path`** | export 已裁切, 人物不完整 |
| 2 | **不要移除 start_fade** | SAM2 预热不足 → 开头人头消失 |
| 3 | **帧数不要超过 400** | SAM2 JPEG 目录加载全量 OOM |
| 4 | **不要用 yuv444p 给 FFmpeg 8.1** | Gyan build 兼容性问题, 用 yuv420p |
| 5 | **GFPGAN 不要装 basicsr 最新版** | torchvision 兼容性, 用现有 monkey-patch |
| 6 | **SAM2 锚点不要只用脸部 bbox** | 远距镜头脸小 → 只覆盖腰 |
| 7 | **bgswap 输出帧率必须和输入帧率匹配** | 否则加速/减速播放 |
| 8 | **ComfyUI Python 环境不要混用** | SAM2 需要 Python 3.11+, Pipeline 用 3.9 |

---

## 七、CLI 命令参考

### 7.1 教练入库

```bash
# 抠人像 + GFPGAN 美颜
python coach_portrait.py --video "source_videos/枫林红11.mp4" --coach 枫林红

# 批量处理
python coach_portrait.py --batch
```

### 7.2 Pipeline 换脸

```bash
# 抖音 (9:16, 简洁)
python main.py process "source_videos/xxx.mp4" --preset douyin --full-video

# 通用 (16:9, 全装饰+换脸)
python main.py process "source_videos/xxx.mp4" --preset shorts --full-video
```

### 7.3 背景替换

```bash
# 一键 (Pipeline + bgswap)
python main.py process "source_videos/xxx.mp4" --preset shorts --bg-swap \
    --bg-image assets/bg/时代广场.jpg --bg-coach 丽丽

# 单独 bgswap (已有视频)
F:/wkspace/ComfyUI/venv/Scripts/python.exe tools/bgswap_stable.py \
    --target "output/xxx_full_16x9.mp4" \
    --bg "assets/bg/时代广场.jpg" \
    --face "tools/lili_gfpgan.png" \
    --output "output/xxx_bgswap.mp4"
```

---

## 八、环境依赖清单

| 组件 | 路径/包 | 大小 | 用途 |
|------|---------|------|------|
| YOLOv8-pose | `ultralytics` (pip) | ~200MB | 抠图人脸检测 |
| GFPGAN | `gfpgan` (pip 1.3.8) + `GFPGANv1.4.pth` | ~340MB | 人脸美颜 |
| InsightFace | `insightface` (pip) | ~500MB | 换脸+人脸检测 |
| inswapper | `~/.insightface/models/inswapper_128.onnx` | ~500MB | 换脸模型 |
| SAM2 | `F:/wkspace/sam2/` (源码) + `sam2_hiera_small.pt` | ~2.4GB | 视频分割 |
| ComfyUI Python | `F:/wkspace/ComfyUI/venv/Scripts/python.exe` | - | SAM2 运行环境 (3.11+) |
| FFmpeg | `C:/Users/18091/ffmpeg/ffmpeg.exe` (8.1 Gyan) | ~100MB | 视频编解码 |
| GPU | RTX 4070 | - | 全部 GPU 加速 |

---

> **维护提醒**: Phase 2 (face_swap) 和 Phase 3 (bgswap) 的 stage 顺序不可动。  
> Phase 3 依赖 ComfyUI Python 3.11 环境, 不可合并到主 Pipeline 的 Python 3.9。
> 六位教练美颜照已在 `tools/` 目录, 新增教练参考 §3.3 入库步骤。

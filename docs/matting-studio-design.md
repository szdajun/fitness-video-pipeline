# Matting Studio — 自研视频抠像软件技术设计

> **状态**: Phase 0 设计文档 (2026-07-04 立项). 配套 memory `cn-video-matting-software-architecture.md` 8 模块 + 8 模型蓝图.
> **目标**: 健身/网红短视频自动抠像换背景, 开源 GitHub, Apache 2.0.

---

## 0. 30 秒读懂

**问题**: 当前 `tools/bg_swap.py` 多轮迭代治了胳膊渗出 + 软抠抖动, 但**结构性 RVM 软抠天花板** (RVM 对细胳膊 α<0.5 + 远处真人幻觉 + 复制人叠加鬼影) 仍存在. 单人美女跳舞 `网红跳舞1.mp4` t=90 帧 RVM mask 也有 2075 个低 α 噪点 → 视觉上读成"半透人形".

**解法**: **集成 SAM2 互动修帧** (用户可手动修穿帮帧) + **多模型组合** (RVM 主抠像 + YOLOv8 二确认治鬼影 + 静态背景 + 简单合成) + **开源软件** (社区贡献 + 用户定制).

**MVP 估时**: 8 人月 (单人全职). MVP Phase 1 (CLI 工具) = 2-3 人月, Phase 2 (UI + SAM2 修帧) = 2-3 人月.

---

## 1. 目标 & 范围

### 1.1 目标场景 (用户拍板)

**Phase 0-2 目标**: 健身/网红短视频自动抠像换背景, **单人 + 多人**, **1080p 30fps**, **MP4 输出**.

**Phase 3+ 扩展**: 直播实时 (OBS 插件) + 影视后期高精度 + Web 端 (TFLite + WebGPU) + 移动端 (TFLite).

### 1.2 核心差异化 (vs 竞品)

| 差异化 | 现有工具 | Matting Studio |
|--------|---------|---------------|
| **SAM2 互动修帧** | 剪映只有自动抠像, 穿帮帧无救 | 鼠标点选/框选修正, 影视级工作流 |
| **健身场景优化** | 通用工具多人 = 鬼影 | YOLOv8 治 RVM 幻觉 (本项目已验证) + D+grow 治细胳膊 |
| **开源 + 可定制** | 商业闭源 (Runway, Remove.bg) | Apache 2.0, 用户改算法加模型 |
| **集成换脸 + 抠像** | 各自独立 | 主管线 (1 个 Pipeline 跑完) |

---

## 2. 技术选型

### 2.1 模型选型 (基于 8 模型蓝图 + 本项目验证)

| 角色 | 选型 | 协议 | 验证状态 |
|------|------|------|---------|
| **AI 抠像主模型** | RVM (RobustVideoMatting) | BSD-3-Clause | 本项目已用, 8G GPU 30-100 FPS |
| **鬼影治本** | YOLOv8n-seg (人物实例分割) | AGPL-3.0 ⚠ | 本项目已验证治 RVM 远处半透真人 (memory `bg-swap-core-matte-arm-bleed`) |
| **互动修帧** | SAM2 (Segment Anything v2) | Apache 2.0 (Meta) | 影视级工作流必备, 模型大 ~2GB |
| **离线高精度 (可选)** | MatAnyone (CVPR2025) | 学术免费 | memory `matanyone-ab-test-negative` 验证 1080p 无优势, 跳过 |
| **绿幕备选** | OpenCV chroma key (HSV/UV) | Apache 2.0 | 直播/绿幕场景备选, 不主推 |
| **时序优化** | 3-5 帧 Alpha 中值滤波 | 自实现 | 简单有效, 治 RVM 抖动 |

**避开**:
- **MatAnyone** = 算力高 (4K) + 阴性验证, 跳过
- **VideoMaMa / SAM3** = 影视级, 超本项目需求, Phase 3+ 考虑
- **MODNet / MediaPipe Selfie Segmentation** = 精度有限, 备选
- **BiRefNet** = 静态图像 SOTA, 动态视频需时序后处理, 备选

### 2.2 技术栈

| 层 | 选型 | 理由 |
|------|------|------|
| **核心语言** | Python 3.11 | 已有 .venv + uv 依赖管理 |
| **AI 推理** | PyTorch (RVM) + ONNX Runtime (YOLOv8) | 主管线已用, 性能稳定 |
| **SAM2** | segment-anything-2 (Meta) | Apache 2.0, 兼容 PyTorch |
| **图像处理** | OpenCV 4.11 + Pillow + scipy.ndimage | 标准栈 |
| **UI** | PyQt6 (桌面) | 跨平台, 互动画布支持 SAM2 点选 |
| **视频** | FFmpeg (含 nvenc/QSV 硬件加速) | 主管线已用, `_resolve_ffmpeg()` 已知好路径 |
| **配置** | YAML + pydantic | 主管线已用 |
| **测试** | pytest + 性能基准 | 主管线已用 110 测试 |
| **打包** | PyInstaller (CLI) + py2app (macOS) + AppImage (Linux) | 跨平台分发 |
| **CI/CD** | GitHub Actions (lint + test + build) | 开源项目标准 |
| **协议** | Apache 2.0 | 健身/网红含算法专利, 商业友好 = 用户增长 |

### 2.3 硬件需求

| 配置 | 最低 | 推荐 | 备注 |
|------|------|------|------|
| GPU | GTX 1060 (6GB) | RTX 4070 (12GB) | RVM 需 ~2GB + YOLO 需 ~4GB + ONNX arena 4GB = 共需 ~10GB |
| RAM | 16GB | 32GB | Windows 进程 OOM 已知 (memory `export-nvenc-probe-fails-after-heavy-gpu`), 长视频分片 |
| CPU | i5 | i7 | 视频解码 + UI 响应 |
| 存储 | 1GB (CLI) + 模型权重 ~500MB | SSD 推荐 | git-lfs 权重 |

---

## 3. 8 模块架构

```
video-matting-studio/
├── README.md                  # 快速开始
├── LICENSE                     # Apache 2.0
├── pyproject.toml              # Python 依赖
├── core/                       # 核心引擎
│   ├── engine.py               # Pipeline + Stage 框架
│   ├── context.py              # PipelineContext 状态传递
│   ├── config.py               # YAML 配置 + 4 方案预设
│   └── types.py                # Frame / Mask / Background 数据类
├── modules/                    # 8 大模块
│   ├── input.py                # 01 视频采集 (FFmpeg + WebRTC 直播预留)
│   ├── preprocess.py           # 02 帧预处理
│   ├── matting.py              # 03 AI 抠像 (RVM + YOLO 二确认 + 色度备份)
│   ├── postprocess.py          # 04 时序后处理 (中值 + SAM2 修)
│   ├── compose.py              # 05 合成渲染
│   ├── export.py               # 06 编码输出
│   ├── ui.py                   # 07 PyQt6 桌面 UI (含 SAM2 修帧交互)
│   └── scheduler.py            # 08 多线程 + GPU 显存管理
├── models/                     # 模型权重 (git-lfs)
│   ├── rvm/                    # rvm_mobilenetv3_fp32.onnx
│   ├── sam2/                   # sam2_hiera_large.pt
│   └── yolov8_seg/             # yolov8n-seg.pt
├── tests/                      # 110+ 守门测试
│   ├── test_rvm.py
│   ├── test_sam2_integration.py
│   ├── test_yolo_ghost.py
│   ├── test_sam2_repair.py
│   ├── test_compose.py
│   ├── test_performance.py     # 性能基准
│   └── test_e2e.py             # 端到端 4 场景
├── docs/
│   ├── matting-studio-design.md   # 本文档
│   ├── architecture.md            # 架构图
│   ├── algorithms.md              # 算法细节
│   ├── api.md                     # API 文档
│   ├── user-guide.md              # 用户手册
│   └── contributing.md            # 贡献指南
├── presets/                    # 4 方案预设
│   ├── single_person.yaml
│   ├── multi_person.yaml
│   ├── livestream.yaml
│   └── pro_filmmaking.yaml
├── examples/                   # 示例视频 + README
├── .github/
│   ├── workflows/
│   │   ├── ci.yml             # lint + test
│   │   └── release.yml         # 自动打包
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
└── tools/                      # CLI 工具
    ├── matting_studio.py      # 主入口
    └── sam2_repair.py         # SAM2 修帧工具
```

---

## 4. 模块 API 草案 (Python 接口)

### 4.1 核心引擎 `core/engine.py`

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import numpy as np

class StageMode(Enum):
    SEQUENTIAL = "sequential"   # 串行
    PARALLEL = "parallel"       # 并行 (GPU/CPU 分流)
    SKIP = "skip"               # 跳过

@dataclass
class Frame:
    data: np.ndarray          # (H, W, 3) BGR uint8
    timestamp_ms: int
    frame_idx: int
    metadata: dict             # 自定义

@dataclass
class Mask:
    alpha: np.ndarray         # (H, W) float32 [0, 1]
    foreground_rgb: np.ndarray # (H, W, 3) uint8 (RVM 输出)
    confidence: float          # 0-1, RVM 内部置信度
    frame_idx: int
    timestamp_ms: int

@dataclass
class PipelineContext:
    frames: List[Frame]            # 输入帧
    masks: List[Mask]              # 抠像结果
    background: Optional[np.ndarray]  # (H, W, 3) 静态背景
    config: dict                   # 配置
    state: dict                    # 跨 stage 状态 (e.g. last_mask, sam2_repairs)
    gpu_memory_mb: int             # GPU 显存管理

class Stage:
    """所有模块继承 Stage, 重写 process()."""
    name: str = "stage"
    mode: StageMode = StageMode.SEQUENTIAL

    def process(self, ctx: PipelineContext) -> PipelineContext:
        raise NotImplementedError

class Pipeline:
    def __init__(self, stages: List[Stage], config: dict):
        self.stages = stages
        self.config = config

    def run(self, input_path: str, output_path: str) -> dict:
        """主入口, 返回统计 {frames, swap_count, ...}"""
        ctx = self._load(input_path)
        for stage in self.stages:
            ctx = stage.process(ctx)
        stats = self._export(ctx, output_path)
        return stats
```

### 4.2 8 模块接口

```python
# 01 input
class VideoInputStage(Stage):
    """加载视频到 ctx.frames (FFmpeg 解码, JPEG 序列, 或 MP4 流)."""
    def __init__(self, source: str, fps: int = 30, max_frames: Optional[int] = None):
        self.source = source

# 02 preprocess
class PreprocessStage(Stage):
    """帧预处理: 缩放到模型输入, 归一化, 去噪, CUDA 加速."""
    def __init__(self, target_size: int = 512, denoise: bool = True):

# 03 matting (核心)
class MattingStage(Stage):
    """AI 抠像: RVM 主 + YOLO 二确认 + 可选色度备份.
    输出: ctx.masks (alpha + foreground RGB + confidence)."""
    def __init__(self, model='rvm', use_yolo_ghost_filter: bool = True, device='cuda'):
        self.model = model  # 'rvm' | 'chromakey' | 'yolov8_seg'
        self.use_yolo_ghost_filter = use_yolo_ghost_filter  # 治 RVM 远处半透真人
        self.device = device

# 04 postprocess (含 SAM2 修帧)
class PostprocessStage(Stage):
    """时序后处理: 3-5 帧中值滤波 + SAM2 互动修帧 (用户手动点选)."""
    def __init__(self, temporal_window: int = 3, enable_sam2_repair: bool = True):
        self.temporal_window = temporal_window
        self.enable_sam2_repair = enable_sam2_repair
        self.sam2_checkpoint = "models/sam2/sam2_hiera_large.pt"

# 05 compose
class ComposeStage(Stage):
    """合成: alpha 混合 + 边缘羽化 + 阴影 + 色温匹配."""
    def __init__(self, edge_feather: int = 11, despill: float = 0.0, shadow: float = 0.0):

# 06 export
class ExportStage(Stage):
    """编码输出: FFmpeg (h264_nvenc / h264_qsv / libx264) + 音频."""
    def __init__(self, encoder: str = 'h264_nvenc', bitrate: str = '10M', audio: bool = True):

# 07 UI (PyQt6)
class StudioMainWindow(QMainWindow):
    """主窗口: 时间轴 + 预览 + SAM2 修帧画布 + 参数面板."""
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.sam2_canvas = SAM2RepairCanvas()  # 鼠标点选/框选

# 08 scheduler
class GPUScheduler:
    """显存管理: 监控 + 自动降级 (切 CPU / 降精度)."""
    def __init__(self, max_memory_mb: int = 10000):
```

### 4.3 配置 YAML

```yaml
# presets/single_person.yaml
name: single_person
description: 单人短视频抠像, 1080p 30fps

stages:
  input:
    source: ${INPUT}
    fps: 30
  preprocess:
    target_size: 512
    denoise: true
  matting:
    model: rvm
    use_yolo_ghost_filter: true  # 治 RVM 远处半透真人
  postprocess:
    temporal_window: 3
    enable_sam2_repair: true
    sam2_checkpoint: models/sam2/sam2_hiera_large.pt
  compose:
    edge_feather: 11
    despill: 0.0
  export:
    encoder: h264_nvenc
    bitrate: 10M
    audio: true

# presets/multi_person.yaml 类似, matting.use_yolo_ghost_filter=true 必开
```

---

## 5. 关键算法伪代码

### 5.1 RVM + YOLO 鬼影治本 (memory `bg-swap-core-matte-arm-bleed` 验证)

```python
# modules/matting.py - MattingStage.process()
def process(self, ctx: PipelineContext) -> PipelineContext:
    for frame in ctx.frames:
        # 1. RVM 主抠像
        rvm_alpha = self.rvm_model.alpha(frame.data)  # (H, W) float32
        rvm_foreground = rvm_alpha[:, :, None] * frame.data  # 前景 RGB

        # 2. YOLOv8 二确认 (治 RVM 远处幻觉真人)
        if self.use_yolo_ghost_filter:
            yolo_mask = self.yolo_model(frame.data, classes=[0])  # person mask (H, W) float32
            # 交集: RVM α × YOLO person mask
            # key: YOLO 边缘锐利 → 剔除 RVM 远处幻觉; RVM 内容丰富 → 平滑 YOLO 锯齿
            final_alpha = rvm_alpha * yolo_mask
        else:
            final_alpha = rvm_alpha

        # 3. 备用: YOLO 漏检时回退纯 RVM (避免误删前景)
        if final_alpha.sum() < 1000:  # < 1000 前景像素
            final_alpha = rvm_alpha
            ctx.state['yolo_miss_count'] = ctx.state.get('yolo_miss_count', 0) + 1

        # 4. 输出 mask
        mask = Mask(
            alpha=final_alpha,
            foreground_rgb=rvm_foreground.astype(np.uint8),
            confidence=final_alpha.mean(),  # 简单置信度
            frame_idx=frame.frame_idx,
            timestamp_ms=frame.timestamp_ms
        )
        ctx.masks.append(mask)

    return ctx
```

### 5.2 SAM2 互动修帧 (新功能, 影视级)

```python
# modules/postprocess.py - PostprocessStage.process()
def process(self, ctx: PipelineContext) -> PipelineContext:
    # 1. 时序中值滤波 (治 RVM 抖动)
    if self.temporal_window > 1:
        ctx = self._temporal_median(ctx, window=self.temporal_window)

    # 2. SAM2 互动修帧 (用户手动)
    if self.enable_sam2_repair:
        repairs = ctx.state.get('sam2_repairs', [])  # 用户在 UI 点选
        for repair in repairs:
            frame_idx = repair.frame_idx
            # SAM2 点选/框选 → mask
            sam2_mask = self.sam2_model.predict(
                point_coords=repair.point_coords,
                point_labels=repair.point_labels,
                box=repair.box,
                image=ctx.frames[frame_idx].data
            )
            # 用 SAM2 mask 替换 RVM mask (或交集)
            if repair.mode == 'replace':
                ctx.masks[frame_idx].alpha = sam2_mask.astype(np.float32)
            elif repair.mode == 'intersect':
                ctx.masks[frame_idx].alpha = ctx.masks[frame_idx].alpha * sam2_mask

    return ctx

def _temporal_median(self, ctx, window=3):
    """3 帧 Alpha 中值滤波, 治 RVM 帧间抖动."""
    for i in range(len(ctx.masks)):
        start = max(0, i - window // 2)
        end = min(len(ctx.masks), i + window // 2 + 1)
        alphas = [ctx.masks[j].alpha for j in range(start, end)]
        ctx.masks[i].alpha = np.median(alphas, axis=0).astype(np.float32)
    return ctx
```

### 5.3 合成渲染 (5 模块, 含边缘羽化 + 色温匹配)

```python
# modules/compose.py - ComposeStage.process()
def process(self, ctx: PipelineContext) -> PipelineContext:
    background = ctx.background
    if background is None:
        return ctx  # 无背景 = 只输出 mask 视频

    for i, (frame, mask) in enumerate(zip(ctx.frames, ctx.masks)):
        # 1. 边缘羽化 (3-11 像素)
        alpha_feathered = cv2.GaussianBlur(
            mask.alpha, (self.edge_feather | 1, self.edge_feather | 1), 0
        )

        # 2. Alpha 混合
        m3 = alpha_feathered[:, :, None]
        composed = (
            mask.foreground_rgb.astype(np.float32) * m3 +
            background.astype(np.float32) * (1.0 - m3)
        ).astype(np.uint8)

        # 3. 溢出抑制 (despill, 可选, 绿幕用)
        if self.despill > 0:
            composed = self._despill_to_bg(composed, alpha_feathered, background, self.despill)

        ctx.frames[i].data = composed

    return ctx
```

### 5.4 YOLO 治鬼影 (本项目 `bg-swap-core-matte-arm-bleed` 验证, 99.8% 治愈 / 2.5% halo)

```python
# modules/matting.py - YOLOGhostFilter
class YOLOGhostFilter:
    """RVM 在动作大时幻觉远处半透真人 = 视觉读成'鬼影'.
    治法: RVM α × YOLO person mask = YOLO 锐利边缘 + RVM 内容丰富.
    测试: 4 个新版本 (含 d_grow1) + 1 个旧 dance 全都有鬼影 (RVM 2075 个低 α 噪点)."""
    def __init__(self, model_path='yolov8n-seg.pt', device='cpu'):  # CPU 避 4GB onnx arena
        from ultralytics import YOLO
        self.model = YOLO(model_path).to(device)

    def filter(self, frame: np.ndarray, rvm_alpha: np.ndarray) -> np.ndarray:
        # YOLO person 检测
        result = self.model(frame, verbose=False, classes=[0])[0]
        if result.masks is None:
            return rvm_alpha  # 漏检 → fallback

        # 合并所有 person mask
        yolo_mask = np.zeros_like(rvm_alpha)
        for m in result.masks.data:
            m_full = cv2.resize(m.cpu().numpy(), (frame.shape[1], frame.shape[0]))
            yolo_mask = np.maximum(yolo_mask, m_full)

        # 交集
        return rvm_alpha * yolo_mask
```

---

## 6. 实施路线图 (TODO 阶段)

### Phase 0: 设计文档 (1-2 周) ✅ 当前

- [x] memory `cn-video-matting-software-architecture.md` 蓝图
- [x] 本文档 `docs/matting-studio-design.md`
- [ ] 架构图 (mermaid, 8 模块流程图)
- [ ] 算法细节 (`docs/algorithms.md` 论文级)
- [ ] API 文档 (`docs/api.md` 自动生成)
- [ ] 用户手册 (`docs/user-guide.md`)

### Phase 1: MVP CLI 工具 (2-3 月, 1 人)

**Week 1-2: 基础设施**
- [ ] 项目脚手架 (新 repo, 8 模块目录, pyproject.toml, CI)
- [ ] core/engine.py Pipeline 框架
- [ ] core/config.py YAML 配置 + 4 方案预设
- [ ] modules/input.py FFmpeg 视频采集
- [ ] modules/export.py FFmpeg 编码输出
- [ ] tests/test_pipeline.py 基础测试

**Week 3-4: AI 抠像核心**
- [ ] modules/matting.py RVM 集成 (复用主管线 RVM 经验)
- [ ] modules/matting.py YOLOv8-seg 集成 (治鬼影)
- [ ] modules/postprocess.py 时序中值滤波
- [ ] tests/test_rvm.py + test_yolo_ghost.py (110+ 测试)

**Week 5-6: 合成 + 端到端**
- [ ] modules/preprocess.py 帧预处理
- [ ] modules/compose.py 合成 + 边缘羽化
- [ ] tools/matting_studio.py CLI 入口
- [ ] 4 预设 YAML (single/multi/livestream/pro)
- [ ] tests/test_e2e.py 4 场景端到端

**Week 7-8: 文档 + 发布**
- [ ] README + 用户手册
- [ ] CI/CD (lint + test + build)
- [ ] v0.1.0 GitHub release
- [ ] 处理 5 个 GitHub issues (来自主管线 bg_swap 已知问题)

**Phase 1 验收**:
- 8 模块全实现 (CLI 模式)
- 110+ 测试零回归
- 单人 + 多人场景鬼影治本 (YOLO 集成)
- 用户能跑通单人 10s + 多人 120s 视频

### Phase 2: UI + SAM2 修帧 (2-3 月, 1 人)

**Week 9-12: PyQt6 UI**
- [ ] modules/ui.py 主窗口 (时间轴 + 预览)
- [ ] SAM2 修帧交互画布 (鼠标点选/框选)
- [ ] 参数面板 (4 方案预设切换)
- [ ] 进度条 + 暂停/继续
- [ ] 跨平台打包 (PyInstaller + AppImage)

**Week 13-16: SAM2 集成 + 高级功能**
- [ ] SAM2 模型下载 + 推理优化
- [ ] SAM2 修帧 UI 工作流
- [ ] 多模型动态切换 (RVM 实时 / MatAnyone 离线)
- [ ] 性能优化 (模型量化 / TensorRT)

**Phase 2 验收**:
- PyQt6 桌面 UI 完整
- SAM2 修帧工作流
- v1.0.0 GitHub release (含 GUI 安装包)
- 用户能修穿帮帧, 不需懂代码

### Phase 3: 社区化 + 扩展 (持续)

- [ ] 文档/教程 (B 站视频)
- [ ] 用户反馈 + 迭代
- [ ] Web 端 (TFLite + WebGPU) - Phase 3.1
- [ ] 移动端 (iOS/Android) - Phase 3.2
- [ ] SaaS 化 (云端 API) - Phase 3.3 (商业版)

### 总计

| Phase | 时长 | 人月 | 验收 |
|-------|------|------|------|
| Phase 0 | 1-2 周 | 0.5 | 本文档 |
| Phase 1 | 2-3 月 | 2-3 | CLI v0.1.0 |
| Phase 2 | 2-3 月 | 2-3 | GUI v1.0.0 |
| Phase 3 | 持续 | TBD | 社区化 + 扩展 |
| **合计** | **8-12 月** | **8-10** | **可商业化开源产品** |

---

## 7. 关键风险

| 风险 | 等级 | 缓解 |
|------|------|------|
| RVM 软抠天花板 | 高 | SAM2 修帧兜底 (用户可手动修) + 多模型动态切换 |
| YOLOv8 AGPL 协议传染 | 中 | 换 RT-DETR (Apache 2.0) 或 YOLOv5 (GPL-3) |
| SAM2 模型大 (~2GB) | 低 | 模型按需下载 + git-lfs + 提供 lite 版 |
| 单人开发周期长 (8 月) | 中 | Phase 1 MVP 先出, 社区贡献加速 |
| 健身复制人场景 (本次 bg_swap 失败根因) | 中 | YOLO 治 + SAM2 修 + 用户手动 |
| Windows 长视频 OOM | 中 | 分片渲染 (start_frame/end_frame) + 显存监控 |
| 竞品压力 (剪映 / Runway) | 高 | 差异化 (SAM2 修 + 健身优化 + 开源) |

---

## 8. 关键决策记录 (待用户拍板)

| 决策 | 状态 |
|------|------|
| 目标场景: 健身/网红短视频 | ✅ 用户拍板 |
| MVP: 完整 8 模块一次性 | ✅ 用户拍板 |
| 模式: 开源 GitHub | ✅ 用户拍板 |
| 协议: Apache 2.0 | 待拍板 (建议) |
| 主模型: RVM + YOLOv8 + SAM2 | ✅ 基于本项目验证 |
| UI 框架: PyQt6 | 待拍板 (建议) |
| 硬件最低: GTX 1060 (6GB) | 待拍板 |
| Phase 1 估时: 2-3 人月 (CLI 工具) | 待拍板 |
| Phase 2 估时: 2-3 人月 (UI + SAM2) | 待拍板 |
| 文档深度: 架构图 + API + 伪代码 + TODO | ✅ 用户拍板 (本文档) |

---

## 9. 验收 (Definition of Done)

**Phase 0** (本文档) ✅
- [x] 8 模块蓝图
- [x] 关键算法伪代码
- [x] 实施路线图
- [x] 风险评估
- [ ] 架构图 (mermaid)
- [ ] 算法细节文档 (`docs/algorithms.md`)

**Phase 1** (CLI v0.1.0)
- [ ] 8 模块全实现
- [ ] 110+ 测试零回归
- [ ] 单人 + 多人场景鬼影治本
- [ ] 4 预设 YAML
- [ ] GitHub release

**Phase 2** (GUI v1.0.0)
- [ ] PyQt6 桌面 UI
- [ ] SAM2 修帧工作流
- [ ] 跨平台打包
- [ ] 用户能修穿帮帧

---

**下一步** (你拍板):
1. **写架构图** (mermaid, 8 模块流程图, 估 1-2 小时) 落 `docs/architecture.md`
2. **写算法细节** (`docs/algorithms.md`, RVM/YOLO/SAM2 论文级, 估 3-4 小时)
3. **创建新 GitHub repo** (本地先建 `F:\wkspace\matting-studio\`, 估 0.5 小时脚手架)
4. **回到主管线继续 bg_swap 相关** (你有其他任务)
5. **暂不动, 等以后再开始**

我建议: **(1) 写架构图 + 算法细节 = 完整 Phase 0 文档** (5-6 小时), 然后 (3) 创建新 repo 脚手架, 准备 Phase 1 编码。

要继续？

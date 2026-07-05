# Matting Studio — 架构图

> **配套**: `matting-studio-design.md` (设计文档), `algorithms.md` (算法细节)
> **状态**: Phase 0 设计产物 (2026-07-04)
> **技术栈**: Python 3.11 + PyTorch + ONNX Runtime + OpenCV + PyQt6 + FFmpeg

---

## 1. 系统架构 (System Architecture)

### 1.1 顶层流程图

```mermaid
flowchart LR
    subgraph Input["01 视频采集"]
        A[视频文件/直播流] --> B[FFmpeg 解码]
        B --> C[Frame 列表]
    end

    subgraph Preprocess["02 帧预处理"]
        C --> D[缩放 512×512]
        D --> E[归一化 0-1]
        E --> F[去噪/亮度均衡]
    end

    subgraph Matting["03 AI 抠像 (核心)"]
        F --> G1[RVM 主模型]
        F --> G2[YOLOv8-seg 二确认]
        G1 --> H[交集 = RVM α × YOLO mask]
        G2 --> H
        H --> I[Mask: alpha + foreground_rgb]
    end

    subgraph Postprocess["04 时序后处理"]
        I --> J1[3-5 帧 Alpha 中值]
        J1 --> J2[SAM2 互动修帧]
        J2 --> K[精炼 Mask]
    end

    subgraph Compose["05 合成渲染"]
        K --> L[Alpha 混合]
        L --> M[边缘羽化]
        M --> N[输出帧]
    end

    subgraph Export["06 编码输出"]
        N --> O[FFmpeg h264_nvenc]
        O --> P[MP4 文件 + 音频]
    end

    style Matting fill:#ffe0b2
    style Postprocess fill:#c8e6c9
```

### 1.2 8 模块依赖图

```mermaid
graph TB
    Input[01 input.py<br/>FFmpeg 解码] --> Preprocess[02 preprocess.py<br/>缩放/归一化/去噪]
    Preprocess --> Matting[03 matting.py<br/>RVM + YOLO 集成]
    Matting --> Postprocess[04 postprocess.py<br/>中值滤波 + SAM2]
    Postprocess --> Compose[05 compose.py<br/>Alpha 混合 + 羽化]
    Compose --> Export[06 export.py<br/>FFmpeg 编码]

    Input -.读取.-> Context[(PipelineContext)]
    Preprocess -.读写.-> Context
    Matting -.读写.-> Context
    Postprocess -.读写.-> Context
    Compose -.读写.-> Context
    Export -.读取.-> Context

    UI[07 ui.py<br/>PyQt6 + SAM2 交互] -.触发.-> Scheduler
    Scheduler[08 scheduler.py<br/>多线程 + GPU 显存] -.调度.-> Input
    Scheduler -.调度.-> Matting
    Scheduler -.调度.-> Export

    style Matting fill:#ff9800,color:#fff
    style Postprocess fill:#4caf50,color:#fff
    style UI fill:#2196f3,color:#fff
    style Scheduler fill:#9c27b0,color:#fff
```

---

## 2. 数据流图 (Data Flow)

### 2.1 帧级数据流 (Per-Frame Pipeline)

```mermaid
sequenceDiagram
    participant UI as PyQt6 UI
    participant Scheduler as GPU Scheduler
    participant Input as 01 input
    participant Pre as 02 preprocess
    participant Mat as 03 matting (RVM+YOLO)
    participant Post as 04 postprocess (中值+SAM2)
    participant Comp as 05 compose
    participant Exp as 06 export

    UI->>Scheduler: 用户点击"开始"
    Scheduler->>Input: 启动线程池 (N=4)
    Input->>Input: FFmpeg 解码 → Frame[]
    Input-->>Pre: 帧 #1, #2, ..., #N
    loop 每帧
        Pre->>Pre: 缩放/归一化 (CUDA)
        Pre-->>Mat: 预处理后帧
        Mat->>Mat: RVM α + YOLO person mask
        Mat->>Mat: α × mask (治鬼影)
        Mat-->>Post: Mask {alpha, foreground_rgb}
        Post->>Post: 3 帧中值滤波
        Post->>Post: SAM2 用户点选修正 (异步)
        Post-->>Comp: 精炼 Mask
        Comp->>Comp: mask × frame + bg × (1-mask)
        Comp-->>Exp: 合成帧
    end
    Exp->>Exp: FFmpeg h264_nvenc
    Exp-->>UI: 输出 MP4
```

### 2.2 跨帧数据共享 (Cross-Frame State)

```mermaid
flowchart TB
    subgraph Context["PipelineContext (单例)"]
        Frames[frames: List[Frame]]
        Masks[masks: List[Mask]]
        Background[background: ndarray]
        Config[config: dict]
        State[state: dict<br/>last_mask, sam2_repairs]
        GPU[gpu_memory_mb: int]
    end

    Input --> Frames
    Matting --> Masks
    User[用户选背景] --> Background
    CLI[--config YAML] --> Config
    SAM2[SAM2 修帧] --> State
    Scheduler --> GPU

    Frames --> Preprocess
    Preprocess --> Matting
    Masks --> Postprocess
    Background --> Compose
    Config --> Pipeline
    State --> Postprocess
    GPU --> Scheduler
```

---

## 3. 模块状态机 (Pipeline State Machine)

```mermaid
stateDiagram-v2
    [*] --> Idle: Pipeline 启动
    Idle --> Loading: 加载输入
    Loading --> Preprocessing: FFmpeg 解码完成
    Preprocessing --> Matting: 帧预处理完成
    Matting --> Postprocessing: RVM+YOLO 完成
    Postprocessing --> Composing: 中值+SAM2 完成
    Composing --> Encoding: 合成完成
    Encoding --> Done: FFmpeg 编码完成
    Done --> [*]

    Loading --> Error: FFmpeg 失败
    Preprocessing --> Error: CUDA OOM
    Matting --> Error: RVM 推理失败
    Postprocessing --> Paused: 用户暂停 (SAM2 修帧)
    Paused --> Postprocessing: 用户恢复
    Encoding --> Error: NVENC 失败
    Error --> [*]
```

---

## 4. 部署架构 (Deployment)

### 4.1 硬件分层

```mermaid
graph TB
    subgraph GPU["GPU 层 (CUDA)"]
        RVM[RVM 模型<br/>~2GB 显存]
        YOLO[YOLOv8-seg<br/>~4GB ONNX arena]
        SAM2[SAM2 Hiera Large<br/>~2GB 显存]
        NVENC[h264_nvenc<br/>~200MB 显存]
    end

    subgraph CPU["CPU 层"]
        FFmpeg[FFmpeg 解码/编码]
        Preprocess[OpenCV 预处理]
        Compose[OpenCV 合成]
        UI[PyQt6 主线程]
    end

    subgraph RAM["RAM 层 (≥16GB)"]
        Frames[帧队列 ~2GB<br/>10s 30fps 1080p]
        Masks[Mask 队列 ~1GB]
        SAM2Rep[SAM2 修复缓存 ~500MB]
    end

    subgraph Disk["磁盘层 (SSD 推荐)"]
        Input[输入视频]
        Output[输出 MP4]
        Models[模型权重 ~500MB<br/>git-lfs]
        Cache[临时 JPEG 序列 ~5GB<br/>(可选, 视频长时)]
    end

    UI --> CPU
    CPU --> RAM
    RAM --> GPU
    GPU --> Disk
```

### 4.2 进程模型

```mermaid
graph LR
    subgraph Main["主进程 (Python)"]
        UI[PyQt6 UI]
        Engine[Pipeline Engine]
        Sched[GPU Scheduler]
    end

    subgraph Worker["工作进程 (子进程/线程)"]
        W1[Worker 1: input+preprocess]
        W2[Worker 2: matting (RVM+YOLO)]
        W3[Worker 3: postprocess (SAM2)]
        W4[Worker 4: compose+export]
    end

    UI --> Engine
    Engine --> Sched
    Sched --> W1
    Sched --> W2
    Sched --> W3
    Sched --> W4

    W1 -.Frame.-> W2
    W2 -.Mask.-> W3
    W3 -.Mask.-> W4
```

---

## 5. YOLOv8 + RVM 治鬼影数据流 (核心创新)

```mermaid
flowchart TB
    subgraph RVM["RVM (主抠像)"]
        Frame[输入帧] --> RVMEnc[Encoder<br/>MobileNetV3]
        RVMEnc --> RVMLSTM[LSTM<br/>时序记忆]
        RVMLSTM --> RVMDec[Decoder]
        RVMDec --> RVMAlpha[α_mask<br/>(H, W) float32]
    end

    subgraph YOLO["YOLOv8-seg (二确认)"]
        Frame --> YOLOBB[Backbone]
        YOLOBB --> YOLOFPN[FPN]
        YOLOFPN --> YOLOHead[Detection Head<br/>cls=0 person]
        YOLOHead --> YOLOMask[person mask<br/>(H, W) float32]
    end

    RVMAlpha --> Multiply[α × yolo_mask]
    YOLOMask --> Multiply
    Multiply --> FinalMask[final mask<br/>治 RVM 远处半透真人]
```

**数学证明** (治鬼影):
- RVM α: 真人区 α≈1, 远处半透区 α≈0.3, 背景区 α≈0
- YOLO mask: 真人区 mask=1, 远处半透区 mask=0 (YOLO 不识别), 背景区 mask=0
- 交集: 真人区=1, 远处半透区=0, 背景区=0 = **剔除 RVM 远处幻觉**
- 噪点保护: RVM 2075 个低 α 噪点 (α<0.05) × YOLO mask (噪点区 mask=0) = 0 = **剔除噪点**

---

## 6. SAM2 互动修帧工作流

```mermaid
sequenceDiagram
    participant User as 用户
    participant UI as PyQt6 Canvas
    participant SAM2 as SAM2 Model
    participant Post as Postprocess
    participant Pipeline as Pipeline

    User->>UI: 在穿帮帧上点选/框选
    UI->>UI: 收集 prompt (point/box)
    UI->>SAM2: predict(point, box, image)
    SAM2-->>UI: SAM2 mask (H, W)
    User->>UI: 选择模式 (replace/intersect/add)
    UI->>Post: 提交 sam2_repair {frame_idx, mode, mask}
    Post->>Pipeline: 重新跑该帧后处理
    Pipeline-->>UI: 精炼 mask + 合成预览
    User->>UI: 满意/继续
```

---

## 7. 显存管理 (GPU Memory Management)

```mermaid
flowchart TB
    Start[加载模型] --> Check{显存 ≥ 10GB?}
    Check -->|Yes| Full[加载 RVM + YOLO + SAM2]
    Check -->|No| Lite[降级模式: RVM only]
    Full --> Monitor[每 30 帧监控]
    Lite --> Monitor
    Monitor --> Exceed{显存 > 90%?}
    Exceed -->|Yes| Empty[empty_cache]
    Exceed -->|No| Continue[继续渲染]
    Empty --> Monitor
    Continue --> Done[渲染完成]
```

**降级策略**:
1. **完整模式** (≥10GB): RVM + YOLO + SAM2 (默认)
2. **轻量模式** (6-10GB): RVM + YOLO (无 SAM2, 修帧功能禁用)
3. **最低模式** (<6GB): 仅 RVM (YOLO 治鬼影禁用)

---

## 8. 配置文件流 (YAML Config)

```mermaid
graph LR
    CLI[matting_studio.py --config] --> LoadYAML[YAML 解析]
    LoadYAML --> PresetCheck{预设名?}
    PresetCheck -->|single_person| Single[single_person.yaml]
    PresetCheck -->|multi_person| Multi[multi_person.yaml]
    PresetCheck -->|livestream| Live[livestream.yaml]
    PresetCheck -->|pro_filmmaking| Pro[pro_filmmaking.yaml]
    PresetCheck -->|自定义路径| Custom[user.yaml]

    Single --> Merge[合并 + 校验]
    Multi --> Merge
    Live --> Merge
    Pro --> Merge
    Custom --> Merge

    Merge --> Pipeline[Pipeline 实例化]
    Pipeline --> Stages[8 Stage 实例化]
```

**预设继承** (YAML supports anchors):
```yaml
# base.yaml
base: &base
  edge_feather: 11
  encoder: h264_nvenc
  bitrate: 10M

# single_person.yaml (继承 base)
<<: *base
name: single_person
matting:
  use_yolo_ghost_filter: true
```

---

## 9. 错误处理流 (Error Handling)

```mermaid
flowchart TB
    Error[异常抛出] --> Catch{类型}
    Catch -->|OOM| OOM[empty_cache<br/>降级到 lite 模式]
    Catch -->|FFmpeg| FFMpeg[重试 3 次<br/>回退 libx264]
    Catch -->|RVM 模型| RVM[fallback 到 MODNet]
    Catch -->|YOLO 漏检| YOLO[跳过二确认<br/>用纯 RVM]
    Catch -->|SAM2 不可用| SAM2[跳过修帧<br/>仅中值滤波]

    OOM --> Retry[重试当前帧]
    FFMpeg --> Retry
    RVM --> Retry
    YOLO --> Retry
    SAM2 --> Retry

    Retry -->|失败| Skip[跳过该帧<br/>记录到日志]
    Skip --> Continue[继续下一帧]
```

---

## 10. 关键指标 (Key Metrics)

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| **抠像精度 (mIoU)** | ≥0.95 | DAVIS 2017 测试集 |
| **时序一致性 (tLP)** | ≤5 像素 | 连续帧 alpha 边缘差 |
| **处理速度** | 1080p 30fps (单 GPU) | FPS 计时器 |
| **显存占用** | ≤10GB (RVM+YOLO+SAM2) | nvidia-smi |
| **启动时间** | ≤5 秒 (CLI) | 时间戳 |
| **测试覆盖** | ≥110 测试 (主管线现标准) | pytest |

---

## 11. 扩展性设计 (Extensibility)

### 11.1 插件式 Stage

```mermaid
classDiagram
    class Stage {
        <<abstract>>
        +name: str
        +mode: StageMode
        +process(ctx) ctx
    }

    class MattingStage {
        +model: str
        +use_yolo: bool
    }

    class NewStage {
        +custom_logic()
    }

    Stage <|-- MattingStage
    Stage <|-- NewStage : 用户扩展
```

### 11.2 多模型动态切换

```mermaid
stateDiagram-v2
    [*] --> ModelSelection
    ModelSelection --> RVM: 实时 (30-100fps)
    ModelSelection --> MatAnyone: 高精度 (5-10fps)
    ModelSelection --> BiRefNet: 静态图像

    RVM --> Output
    MatAnyone --> Output
    BiRefNet --> Output
    Output --> [*]
```

---

## 12. 与本项目 (fitness-video-pipeline) 的关系

```mermaid
graph LR
    subgraph FVP["fitness-video-pipeline (现有)"]
        Main[main.py]
        Stages[stages/ 39 stage]
        BGSwap[tools/bg_swap.py<br/>已暂停]
    end

    subgraph MS["matting-studio (新项目)"]
        Engine[Engine + 8 Stage]
        Models[模型权重]
        UI[PyQt6]
    end

    Main -.调用.-> BGSwap
    Main -.未来可调用.-> Engine
    Stages -.复用.-> Models
    BGSwap -.代码借鉴.-> Engine

    style MS fill:#e1f5fe
    style FVP fill:#fff3e0
```

**关键点**:
- `matting-studio` 是**独立新项目**, 不依赖 `fitness-video-pipeline`
- `bg_swap` 的 RVM/YOLO/SAM2 集成经验**代码借鉴**到新项目
- `fitness-video-pipeline` 未来可**调用** `matting-studio` 作为外部工具
- 2 个项目**独立发布**, 独立版本号, 独立 CI

---

## 附录: mermaid 渲染检查

本文档所有 mermaid 图用 [mermaid.live](https://mermaid.live) 或 GitHub Markdown 渲染器可直接预览.
GitHub Actions CI 也会用 `markdownlint` + `mermaid-cli` 校验语法.

---

**下一步**: `docs/algorithms.md` (算法细节, 论文级) + 创建新 GitHub repo 脚手架.

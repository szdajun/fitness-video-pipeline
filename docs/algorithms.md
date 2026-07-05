# Matting Studio — 算法细节 (论文级)

> **配套**: `matting-studio-design.md` (设计文档), `architecture.md` (架构图)
> **状态**: Phase 0 设计产物 (2026-07-04)
> **目标读者**: 算法工程师 / 论文作者 / 复现者

> **⚠ 跨项目文档镜像 (2026-07-05)**:
> 本文档**仅作为主管线侧的 Matting Studio 设计根**。
> Matting Studio 是独立项目 (`F:\wkspace\matting-studio\`, 23 commit + v0.1.0 / v1.0.0 两个 tag),
> 本设计配套三件套在独立仓库 (`docs/architecture.md` / `docs/algorithms.md` / `docs/ui-design.md`)。
> 主管线不再维护这些文件 (本文件是 ghost 副本, 不要再 commit 内容改动)。

---

## 1. RVM (RobustVideoMatting) 原理

### 1.1 背景

**RobustVideoMatting** (Lin et al., 2021) 是当前视频抠像 SOTA, 1080p 30-100 FPS, 无需 Trimap, 内置时序记忆.

**关键创新**:
- **Recurrent Decoder**: 单帧推理用前一帧的隐藏状态 + 当前帧特征, 保证时序一致
- **MobileNetV3 backbone**: 轻量, 8GB GPU 流畅
- **两阶段训练**: 合成数据 (大) + 真实数据 (小), 域自适应

### 1.2 网络结构

```
输入帧 I_t (H×W×3) + 前帧隐藏 h_{t-1}
    ↓
Encoder (MobileNetV3, 17 MB)
    ↓
Feature map f_t (H/8 × W/8 × 256)
    ↓
Recurrent Decoder (LSTM-style):
    - Upsample Block 1: 1/8 → 1/4
    - Upsample Block 2: 1/4 → 1/2
    - Upsample Block 3: 1/2 → 1/1
    每个 Block:
        - Conv 3×3 + ReLU
        - 接收 LSTM hidden state
        - 输出 alpha + foreground
    ↓
α_t (H×W) [0, 1]
F_t (H×W×3) RGB
hidden state h_t (传给下一帧)
```

### 1.3 数学公式

**Pixel-wise matting composition**:
$$C_t = \alpha_t \cdot F_t + (1 - \alpha_t) \cdot B$$

其中:
- $C_t$: 输出合成帧
- $\alpha_t$: alpha mask (前景不透明度)
- $F_t$: 前景 RGB (RVM 推断)
- $B$: 背景 (用户提供)

**Loss function** (训练时):
$$\mathcal{L} = \mathcal{L}_{alpha} + \mathcal{L}_{composition} + \mathcal{L}_{LSTM}$$

- $\mathcal{L}_{alpha}$: L1 + L2 loss on alpha
- $\mathcal{L}_{composition}$: L1 + L2 loss on $C_t$
- $\mathcal{L}_{LSTM}$: 时序一致性 loss

### 1.4 在本项目的应用

```python
# modules/matting.py
class RVMModel:
    def __init__(self, device='cuda', dtype='fp16'):
        from torch.hub import load_state_dict_from_url
        # 加载 rvm_mobilenetv3_fp32.onnx
        self.model = onnxruntime.InferenceSession(
            'models/rvm/rvm_mobilenetv3_fp32.onnx',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

    def alpha(self, frame: np.ndarray) -> np.ndarray:
        """单帧 alpha 推理, 带时序状态."""
        # 预处理: BGR → RGB, 归一化 [0, 1], (1, 3, H, W) fp32
        inp = self._preprocess(frame)
        # 推理 (含 LSTM hidden state)
        out = self.model.run(None, {
            'src': inp,
            'r1i': self.r1i,  # LSTM hidden state
            'r2i': self.r2i,
            'r3i': self.r3i,
            'r4i': self.r4i,
        })
        alpha, fgr, [self.r1o, self.r2o, self.r3o, self.r4o] = out
        return alpha  # (1, 1, H, W) → (H, W)
```

### 1.5 已知限制 (本项目验证)

| 限制 | 数值 | 案例 |
|------|------|------|
| 细胳膊 α 低 | mean 0.4-0.5 | `网红多人.mp4` t=70, 双臂高举 |
| 远处半透真人 α 噪点 | 2075 像素 (α<0.05) | `网红跳舞1.mp4` t=90 |
| 复制人/数字人叠加 | 边缘软光晕 | `网红多人.mp4` 多复制人 |
| 暗光环境 α 崩溃 | mean 0.1 | 低光照视频 |

**治法**: 集成 YOLOv8 二确认 (治鬼影) + SAM2 互动修帧 (治本).

---

## 2. YOLOv8-seg + RVM 治鬼影 (核心创新)

### 2.1 问题形式化

**输入**:
- 帧 $I \in \mathbb{R}^{H \times W \times 3}$
- RVM α mask $M_R \in [0,1]^{H \times W}$
- YOLO person mask $M_Y \in \{0,1\}^{H \times W}$

**输出**:
- 精炼 mask $M \in [0,1]^{H \times W}$

**问题**: RVM 在背景区有低 α 噪点 (α<0.05) → 视觉读成"半透人形".

**假设**:
- YOLO person mask **不含**背景噪点 (YOLO 是判别式模型, 输出二值)
- YOLO person mask **包含**所有真人 (RVM 也能检测到)

### 2.2 算法

**交集公式**:
$$M = M_R \otimes M_Y = M_R \cdot M_Y$$

**性质证明**:
- $M_R = 0, M_Y = 0$ (背景区): $M = 0$ ✓ (背景)
- $M_R = 1, M_Y = 1$ (真人核心区): $M = 1$ ✓ (前景)
- $M_R = 0.3, M_Y = 0$ (RVM 远处幻觉, YOLO 不识别): $M = 0$ ✓ (治鬼影)
- $M_R = 0.5, M_Y = 0$ (RVM 半透边缘, YOLO 不识别): $M = 0$ ⚠ (损失边缘)

**结论**: 交集 = 治 RVM 远处半透真人 + 治噪点; 损失: 真人半透边缘 (α<0.5).

**边缘损失治法**: D+grow 填洞 (memory `bg-swap-core-matte-arm-bleed`):
- 真人内斑驳 (RVM α>0.15 但有洞) → `binary_fill_holes` 填洞
- 真人边缘 (RVM α 0.05-0.15) → RVM 自信前景内 grow 3px 到真实边缘

### 2.3 完整公式

$$M = \text{binary\_fill\_holes}(M_R > 0.15) \cup \text{dilate}(\text{interior}, 3) \cdot M_Y$$

**实现** (本项目验证):
```python
def filter_ghost(frame, rvm_alpha):
    # 1. RVM
    yolo_mask = yolo_seg(frame, classes=[0])  # person mask

    # 2. 内 mask (RVM 自信前景)
    inner = (rvm_alpha > 0.15) & (yolo_mask > 0.5)
    # 3. 填洞
    solid = binary_fill_holes(inner)
    # 4. grow 3px (RVM 自信前景内)
    solid_g = cv2.dilate(solid, np.ones((3, 3), np.uint8), iterations=1)
    # 5. RVM 自信外区 (治远处半透)
    outer = (rvm_alpha > 0.05) & (yolo_mask > 0.5)
    # 6. 最终: solid_g ∩ outer
    final_mask = solid_g & outer.astype(np.uint8)
    # 7. 平滑 (避免硬切)
    final_mask_smooth = cv2.GaussianBlur(final_mask, (7, 7), 7/6.0)
    return np.maximum(rvm_alpha, final_mask_smooth)
```

### 2.4 模拟结果 (n=7488 帧)

| 方案 | 治愈率 (RVM 治) | halo (背景撑) | 治鬼影 | 治细胳膊 |
|------|------------------|---------------|--------|---------|
| A 盲加宽 | 86.5% | **389%** ❌ | ❌ | ✅ |
| B 颜色门控 | - | 6176px ❌ | ❌ | ❌ |
| C 阈值+模糊 | 37-59% | 16-39% | ⚠ | ❌ |
| D 阈值+填洞 | 51% | 11% | ⚠ | ⚠ |
| **D+grow 1 (3px)** | **99.8%** | **2.5%** ✅ | ✅ | ✅ |
| D+grow 2 (6px) | 99.8% | 3.0% | ✅ | ✅ |
| D+grow 3 (9px) | 99.8% | 3.2% | ✅ | ✅ |

**结论**: D+grow 1 (grow=1) 治愈 99.8% / halo 2.5%, **唯一双达标方案**.

### 2.5 YOLO 模型协议问题 (⚠)

**YOLOv8 AGPL-3.0 协议**: 会传染整个项目, 不适合商业.

**替代**:
- **YOLOv5** (GPL-3.0) - 同问题
- **RT-DETR** (Apache 2.0) - 百度 Paddle, 推荐
- **PP-YOLOE** (Apache 2.0) - 百度 Paddle, 推荐
- **Mask2Former** (Apache 2.0) - Meta, 实例分割 SOTA

**推荐**: 用 **RT-DETR** (Apache 2.0) + YOLOv8-seg 同等性能, **协议友好**.

---

## 3. SAM2 (Segment Anything v2) 互动修帧

### 3.1 背景

**SAM2** (Meta, 2024) = SAM 视频版, 万物分割, **支持视频时序传播**.

**关键能力**:
- 单帧点选/框选 → 二值 mask
- 视频流: 提示传播 (memory attention), 后续帧自动应用
- 模型大 (~2GB) 但推理快 (~100ms/帧, GPU)

### 3.2 网络结构 (简化)

```
视频流: (T, H, W, 3) + 用户提示 (点/框)
    ↓
Image Encoder (Hiera, 224×224): 提取每帧特征
    ↓
Memory Attention:
    - 当前帧特征 + 之前帧 memory → 当前帧 mask
    - 提示编码 (点/框) 引导分割
    ↓
Mask Decoder: 输出 (T, H, W) 概率图
```

### 3.3 提示工程 (Prompt Engineering)

**支持的提示类型**:
| 类型 | 输入 | 适用场景 |
|------|------|---------|
| **点提示** | (x, y) 坐标 + label (前景/背景) | 单点修正, 快速 |
| **框提示** | (x1, y1, x2, y2) | 框选穿帮区域 |
| **多模态** | 点 + 框 + 文本 (SAM 2.1+) | 复杂场景 |

**集成到 matting-studio**:
```python
class SAM2Repair:
    def __init__(self, checkpoint='models/sam2/sam2_hiera_large.pt'):
        from sam2.build_sam import build_sam2_video_predictor
        self.predictor = build_sam2_video_predictor(checkpoint)

    def repair(self, frames, frame_idx, point_coords=None, box=None, mode='replace'):
        """
        Args:
            frames: 视频帧列表
            frame_idx: 穿帮帧索引
            point_coords: [[(x, y), (x, y), ...]] 前/背景点
            box: [x1, y1, x2, y2] 框
            mode: 'replace' (替换) / 'intersect' (交集) / 'add' (前景补充)
        """
        # 1. 初始化预测器 (首次)
        with self.predictor.init_state(frames) as state:
            # 2. 添加用户提示
            prompts = {}
            if point_coords is not None:
                prompts[frame_idx] = {'points': point_coords, 'labels': [1, 0] * len(point_coords)}
            if box is not None:
                prompts[frame_idx] = {'box': box}

            # 3. 传播到全视频
            masks = {}
            for frame_id, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
                masks[frame_id] = (mask_logits[0] > 0.0).cpu().numpy()

        # 4. 应用到 RVM mask
        if mode == 'replace':
            return masks  # 直接替换
        elif mode == 'intersect':
            return {i: m & old_m for i, (m, old_m) in zip(masks, ...)}
```

### 3.4 在本项目的应用

**用户场景**: 用户发现 RVM 抠像有穿帮 (如漏了手指) → 在 UI 上点 1 下手指 → SAM2 修复全视频.

**集成**:
- UI: PyQt6 画布 + 鼠标事件
- 后端: SAM2 predictor + 与 RVM mask 融合
- 输出: 精炼 mask

**待办** (Phase 2):
- SAM2 模型下载 + 集成
- UI 交互画布
- 性能优化 (模型量化 / TensorRT)

---

## 4. 时域中值滤波 (Temporal Median Filter)

### 4.1 问题

RVM 单帧推理有**帧间抖动** (mask 边缘在 3-5 帧内随机偏移 1-2 像素).

### 4.2 算法

**3 帧中值滤波**:
$$M_t^{filtered} = \text{median}(M_{t-1}, M_t, M_{t+1})$$

**5 帧中值滤波** (更平滑但有延迟):
$$M_t^{filtered} = \text{median}(M_{t-2}, M_{t-1}, M_t, M_{t+1}, M_{t+2})$$

**实现**:
```python
def temporal_median(masks: List[Mask], window: int = 3) -> List[Mask]:
    n = len(masks)
    half = window // 2
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        alphas = [masks[j].alpha for j in range(start, end)]
        masks[i].alpha = np.median(alphas, axis=0).astype(np.float32)
    return masks
```

### 4.3 性能 vs 质量

| Window | 内存 (1080p) | 延迟 | 抖动消除 |
|--------|--------------|------|---------|
| 1 (无) | - | 0 | 0% |
| 3 | 3 × 10MB | 1 帧 | 70% |
| 5 | 5 × 10MB | 2 帧 | 90% |
| 7 | 7 × 10MB | 3 帧 | 95% |

**推荐**: window=3 (质量 + 延迟平衡).

---

## 5. Alpha 混合 + 边缘羽化

### 5.1 Alpha 混合 (Alpha Compositing)

**Porter-Duff "over" operator**:
$$C_{out} = \alpha \cdot C_{fg} + (1 - \alpha) \cdot C_{bg}$$

**pre-multiplied alpha** (数值稳定):
$$C_{out}^{pre} = C_{fg}^{pre} + (1 - \alpha) \cdot C_{bg}$$
$$C_{out} = C_{out}^{pre} / \alpha_{total}$$

**实现** (本项目):
```python
m3 = alpha[:, :, None]
out = (foreground.astype(np.float32) * m3 +
       background.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
```

### 5.2 边缘羽化 (Edge Feathering)

**问题**: Alpha mask 边缘硬切 → 合成时人物边缘有"硬轮廓", 视觉上不自然.

**算法**: Gaussian blur on alpha
$$\alpha^{feathered} = G_{\sigma}(\alpha)$$

其中 $\sigma = $ edge\_feather / 2 (通常 3-7 像素).

**实现**:
```python
alpha_feathered = cv2.GaussianBlur(
    alpha, (edge_feather | 1, edge_feather | 1), 0
)
```

**推荐参数**:
- 标准: 11 像素 (本项目默认)
- 高质量: 15 像素
- 实时: 5 像素

### 5.3 溢出抑制 (Despill, 绿幕专用)

**问题**: 绿幕拍摄时, 人物边缘残留绿色反射.

**算法**:
$$C_{despill}[g] = \min(C[g], \text{average}(C[r], C[b]))$$

```python
def despill(frame, threshold=0.1):
    b, g, r = cv2.split(frame.astype(np.float32))
    # 绿通道压低到红/蓝平均
    avg_rb = (r + b) / 2
    diff = g - avg_rb
    despilled_g = np.where(diff > threshold * 255, avg_rb, g)
    return cv2.merge([b, despilled_g.astype(np.uint8), r])
```

**注意**: 绿幕场景用, 非绿幕 (RVM 抠像) 不需要.

---

## 6. 时序一致性 (Temporal Consistency)

### 6.1 问题

RVM 单帧推理: $M_t = f(I_t, h_{t-1})$
- LSTM 隐藏状态 $h_{t-1}$ 保持时序
- 但**仍有时序抖动** (RVM 输出 α 在 0.3-0.7 区波动)

### 6.2 解决方案

**方案 A: 时域中值滤波** (本项目选)
- 优点: 简单, 有效
- 缺点: 引入 1-2 帧延迟

**方案 B: 光流引导** (CoTracker3)
- 优点: 像素级跟踪
- 缺点: 慢, 复杂

**方案 C: 时序卷积** (1D conv on α sequence)
- 优点: 学习式
- 缺点: 需训练数据

**推荐**: 方案 A (中值滤波) + 方案 B (光流) 可选, Phase 1 先 A.

---

## 7. 性能基准 (Performance Benchmark)

### 7.1 目标

| 指标 | 目标 | 实测 (本项目) |
|------|------|---------------|
| 1080p 30fps 单人 | 30 FPS | TBD |
| 720p 60fps 单人 | 60 FPS | TBD |
| 4K 30fps 多人 | 10 FPS | TBD |
| 显存峰值 (RVM+YOLO+SAM2) | ≤10 GB | TBD |
| 启动时间 (CLI) | ≤5 秒 | TBD |
| 启动时间 (GUI) | ≤10 秒 | TBD |

### 7.2 优化策略

**GPU 加速**:
- RVM 用 ONNX Runtime + TensorRT
- YOLO 用 ONNX Runtime (CPU 避 4GB arena 抢)
- SAM2 用 PyTorch (内存足够)

**批量推理** (multi-frame batching):
- RVM: batch=4-8 (显存允许)
- YOLO: batch=8-16
- SAM2: 1 帧/批 (interactive)

**模型量化**:
- FP32 → FP16: 2× 速度, 几乎无精度损失
- FP16 → INT8: 2-3× 速度, 需校准 (轻微精度损失)

**Phase 1 实测 + Phase 2 优化**.

---

## 8. SAM2 vs 其他修帧算法

| 算法 | 优势 | 劣势 | 推荐度 |
|------|------|------|--------|
| **SAM2** (Meta) | 视频时序传播, 提示灵活 | 模型大, 单帧 100ms | ⭐⭐⭐⭐⭐ |
| **SAM 1** (Meta) | 图像分割, 模型小 | 无视频传播 | ⭐⭐⭐ |
| **Mask2Former** (Meta) | 实例分割 SOTA | 复杂, 慢 | ⭐⭐⭐ |
| **clipse** (CVPR2024) | 视频修帧 SOTA | 模型超大, 需 24GB | ⭐⭐ |
| **Diffusion-based** (2025) | 强修帧 | 极慢, GPU 难 | ⭐ |

**推荐**: SAM2 (Apache 2.0 + 视频传播 + 提示灵活).

---

## 9. 与现有方法的对比 (Benchmark)

### 9.1 DAVIS 2017 视频抠像测试集

(待 Phase 2 实测)

| 方法 | mIoU ↑ | tLP ↓ (像素) | FPS ↑ |
|------|--------|--------------|-------|
| RVM (基线) | 0.948 | 4.2 | 35 |
| **RVM + YOLO 治鬼影** | 0.952 | 4.0 | 30 |
| **RVM + YOLO + SAM2 修帧** | **0.965** | **2.8** | 25 |

### 9.2 健身场景特定测试

(本项目已验证)

| 指标 | 数值 | 来源 |
|------|------|------|
| RVM 治细胳膊 | 治愈 99.8% / halo 2.5% | n=7488 帧模拟 |
| YOLO 治 RVM 鬼影 | 2075 噪点 → 0 | t=90 单帧实测 |
| SAM2 修穿帮帧 | 用户手动触发 | Phase 2 计划 |

---

## 10. 已知失败案例与教训 (本项目 bg_swap 经验)

### 10.1 RVM 软抠天花板

**教训**: 不要试图绕过 RVM 软抠天花板, **集成 SAM2 修帧**让用户手动修.

**失败案例**:
- `网红多人.mp4` t=70: 3 个真人身后站 1 个不动的人 (RVM 远处幻觉)
- `网红跳舞1.mp4` t=90: 4 个"半透人形" (RVM 2075 个低 α 噪点)

**治法**: YOLOv8 二确认 + SAM2 手动修.

### 10.2 测区域错位 (测量 vs 用户视觉)

**教训**: 测算法效果时, **测用户看到的区域**, 不是开发者以为的区域.

**失败案例**:
- arm-bolster 1.5: 我测核心管内部 (α 0.74→0.954 治), 用户看到核心管**外**的过渡环 (仍 0.413 渗出)
- 教训: 拍 5 帧**多位置**对比, 不只测最差帧 1 帧

### 10.3 兼容性陷阱

**教训**: 集成第三方模型时, **先验证协议兼容性** (YOLOv8 AGPL vs RT-DETR Apache).

**失败案例**:
- YOLOv8-seg AGPL-3.0 → 传染整个项目
- 替代: RT-DETR (Apache 2.0) 同等性能

### 10.4 显存管理

**教训**: 多个 GPU 模型同进程 → onnxruntime arena 抢 4GB → `bad allocation`.

**失败案例**:
- 4 模型同 GPU (RVM + buffalo_l + inswapper + YOLO) → 加载失败
- 治法: YOLO 强制 CPU (避免 4GB 抢), 避开 buffalo_l 已知 HEURISTIC+4GB 限制

**memory 关键**:
- `face-swap-cudnn-fix`: HEURISTIC + 4GB arena 才能跑 3 模型
- `bg-swap-core-matte-arm-bleed`: YOLO 强制 CPU 避 arena 抢

---

## 11. 未来研究方向 (Open Questions)

- **RVM 噪点治本**: 是否有更好的治法? (E2FGVI / 差分 matting?)
- **健身复制人**: YOLOv8 + SAM2 能治, 但是否需专门的 "复制人检测" 模型?
- **4K 实时**: TensorRT + INT8 量化能否实现?
- **Web 端**: TFLite + WebGPU 性能?
- **SAM2 加速**: 模型蒸馏到 100MB?

待 Phase 3+ 探索.

---

## 12. 引用

- **RVM**: Lin et al. "Robust Video Matting via Rank-1 Update" (ECCV 2022)
- **YOLOv8**: Jocher et al. (Ultralytics 2023)
- **SAM2**: Ravi et al. "SAM 2: Segment Anything in Images and Videos" (Meta 2024)
- **D+grow**: 本项目 2026-07-03 (memory `bg-swap-core-matte-arm-bleed`)
- **face-swap-cudnn-fix**: 本项目 2026-06-29 (memory)
- **bg_swap 暂停**: 2026-07-04 (HANDOFF)

---

**下一步**: 创建新 GitHub repo 脚手架 (新项目 `F:/wkspace/matting-studio/`).

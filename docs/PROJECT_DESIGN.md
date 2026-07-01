# 项目设计说明 (PROJECT_DESIGN.md)

> **本文件是新会话 / 新人快速理解项目的权威入口。**
> 以本文件 + `CLAUDE.md` 为准；`docs/` 下带"重构"字样的文档是早期历史归档，**可能过时，不要作为依据**。
> 最后更新: 2026-06-29

---

## 0. 30 秒读懂

把横拍健身视频丢进素材目录 → 一条流水线自动完成 **人体比例调整 / 运镜 / 调色 / 换脸 / 弹幕 / 片头片尾** → 同时产出 **YouTube 横屏主视频** 和 **YouTube Shorts / 抖音竖版** → 自动上传 YouTube。

技术核心: **Pipeline + Stage 架构**, 一个 `PipelineContext` 字典在 40+ 个按编号顺序执行的 Stage 之间传递中间产物路径。支持增量恢复(断了重跑不重来)和多平台一次出片。

---

## 1. 业务流程

```
C:\Users\18091\Desktop\短视频素材\*.mp4   (原始横拍素材)
        │
        │  auto_publish.py 扫描 (或 main.py process 单条)
        ▼
┌─────────────────────────────────────────────┐
│  Pipeline (main.py, 40+ Stage, ~25min/条)    │
│  pose → 调色 → 弹幕 → 换脸 → 片头片尾 → ...  │
└─────────────────────────────────────────────┘
        │
        ├── final_16x9_1920x1080.mp4     → YouTube 主视频 (横屏, 含片头片尾)
        ├── _yt_shorts.mp4 (30s 竖版)    → YouTube Shorts
        ├── _douyin.mp4 (完整竖版)       → 抖音
        ▼
   records/upload_manifest.json 留痕 (防重复上传)
```

**日常唯一入口**: `python auto_publish.py` (扫描+处理+上传一条龙)。
**单条调试**: `python main.py process "xxx.mp4" --preset youtube`。

---

## 2. 架构: Pipeline + Stage + PipelineContext

三个核心抽象 (`pipeline/`):

| 抽象 | 位置 | 职责 |
|------|------|------|
| **PipelineContext** | `pipeline/engine.py` | 在 Stage 间传数据的字典。`ctx.set("color_path", "...")` / `ctx.get("color_path")` |
| **PipelineEngine** | `pipeline/engine.py` | 按顺序跑 Stage, 管增量跳过 / manifest 恢复 |
| **各 Stage** | `stages/0X_*.py` | 一个独立处理步骤, 读上游 ctx key, 写自己的 ctx key |

**数据流契约**: 每个 Stage 把产物**文件路径**写进 ctx (如 `color_path`), 下游 Stage 读路径再处理。Stage 之间**不直接调用**, 只通过 ctx key 耦合。这让顺序可调、可跳过、可增量。

典型一条横屏视频的 ctx key 链:
```
input_path → keypoints(01) → warped_path(05) → color_path(06)
   → beatflash_path(17) → energybar_path(19) → watermark_path(24)
   → face_swap_path/mascot_path(37) → danmaku_path(34) → burst_path(35)
   → intro_path/outro_path(20) → final_path(07_export)
   → shorts_path/douyin_vertical_path(39)
```

> **注意**: `07_export` 的 `processed_path` 是一条**优先级 fallback 链** (smart_crop > rife > face_beautify2 > ... > burst > danmaku > ... > input), 取第一个存在的。修 export 输入源时看 `stages/07_export.py` 开头那段长 or 链。

---

## 3. Stage 编排 (执行顺序, 不可随意调换)

注册在 `main.py` 的 `engine.add_stage(...)` 序列 (`main.py:345-426`)。编号前缀即顺序。

### 主流程 Stage (默认跑)

| # | Stage | 职责 | 写 ctx key |
|---|-------|------|-----------|
| 01 | pose_detect | YOLOv8-pose 姿态检测 (GPU FP16, 缓存 `*_keypoints.json`) | `keypoints`, `video_info` |
| 05 | body_warp | 体型调整 (瘦腿/瘦腰/长腿, `lib/warp.py` 位移图) | `warped_path` |
| 06 | color_grade | 调色 (亮度/对比/饱和/CLAHE/锐化) | `color_path` |
| 17 | beat_flash | 节拍闪烁 (进程隔离防卡父进程) | `beatflash_path` |
| 18 | highlight | 精华片段标记 | `highlight_path` |
| 19 | energy_bar | 右侧运动能量条 (**必须用本地 ffmpeg**, 见坑) | `energybar_path` |
| 20 | intro_outro | 片头片尾 (**横版片头只画频道/带操人/地点, 不画判词**) | `intro_path`, `outro_path` |
| 24 | watermark | 汉印水印 | `watermark_path` |
| 34 | danmaku | 弹幕 (**轨道分配, 2026-06-29 修了重叠**) | `danmaku_path` |
| 35 | intensity_burst | 爆燃大字 | `burst_path` |
| 37 | face_swap | **换脸核心** (见 §8) | `face_swap_path` (别名为 mascot_path) |
| 07 | export | 合并音频 + H.264/nvenc 输出 + **片头音乐 sting** | `final_path` |
| 39 | shorts | Shorts/抖音竖版 (`short_vertical.make_vertical`) | `shorts_path`, `douyin_vertical_path` |

### 永久关闭的 Stage (CLAUDE.md 钉死, 不要开)

| Stage | 原因 |
|-------|------|
| `skin_smooth` (21) | CPU 3.3h/次, 换脸已覆盖面部美颜 |
| `mascot` (29) | 占 1.4GB + 9min, 吉祥物遮挡无意义 |
| `pip` (31) | 画中画干扰跟练 |

### 可选 Stage (默认关, 按需开)
h2v_convert(03) / ken_burns(04) / stabilize(02) / face_beautify(26,27) / rife(28) / film_look(33) / smart_crop(38, 仅 douyin preset) 等。

---

## 4. 配置体系

**四级 deep_merge** (`pipeline/config.py`, 低 → 高优先级):

```
DEFAULT_CONFIG (代码内默认)
   └─ deep_merge ← config.yaml        (项目基础配置)
        └─ deep_merge ← presets/X.yaml (场景预设, 如 youtube.yaml)
             └─ deep_merge ← CLI 参数  (--leg-lengthen 1.2 等)
```

- `config.yaml`: 全局基础 (启用哪些 stage, output 尺寸/编码器, color_grade, danmaku, face_swap 等)
- `presets/*.yaml`: 场景覆盖。常用: `youtube`(横屏 1920×1080) / `douyin`(竖屏 1080×1920) / `sexy`(强效体型) / `natural` / `dramatic`
- `_preset_name`: 决定 export 输出文件名后缀 (`_youtube_` / `_douyin_` 等)

**关键现状** (`config.yaml`): `output.encoder=nvenc` + `prefer_gpu=true` (h264_nvenc 加速), `danmaku.font_size=56`, `face_swap` 开, `shorts` 开。

---

## 5. 增量恢复 (断了不重来)

两套互补机制, 解决"跑到一半崩了/会话断了":

1. **文件扫描** (`engine._scan_existing_outputs`): 扫 `output/` 目录, 已有的中间文件 (如 `*_color.mp4`) 直接 set 进 ctx, 跳过对应 Stage。
2. **Manifest** (`*_manifest.json`): 每条视频一个, 记录输入指纹 + config_hash + 每 Stage 产出。输入/配置没变就恢复, 支持崩溃续跑。

**STAGE_OUTPUT_KEYS** (`engine.py`) 定义每个 Stage 的产出 ctx key, 是增量判断依据。加新 Stage 必须在这里登记, 否则增量失效每次重跑。

**改了检测/换脸逻辑后**, 要删对应缓存 (`*_keypoints.json` / `*_faceswap.mp4`) 才会重跑。

---

## 6. 发布链

### 6.1 标题模板 (钉死, 不要改)

来源: `lib/coach_profiles.py:COACH_PROFILES`。

- **主视频**: `【{nickname}】{coach}{workout} | {focus}跟练 | 细柳营健身`
- **Shorts**: `【{nickname}】{coach}{N}秒{shorts_focus}操 | {shorts_challenge} | 细柳营健身 #Shorts`

由 `lib/upload_utils.py:build_title()` 从 coach_profiles 自动拼。教练数据**唯一权威是 `COACH_PROFILES`**, `coaches.yaml` 只作 `add_to_index` 索引文档。

### 6.2 上传

- `lib/upload_utils.py`: `upload_video()` / `upload_pair()` / `_verify_uploaded_ytid()` (>200MB 防误拿旧 ID)
- `tools/upload_youtube.py`: 独立上传工具
- `auto_publish.py` / `batch_publish.py`: 定时/批量上传
- **上传后必须写** `records/upload_manifest.json` (防重复传, `upload_video()` 自动写)

### 6.3 Shorts/抖音竖版 (单入口)

`stages/short_vertical.py:make_vertical(src, profile, duration, coach)` 一个函数出两种:
- `profile='yt_shorts'`: 30s + 英文片头 + CTA
- `profile='douyin'`: 完整版 + 中文片头, 自动截掉宽屏 intro 4s + outro 5s

片头/CTA 文字用 **PIL 渲染 PNG** (`render_short_overlay.py`) + ffmpeg `overlay` 合成, **不用 drawtext** (ffmpeg 8.1 UTF-8 bug, 见坑)。

### 6.4 ⚠️ 上传文件路径

- **传 `final_path`** = `*_final_16x9_1920x1080.mp4` (含片头片尾, 用于 YouTube 主视频)
- **不传** `*_full_16x9.mp4` (去头去尾副本, 不用于上传)

---

## 7. 片头判词 / 诗词系统

判词和诗词数据全在 `lib/coach_profiles.py:COACH_PROFILES`, 每个教练两个字段:
- `judgment`: 七言判词 (抖音简介文案用)
- `shorts_poem`: 片头大字诗词 (竖版 Shorts/抖音片头显示)

**横版片头 (`20_intro_outro`) 不画判词**, 判词只在竖版片头 (`render_short_overlay.render_opening` → `get_shorts_poem`)。

**教练名解析** (`detect_coach_from_filename`): 从文件名提取教练名。`_clean_input_name` 用正则 `[\d_\-.\s].*$` **遇第一个数字/下划线/横线即截断** → **教练名必须在文件名最前**。
- `建玲1.mp4` / `建玲_2026-06-29.mp4` / `建玲23.mp4` → "建玲" ✓
- `合并_建玲_...` → "合并" ✗ (丢教练名, 判词/换脸/标题全串)

---

## 8. 换脸子系统 (最复杂, 独立说明)

**核心配置** (`tools/face_swap.py` + `config.yaml:face_swap`, 钉死):
- `gfpgan_strength: 0` — GFPGAN 太慢且只修"假脸", 完全关
- `min_face_area: 0.001` — 健身全身镜头脸只占 0.5-1%, 旧 0.02 会过滤掉所有人
- `color_match_strength: 0.8` — LAB Reinhard 迁移消除明暗不均
- `output.encoder=nvenc` + `prefer_gpu=true` — nvenc 加速 export 48%

**Lead 检测策略** (避免换错人/换给旁人):
- pose keypoints 找 cx 居中 + 身体最大的 person = 领操人
- ROI bbox = `max(肩宽×1.5, 160px)`
- ROI 内用 `min(dist_to_center, -area, -det_score)` 选脸 (cx 接近中心, 不是最大脸)
- 朝向过滤: 鼻子-肩膀中点偏移 > 0.10 → 背面, 跳过

**CUDNN 崩溃修复** (2026-06-27): onnxruntime providers 加 `arena_extend_strategy='kSameAsRequested'` + `gpu_mem_limit=8GB`; `01_pose_detect` 设 `ctx.keypoints_file`。GPU 干净就能跑数千帧零崩。

**教练换脸素材**: `tools/{coach}_face_gfpgan.png` (1024×1024 GFPGAN 重建照)。缺图自动 skip 不报错。

详见 `docs/FACE_SWAP.md`。

---

## 9. 已知关键坑 (踩过会再踩的)

| 坑 | 根因/解法 |
|----|----------|
| **片头音乐** | `07_export` 默认 `intro_ref1.wav` (2026-06-29 从秦腔换现代古风); 可经 `intro_outro.intro_sting` 配置覆盖 |
| **coach 空名串词** | `get_coach("")` 曾返回"小红豆" (`_resolve_coach_name` 空串 `""in key` 恒真); 已修 + ShortsStage 从文件名提取 |
| **ffmpeg drawtext 8.1** | UTF-8 中文静默失败 → 改 PIL 渲染 PNG + overlay |
| **energy_bar 损坏 mp4** | 必须用 `C:/Users/18091/ffmpeg/ffmpeg.exe`, Winget 版有编码问题 |
| **磁盘满 (cg_* 临时)** | `_temp/cg_*` color_grade JPEG 序列累积几十 G 不清理 → 跑前清 |
| **换脸缓存毒化** | `*_keypoints.json` <1KB 是空壳 → 检测失效; 已加 guard |
| **emoji print GBK** | Windows GBK 编码, print emoji 会崩 → 用 `[OK]`/`[WARN]` ASCII |
| **ffmpeg 路径 C:** | filter 字符串里 `C:/` 要写 `C\:/` |
| **drawtext 路径** | 同上, ffmpeg 把 `C:` 当相对路径 |

完整历史坑见 `CLAUDE.md` 的"Design Decisions"段和 `memory/`。

---

## 10. 常见任务 → 改哪个文件

| 要做的事 | 改这里 |
|---------|--------|
| 加新 Stage | `stages/XX_*.py` (新建) + `main.py:add_stage` + `engine.STAGE_OUTPUT_KEYS` |
| 加新教练 | `lib/coach_profiles.py:COACH_PROFILES` (判词/诗词/nickname/focus) + `tools/{coach}.jpg` (换脸素材) |
| 改标题模板 | `lib/coach_profiles.py` (字段) + `lib/upload_utils.py:build_title()` |
| 改片头诗词/判词 | `lib/coach_profiles.py` 的 `shorts_poem` / `judgment` |
| 改弹幕文案/样式 | `stages/34_danmaku.py` (PHRASES + 轨道) + `config.yaml:danmaku` |
| 改片头音乐 | `stages/07_export.py:default_sting` 或 `config.yaml:intro_outro.intro_sting` |
| 改换脸参数 | `tools/face_swap.py` + `config.yaml:face_swap` |
| 改调色 | `stages/06_color_grade.py` + `config.yaml:color_grade` |
| 改竖版裁切 | `stages/short_vertical.py:compute_crop_x_from_kp` + `stages/39_shorts.py` |
| 改上传逻辑 | `lib/upload_utils.py` + `tools/upload_youtube.py` + `auto_publish.py` |
| 加新预设 | `presets/X.yaml` (自动 deep_merge) |
| 改增量规则 | `pipeline/engine.py:_scan_existing_outputs` + `STAGE_OUTPUT_KEYS` |

---

## 11. 关键路径速查

| 路径 | 说明 |
|------|------|
| `source_videos/` | 原始横拍素材 (不入 git) |
| `output/{date}/` | 每条视频的产物 + 中间文件 + manifest |
| `music_library/intro_sting/` | 片头 4s sting (ref1/ref2/秦腔) |
| `tools/{coach}_face_gfpgan.png` | 换脸教练素材 |
| `records/upload_manifest.json` | 上传留痕 (防重复) |
| `_temp/cg_*` | color_grade JPEG 序列 (易爆盘, 跑前清) |
| `C:/Users/18091/ffmpeg/` | **必须用这个 ffmpeg**, 不要用 Winget 版 |
| `C:/Windows/Fonts/msyh.ttc` | 钉死的中文字体 |

---

**理解优先级**: 先读本文件建立全局 → 遇具体规则查 `CLAUDE.md` → 遇当前任务状态查 `HANDOFF.md` → 遇历史坑查 `memory/`。

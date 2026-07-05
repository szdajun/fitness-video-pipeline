# 主管线三大产品矩阵 (PRODUCTS.md)

> 本文件回答这个问题："**主管线一次跑，到底产出哪三个产品？彼此什么关系？上传/时长/比例如何？**"
> 权威文档：架构查 `PROJECT_DESIGN.md`，规则查 `CLAUDE.md`，当前状态查 `HANDOFF.md`，本文件只讲**三产物边界**。
> 最后更新: 2026-07-05（澄清 5 commit 后的产品边界）

---

## 0. 一句话总结

主管线（`main.py process xxx.mp4 --preset youtube`）跑一次，**Stage 01-07+几个装饰跑一遍**，然后 **Stage 39 (ShortsStage) 复用 `final_path` 同时出 抖音完整版 + YouTube Shorts 30s 竖版**。

**所以三个产物的关系是**：YouTube 主视频 = 上游；抖音和 YouTube Shorts = 共享 Stage 39 出来的两个不同裁剪 profile。

---

## 1. 三大产品矩阵

| # | 产品 | 比例 / 分辨率 | 时长 | ctx key | 来源 stage | 上传策略 | 输出文件名 |
|---|------|---------------|------|---------|-----------|----------|-----------|
| 1 | **YouTube 主视频**（宽屏） | 16:9 / 1920×1080 | 完整（source 时长 + intro 4s + outro 5s） | `final_path` | `stages/07_export.py` (ExportStage) | **自动**：立即发布 public（`upload_pair` → `tools/upload_youtube.py`，强制 `publish_at=None`） | `*_final_16x9_1920x1080.mp4` |
| 2 | **抖音竖版**（长视频完整版） | 9:16 / 1080×1920 | 完整（-ss 4s 跳宽屏 intro，保留 workout+outro？） | `douyin_vertical_path` | `stages/39_shorts.py` → `make_vertical(profile='douyin', duration=None)` | **手工**（自动被封号，钉死规则） | `*_douyin.mp4` |
| 3 | **YouTube Shorts**（竖版 30s 短版） | 9:16 / 1080×1920 | **30 秒**（默认，可调 `--shorts-duration`） | `shorts_path` | `stages/39_shorts.py` → `make_vertical(profile='yt_shorts', duration=30)` | **自动**：立即发布 public（受 YT Shorts 版权 60s 规则约束） | `*_yt_shorts.mp4` |

---

## 2. 数据流（实际跑一次发生了什么）

```
C:\Users\18091\Desktop\短视频素材\教练x.mp4   (横拍原始素材)
   │
   │  main.py process xxx.mp4 --preset youtube
   │  (default also: --with-shorts --with-douyin, both on)
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 主流程：Stage 01-37 + 07_export （按 main.py:add_stage 顺序）    │
│                                                                   │
│  01_pose_detect  →  keypoints.json                                │
│       ↓                                                            │
│  02_stabilize (默认关)                                             │
│       ↓                                                            │
│  03_h2v_convert (默认关，因源是横屏, 不需要转 9:16)               │
│       ↓                                                            │
│  04/05 body_warp / face_warp (默认关)                              │
│       ↓                                                            │
│  06_color_grade (默认开)  →  color_path                            │
│       ↓                                                            │
│  17_beat_flash (默认关)                                            │
│       ↓                                                            │
│  18_highlight (默认关)                                             │
│       ↓                                                            │
│  19_energy_bar (默认关, preset 可开)  →  energybar_path             │
│       ↓                                                            │
│  20_intro_outro (默认关, **preset youtube 必开**)                  │
│       横版片头：channel + 带操人 + 地点（**不画判词**）            │
│       ↓                                                            │
│  24_watermark (默认关) → watermark_path                            │
│       ↓                                                            │
│  34_danmaku (默认关) → danmaku_path                                │
│       ↓                                                            │
│  35_intensity_burst (默认关) → burst_path                          │
│       ↓                                                            │
│  37_face_swap (默认开) → face_swap_path (mascot_path 别名)         │
│       ↓                                                            │
│  07_export (默认开) → **final_path** (16:9 1920×1080 含片头片尾)   │
└─────────────────────────────────────────────────────────────────┘
                       │
                       │  final_path 进入 Stage 39
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 39 shorts (ShortsStage) — 单次触发，分两条分支              │
│                                                                   │
│  src = ctx.get("final_path") (fallback: burst/mascot/wm/...)      │
│  audio_src = final_path (不是 source, 避免 4s 错位)               │
│  cx = 前 N 帧最大体型人 cx 中位数 (per-segment 看 segments 数)    │
│  intro_skip = 4s (检测宽屏 intro, -ss 跳)                         │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 分支 A: profile='douyin' (默认开, shorts.yaml.shorts_douyin)│ │
│  │   duration=None (完整)                                     │ │
│  │   + 中文片头（竖版诗词 + 教练雅号）                        │ │
│  │   + 中文竖版 CTA（关注不迷路）                             │ │
│  │   + 抖音特化：关 intro_outro / pip / mascot                │ │
│  │   → 产物: *_douyin.mp4 (9:16 1080×1920, 完整时长)           │ │
│  │     → ctx.douyin_vertical_path                              │ │
│  │     → 上传: 手工 (auto_publish 不上传抖音)                  │ │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 分支 B: profile='yt_shorts' (默认开, shorts.yaml.shorts_yt)│ │
│  │   duration=30 (默认, 可 --shorts-duration 调)             │ │
│  │   + 英文片头 + English subtitle + 中文诗词                  │ │
│  │   + 英文 CTA (LIKE & SUBSCRIBE / Full Workout on Channel)│ │
│  │   → 产物: *_yt_shorts.mp4 (9:16 1080×1920, ≤30s)            │ │
│  │     → ctx.shorts_path                                      │ │
│  │     → 上传: 自动 (shorts 强制 60s 上限版权规则)            │ │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. preset 与产品对应（新旧对照）

### 当前主线（默认主流程）

```bash
python main.py process xxx.mp4 --preset youtube
# default: --with-shorts=true --with-douyin=true
# 走主管线 → final_path + ShortsStage 出两个 9:16 产物
```

| Preset | 用途 | 比例 | 输出 | 备注 |
|--------|------|------|------|------|
| `youtube` | **主管线默认入口** | 16:9 + auto 9:16 | `*_final_16x9` + `*_yt_shorts` + `*_douyin` | 横屏 16:9 主视频 + ShortsStage 出两竖版 |
| `youtube_long` | YT 横屏完整装饰版 | 16:9 / 1920×1080 | 仅 final_path（不走 ShortsStage） | e1267c7 加，区别 youtube.yaml（关 shorts）= 强制只出 final |
| `douyin_long` | 抖音专用长视频 | 9:16 / 1080×1920 | 仅 douyin 完整版 | e1267c7 加，替代旧 `douyin.yaml` |
| `youtube_shorts` | **已弃** = "YouTube Shorts 直接用抖音 9:16 成品裁前 30s" | 16:9 + 9:16 | final + shorts（不走 douyin） | CLAUDE.md 钉死："不单独跑 youtube_shorts preset"。990b9b6 关掉 pip/mascot 让其 ≈ 抖音版（保留兼容兜底） |

### 默认会跑的 Stage 装饰（preset youtube 实际开 stage）

| Stage | enabled | 用途 |
|-------|---------|------|
| 20_intro_outro | true | 4s intro + 5s outro（横版片头**不画判词**，判词只在竖版片头渲染）|
| 37_face_swap | true | 教练换脸（缺 `{coach}_face.jpg/.png` 时自动 skip 不报错）|
| 其他 06_color_grade 等 | 看 preset | preset 控制 |

### Preset 上**默认关闭但常用开**的（preset 内覆盖）

| Stage | youtube | douyin_long | youtube_long |
|-------|---------|-------------|---------------|
| energy_bar | true | true | true |
| danmaku | true | true | true |
| intensity_burst | true | true | true |
| intro_outro | true | **false**（抖音无片头）| true |
| pip | false | false | false |
| mascot | false | false | false |
| watermark | false（youtube_shorts 改关）| true（抖音汉印）| true |

---

## 4. 边界陷阱（容易犯的错）

### 4.1 YT 上传只传 `final_path`，**绝不传 `full_16x9`**

`final_path` = `*_final_16x9_1920x1080.mp4` = **含片头片尾**，是 YT 主视频。
`full_16x9` = `*_full_16x9.mp4` = **去头去尾副本**，**不是上传品**（曾误传过，已删重传）。

### 4.2 YT 必须立即发布 public，**绝不 schedule**

`upload_utils.upload_video()` 对 `video_type=="long"` 强制 `publish_at=None`。scheduled/publishAt 延迟发布会挂死在平台得不到处理（HD processing 卡死）。

### 4.3 抖音坚持手工上传，**绝不能自动**

`auto_publish.py` 不上传抖音（自动被平台检测封号）。抖音产物体 desk 上同步一份供 `~/Desktop/短视频素材/{coach}_{date}_douyin.mp4`，人工传抖音。

### 4.4 YouTube Shorts 版权 60s 上限

`--shorts-duration` 默认 30s；用户曾传 62s 撞规则被禁播，需 ≤60s（推荐 ≤30s 走版权池）

### 4.5 Stage 顺序是 `main.py add_stage` 调用顺序，**不是**文件号

`main.py:345-426` 的 `engine.add_stage(...)` 序列 = 实际执行顺序。`config.yaml` 的 `stages:` 字典只控制 enable/disable。

实际主流程顺序（已加锁，不可改）：
```
pre_deblock → pose → stabilize → h2v → body_warp → ken_burns → face_warp
→ color_grade → skin_smooth → skin_tone → denoise → audio → skeleton_overlay
→ person_count → lead_box → lead_ghost → face_blur → motion_heatmap → sync_score
→ beat_flash → highlight → energy_bar → intro_outro → watermark → mascot
→ face_swap → blush → face_beautify → face_beautify2 → rife → speed_ramp
→ smart_crop → danmaku → intensity_burst → film_look → pip → bgm_beat
→ qin_cold_open → export → shorts
```

### 4.6 三个产物的输出文件名后缀

`*_final_16x9_1920x1080.mp4` ← YouTube 主视频
`*_yt_shorts.mp4`            ← YouTube Shorts
`*_douyin.mp4`               ← 抖音完整版
`*_full_16x9.mp4`            ← 主管线副本（去头去尾，**不用于上传**）

---

## 5. 常见任务 → 改什么

| 要做的事 | 改这个 |
|---------|--------|
| 加新 Stage | `stages/XX_*.py` (新建) + `main.py:add_stage` + `engine.STAGE_OUTPUT_KEYS` |
| 改 YT 长视频装饰 | `stages/01-37` + `config.yaml` + `presets/youtube.yaml` / `presets/youtube_long.yaml` |
| 改抖音特化 | `presets/douyin_long.yaml` + `stages/39_shorts.py:make_vertical(profile='douyin')` |
| 改 YT Shorts | `stages/39_shorts.py:make_vertical(profile='yt_shorts', duration=30)` + `render_short_overlay.py` |
| 改片头诗词/判词 | `lib/coach_profiles.py:COACH_PROFILES.shorts_poem` / `.judgment` |
| 改 YT 标题模板 | `lib/coach_profiles.py` 字段 + `lib/upload_utils.py:build_title()` |
| 改上传逻辑 | `lib/upload_utils.py` + `tools/upload_youtube.py` + `auto_publish.py` |

---

## 6. 验证清单（每跑一条视频查这 6 项）

1. ✅ `final_path` 存在（16:9 1920×1080） → YT 主视频
2. ✅ `shorts_path` 存在（9:16, ≤30s） → YT Shorts（如果开）→ 自动上传
3. ✅ `douyin_vertical_path` 存在（9:16, 完整） → 抖音（人工）
4. ✅ `records/upload_manifest.json` 有 YT 两条 entry（long + short）
5. ✅ `~/Desktop/短视频素材/{coach}_{date}_douyin.mp4` 桌面已有抖音副本
6. ✅ 抖音产物体 desk 同步 OK（手工上传流程就绪）

---

## 7. 与独立工具的边界

| 工具 | 在哪 | 与主管线的关系 |
|------|------|---------------|
| `tools/bg_swap.py` | 本仓库 `tools/` | 独立工具，**不入主管线**（网红换背景+换脸，不走 Stage 39）|
| `tools/prefilter_person.py` | 本仓库 `tools/` | bg_swap 前的预处理，**不入主管线** |
| `tools/student_closeup.py` | 本仓库 `tools/` | 学员特写独立工具，**不入主管线** |
| `tools/face_swap.py` | 本仓库 `tools/` | 换脸核心模块，**被 `stages/37_face_swap.py` 和 bg_swap 都复用** |
| `F:\wkspace\matting-studio\` | **独立仓库** | qml-bg-swap 项目，**与本仓库无 git 关系**（自己 23 commit + v1.0.0）|

**核心边界**：主管线 = `main.py process` 一条命令跑完 → 三产物。独立工具 = 单独命令，针对特定场景。

---

## 8. 决策历史

- **2026-06-27 ShortsStage 重构**：从"douyin preset 独立跑 → yt_shorts 单独跑"合并为"youtube preset 跑一次 → ShortsStage 同时出 yt_shorts + douyin"，节省一整套 Stage 重跑
- **2026-06-29 PIL 渲染修复 ffmpeg 8.1 drawtext UTF-8 bug**：render_short_overlay.py 用 PIL → RGBA PNG → ffmpeg overlay 合成
- **2026-06-29 SHORTS 边界修复**：douyin profile duration=None 修复（之前 fallback 30s）；audio_src=final_path 修复（之前 =source 错位 4s + 截短 190s）
- **2026-07-02 段级裁切修复**：compute_crop_x_segments 治合并视频第二段领操人被裁出
- **2026-07-05 preset 拆分**：youtube_long + douyin_long 替代旧 douyin.yaml，新增 youtube_long 关 shorts 显式 long-only
- **2026-07-05 youtube_shorts 简化**：pip/mascot 关（西方平台不需要东方装饰）
- **2026-07-05 三产品矩阵文档化**：docs/PRODUCTS.md (235 行) 写清"YouTube 主视频 + 抖音 + YouTube Shorts"的边界、来源 stage、上传策略
- **2026-07-05 Matting Studio 设计文档归位**：docs/architecture.md / docs/algorithms.md / docs/ui-design.md 已在头部加 cross-link comment 标注是独立 Matting Studio 项目的"主管线镜像根"，主源在 `F:\wkspace\matting-studio\` 仓库 docs/。主管线不再维护这 3 文件内容

---

**理解优先级**：本文件 = 产品边界矩阵 + 关系图；架构 = `PROJECT_DESIGN.md`；规则 = `CLAUDE.md`；状态 = `HANDOFF.md`；历史 = `memory/`。

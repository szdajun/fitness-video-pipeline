# Fitness Video Pipeline

## 🔖 会话开局协议（新会话第一步，必读）

这是长期项目，会话常因 token 耗尽而重启。**每次开新会话，第一件事是恢复上下文，不要凭空猜：**

1. **读 `HANDOFF.md`** — 当前迭代状态（正在做什么 / 上次停在哪 / 下一步 / 待用户确认）。这是**活文档**，每次会话结束前必须更新。
2. **读 `docs/PROJECT_DESIGN.md`** — 架构与数据流总览（全局已清楚可跳过）。
3. **跑 `git log --oneline -10` + `git status`** — 最近改动与未提交工作。
4. **`memory/MEMORY.md`** — 已自动注入，看历史坑与决策（这些是背景，引用的文件/函数用前先验证还在）。
5. 然后开工。**会话结束前更新 `HANDOFF.md`**（当前进度 / 下一步 / 待办 / 卡点），供下次衔接。

> **权威顺序**：规则查 `CLAUDE.md`（本文件），架构查 `docs/PROJECT_DESIGN.md`，活状态查 `HANDOFF.md`，历史坑查 `memory/`。`docs/` 下带"重构"字样的文档是早期历史归档，**可能过时，不要作为依据**。

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

## 环境管理 (uv)

Python **>=3.11**（3.9 已 EOL 2025-10）。用 **uv** 管 Python 版本 + 依赖，`uv.lock` 锁定可复现。

```bash
# 首次/拉新依赖: uv 按 .python-version (3.11) + pyproject.toml 建 .venv 并装依赖
uv sync --extra dev          # dev=pytest; 加 --extra gpu 装 torch (RVM 抠像用)

# 跑管线/测试 (uv run 自动用 .venv, 无需手动 activate)
uv run python main.py process "input.mp4" --preset youtube --shorts-coach 丽丽
uv run pytest tests/ -q

# GPU torch (RVM/bg_swap 用; 默认 ultralytics 只拉 CPU torch)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

- **不要**直接 `python main.py`（pyenv shim 可能指向 3.9）；用 `uv run` 或 `.venv/Scripts/python`。
- **不要**碰 `F:/wkspace/ComfyUI/venv/`（3.11.9，SAM2/bgswap 子进程用，独立环境，借用其权重 + custom_nodes）。本项目与 ComfyUI **子进程解耦，版本无需对齐**。
- numpy 钉 `<2`（2.x 有 breaking，升前需验证）。`.venv/` 入 `.gitignore`，`uv.lock` **入 git**（可复现）。

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

## Design Decisions (2026-06-25)

以下决策已写入代码，不要再改回去：

### 永久关闭的 Stage

| Stage | 原因 |
|-------|------|
| `skin_smooth` | CPU 3.3h/次，换脸已覆盖面部美颜；健身出汗场景不需要全帧磨皮 |
| `mascot` | 占用 1.4GB + 9min，健身视频不应有吉祥物遮挡 |
| `pip` | 画中画干扰跟练，无意义 |

### smart_crop v21 — 拼接视频自动分段

`stages/38_smart_crop.py` v21：滑动窗口中位数检测 cx 突变点（>0.08），
自动识别正面→背面机位切换，各段独立算稳定 cx。不再硬编码帧范围。

### export 链修复

`stages/07_export.py`：fallback 链补充 `danmaku_path`，确保弹幕/水印/能量条
不被跳过。顺序：`burst → danmaku → mascot → watermark → energybar → ...`

### 关键 Bug 修复

- **energy_bar ffmpeg 路径** (`stages/19_energy_bar.py`)：优先用 `C:/Users/18091/ffmpeg/ffmpeg.exe`，
  Winget 版本有编码兼容问题会生成损坏 mp4

### YouTube 上传标题模板（钉死的规则）

**模板**： `【{nickname}】{coach}{focus}操 | {focus}跟练 | 细柳营健身`

**示例**：
- 【老兵不老】郭海军力量燃脂操 | 刚劲塑形跟练 | 细柳营健身
- 【胭脂虎】艳青暴汗燃脂操 | 塑腰弯跟练 | 细柳营健身

**来源**：`lib/upload_utils.py:build_title()`，从 `lib/coach_profiles.py:COACH_PROFILES.{coach}.nickname/focus` 自动取。
**不要改**为 `细柳营·{coach} | 有氧健身操·燃脂暴汗 | {date}` —— 旧版扁平标题，已弃用。

上传完成后必须 `records/upload_manifest.json` 留痕（`lib/upload_utils.upload_video()` 自动写）。

### Face Swap 流水线（2026-06-26 验证可用）

**核心配置**（钉死，不要改）：
- `gfpgan_strength: 0` — GFPGAN 跑 CPU 要 7h/视频, 跑 GPU 也只修"假脸"无意义。**完全关闭**。
- `min_face_area: 0.001` — 健身视频全身镜头里领操人脸只占 0.5-1%, 旧 0.02 会过滤掉所有人。
- `color_match_strength: 0.8` — LAB Reinhard 迁移, 消除 inswapper 输出和场景的明暗不均。
- `output.encoder: nvenc` + `output.prefer_gpu: true` — h264_nvenc 加速 export **48%** (275s → 142s)。

**Lead 检测策略**（2026-06-26 解决"间歇性换脸不均"):
- pose keypoints 找 cx 居中 + 身体最大的 person = 领操人
- bbox 大小 = `max(肩宽 × 1.5, 160px)`. ROI 太小 (<160) insightface 漏检, 太大远处小脸混入被误选.
- ROI 内用 **`min(dist_to_center, -area, -det_score)`** 选 lead —— 选 cx 接近 ROI 中心的, 不是最大脸. 避免远处大婶被误选.
- 朝向过滤: 鼻子-肩膀中点偏移 > 0.10 → 'back', 跳过换脸.

**已知限制**:
- 领操人远景/仰头 (郭海军1) 时 insightface buffalo_l 仍漏检 → 视觉上看不出换
- 解决方向: 换 SCRFD/YOLO-face 检测器; 当前接受这个限制

**Pre-commit 守门**:
- `tests/test_face_swap_no_gfpgan.py` — gfpgan_strength=0 不能改
- `tests/test_face_swap_min_face_area.py` — min_face_area ≤ 0.001
- `tests/test_face_swap_lead_selection.py` — cx 中心选脸不能改 max area
- `tests/test_upload_manifest_dedup.py` — 同一 file/ytid 不重复

### 教练换脸资源

`tools/{coach}_face_gfpgan.png` (1024×1024) — GFPGAN 重建过的高清照.
- 优先级: `{coach}_gfpgan.png > {coach}_face.png > {coach}_face.jpg > {coach}.png`
- 缺图时 face_swap 自动 skip (不报错).
- 加新教练: 丢一张清晰照到 `tools/{coach}.jpg` 即可, 首次跑时自动 GFPGAN 增强生成 `_gfpgan.png` 长期复用.

- **pip timeout** (`stages/31_pip.py`)：120s → 600s，长视频编码不会超时
- **lib/seal.py**：汉印水印叠加（AI PNG `tools/seal_ai.png` 优先 + PIL 篆体朱文兜底），被 `stages/24_watermark.py` 调用；接口 `overlay_seal(frame, text, pos, size, margin, alpha, **kwargs)`
- **pre-commit hook 建议**：拒绝 >10MB 文件（防大视频再误入 git）

### 平台策略

| 平台 | Preset | 格式 | 时长 | 上传 |
|------|--------|------|------|------|
| YouTube | `youtube` | 16:9 1920×1080 | 完整 | **立即发布(public)** |
| YouTube Shorts | `douyin` 30s cut | 9:16 1080×1920 | 30 秒 | **立即发布(public)** |
| 抖音 | `douyin` | 9:16 1080×1920 | 完整 | 人工 |
| ~~小红书~~ | — | — | — | **放弃** (3:4 无增量价值) |

YouTube Shorts 直接用抖音 9:16 成品裁前 30 秒，不单独跑 youtube_shorts preset。

**⚠ YT 上传必须"立即发布"**（`privacy=public`, `publish_at=None`）— scheduled/延迟发布（`publishAt`）的长视频近期全部**挂死在平台得不到处理**（HD processing 卡死）。代码 `upload_pair`/`tools/upload_youtube.py` 默认即立即，`upload_utils.upload_video` 对 `video_type=="long"` 强制 `publish_at=None`（即便传了也忽略+告警）。**别加 `--publish-at`/schedule**。详见 memory `yt-long-video-publish-immediately`。

### .gitignore 原则

大文件绝不入 git：`*.mp4` `*.pt` `*.bin` `*.zip` `source_videos/` `output/` 
`_temp/` `models/` `weights/` `gfpgan/` `venv/` `music_library/` `coach_portraits/`

## Conventions

- Python 3.9+，依赖 OpenCV / ultralytics / FFmpeg / NumPy / PyYAML
- 体型参数范围：slim 0.7~1.0, enlarge/lengthen 1.0~1.4
- Stage 编号前缀 (`01_`, `02_`) 表示执行顺序，不可随意调换
- `lib/` 目录存放跨 stage 共享的底层模块
- `output/` 目录存放处理结果和中间文件
- `source_videos/` 放原始素材，不入 git

## Files for Distribution

- `README.md` — 项目概述、快速开始、命令参考
- `docs/manual.md` — 完整用户手册
- `docs/FACE_SWAP.md` — 换脸流水线经验总结
- `docs/BG_SWAP.md` — 网红换背景+换脸独立工具经验总结 (`tools/bg_swap.py` + `tools/prefilter_person.py`, 预设 `bgswap_{fitness,clean,dance}.yaml`)
- `presets/README.md` — 预设风格详解 (含 bgswap 预设段)
- `requirements.txt` — Python 依赖
- `pyproject.toml` — 项目打包配置

## 独立工具 (`tools/`, 主管线零改动)

- `tools/bg_swap.py` — 网红视频换背景 (默认西安时代广场) + 换脸. RVM 抠像 + only_lead 换脸 + 色温匹配 + 接地感 + 静态背景 + **pose core-matte 撑实胳膊 (治虚化/渗出, `--core-bolster` 默认开)**. `--preset fitness|clean|dance`. **经验查 `docs/BG_SWAP.md` (坑 9 = core-matte)**, 守门 `tests/test_bg_swap_defaults.py` (14 tests). ffmpeg 走 `_resolve_ffmpeg()` (已知好路径 `C:/Users/18091/ffmpeg/ffmpeg.exe` 优先于 PATH, Winget 版有编码 bug).
- `tools/prefilter_person.py` — 换背景前清洗: pose 逐帧判人物完整性, 剪掉出画/缺头缺脚片段. 配合 bg_swap 用.
- `tools/student_closeup.py` — 学员特写 (认人+推近+暖调+节拍闪).
- `tools/face_swap.py` — 换脸核心 (被 stages/37 和 bg_swap 复用).


## Post-2026-06-27 Pipeline Improvements (verified working on 丽丽2)

### 1. ShortsStage 单入口重构 (`stages/short_vertical.py`)

**新逻辑**: 跑 youtube preset 一次, ShortsStage 同步产出 YT Shorts + 抖音竖版.
- `make_vertical(src, profile, duration)` 单入口
- profile='yt_shorts' → 30s + 英文片头+CTA
- profile='douyin' → 完整版 + 中文片头, **自动截掉 intro 4s + outro 5s** (-ss/-t)
- cx 自适应裁切: 前 60 帧 cx 中位数 → 静态 crop_x (复用 face_swap.find_lead_person)
- padding ±30px clamp, kp 缺失 fallback 居中

### 2. 新增 CLI flag (`main.py`)

```
--reset-gpu                跑 pipeline 前清残留 GPU 状态 (杀残留进程 + reset clocks + torch empty_cache)
--with-shorts / --no-shorts  生成 YouTube Shorts (30s) (默认开)
--with-douyin / --no-douyin 生成抖音竖版完整版 (默认开)
--shorts-duration <sec>    Shorts 时长 (默认 30)
--shorts-coach <name>      教练名 (用于片头诗词 + 英文标题)
```

### 3. Face Swap 可靠性修复 (核心 - 解决 CUDNN 崩溃)

**问题**: face_swap 跑到 ~500 帧 onnxruntime CUDNN_STATUS_EXECUTION_FAILED (Conv_73 节点) 进程 abort.

**根因**:
- GPU 长时间跑 pipeline 后 onnxruntime arena 碎片化 (kNextPowerOfTwo 默认策略)
- `ctx.keypoints_file` 没人 set, pose bbox 分支永远走 fallback (老路径)

**修复 (3 处)**:
- `stages/01_pose_detect.py`: `ctx.set("keypoints_file", cache_path)` (cache hit + miss 两处)
- `tools/face_swap.py`: providers 加 `arena_extend_strategy='kSameAsRequested'` + `gpu_mem_limit=8GB`
- `tools/reset_gpu.py` (新建): 跑前 `--reset-gpu` 触发, 杀残留 + reset clocks + torch empty_cache

**效果**: 丽丽2 之前永远跑不到 500 帧就崩, 现在 2898/2898 帧 swap 完成, swap_count 100%, back=0 (无背面跳过).

### 4. FFmpeg drawtext 路径坑 (Windows)

**Bug**: filter 字符串里 `'C:/Windows/Fonts/msyh.ttc'` ffmpeg 把 `C:` 当相对路径, 报 `No option name near '/Windows/...'`.
**Fix**: 在 `stages/short_vertical.py` 写文件前 `vf.replace("C:/", "C\\:/")` (双反斜杠 + 冒号).

### 5. emoji print GBK 坑

**Bug**: `youtube_upload.py` 用 ⚠ ✓ 等 emoji print, Windows GBK 编码崩.
**Fix**: 已替换为 [WARN] [OK] ASCII. 同样适用未来 print emoji 的代码.

### 6. YT 上传路径坑

**关键区别** (CLAUDE.md 之前没写):
- `final_path = *final_16x9_1920x1080.mp4` → **main, 含片头片尾** (用于 YT 主视频, 105.6s for 丽丽2)
- `full_16x9 = *_full_16x9.mp4` → **去头去尾 16:9 副本** (不用于 YT 上传)

**Always** 上传 `final_path`, NOT `full_16x9`. (我刚传错过, 已删 + 重传).

### 7. 上传 manifest 必填

`records/upload_manifest.json` 记录 ytid + file + title + privacy + uploaded_at.
- 写失败不影响上传 (try/except 包了)
- 但 manual 删除 video 后必须同步删 manifest entry (否则下次跑会以为没传过)

### 8. douyin preset 现状 (可选快路)

保留为可选: 想跑抖音干净版 (无汉印无mascot无能量条) 可单独 `--preset douyin`.
默认主流程是 `--preset youtube` → ShortsStage 自动出 抖音完整版 (含所有效果).


## ~~2026-06-27 ShortsStage CTA 已知问题 (ffmpeg 8.1 bug)~~ 【已解决 2026-06-29】

> **已解决**: `stages/render_short_overlay.py` 用 PIL 把片头(英文标题+副标+中文诗词)和 CTA(SUBSCRIBE+红分割线)渲染成 1080×1920 RGBA PNG, `short_vertical.py:make_vertical` 用 ffmpeg `overlay=0:0:enable='between(t,a,b)'` 合成 (**非 drawtext**, 不受 8.1 bug 影响). 艳青1 实测 yt_shorts: t=1.5s 片头黄字 22439px, t=28s CTA 黄字 21500px+红线 3456px, **overlay 正常渲染**. 下面是历史根因记录, 保留以防回退.

ShortsStage 跑通 (cx 裁切, intro 跳过, 抖音完整版都 OK), 但 **YouTube Shorts 视频画面里没有 CTA 文字** (点赞 LIKE & SUBSCRIBE 关注 / 完整版 Full Workout on Channel / 新视频 New Videos Daily).

**根因**: ffmpeg 8.1 drawtext filter chain parser bug:
- `text='...UTF-8 中文...'` 单引号字符串内含多字节 UTF-8 字符后, parser 静默截断/失败
- `alpha='if(lt(t,a), b, if(lt(t,c), d, ...))'` 嵌套 if() + 逗号 在 8.1 长 chain 里也解析失败
- 失败的 drawtext ffmpeg 静默跳过, 不报错, 不画文字

**已尝试的 fix (都没成功)**:
- `_escape_ffmpeg_text` 加 `: ` `\` `'` `/` 转义
- `textfile=` 重写 (FFMPEG 读 UTF-8 文件, 避开 string parser)
- `C\:/` 路径 escape
- `-filter_complex` 直接传字符串 (不用文件)
- ASCII 化所有 subtitle 和 CTA 文案
- `enable=between(t;0.5;3)` 用 `;` 替代 `,`

**【已解决 2026-06-29】PIL 方案生效**: `stages/render_short_overlay.py` (`render_opening`/`render_cta`) 用 PIL 渲染文字到 RGBA PNG, `make_vertical` 用 ffmpeg `overlay` 滤镜合成, 避开 drawtext. opening/CTA 现在正常画 (黄字像素验证 >2000). **不要回退到 drawtext 链**, 会再触发 8.1 bug.

**验证 overlay 渲染** (别靠日志 [OK], 靠像素): `ffmpeg -ss 1.5 -i shorts.mp4 -frames:v 1 f.png` 抽帧, 检测黄字 `(R>200)&(G>170)&(B<110)` 像素数 >2000 = 渲染了.

**附带修复 (2026-06-29 同期)**: make_vertical 还有两个 bug 一并修了 (详见 memory `shorts-vertical-duration-audio-bug`):
- douyin(duration=None) 被 fallback 成 30s → 改用完整时长 (douyin 现 194s)
- audio_src 用 source 导致音视频错位 4s + 截短到 190s → 改用 final_path 对齐

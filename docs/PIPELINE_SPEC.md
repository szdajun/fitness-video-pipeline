# 健身视频处理流水线 — 设计规范与操作手册

> **目的**: 记录输出规格、Pipeline 流程、各平台差异、设计决策与经验教训。  
> **受众**: 项目维护者与 AI 编程助手。  
> **最后更新**: 2026-06-06

---

## 一、输出格式矩阵

| 平台 | Preset | 主输出尺寸 | 视频规格 | 核心特征 |
|------|--------|-----------|---------|---------|
| **YouTube 长视频** | `youtube` / `shorts` | 1920×1080 | 16:9 横版 | 完整片头片尾 + 全部装饰元素 |
| **YouTube Shorts** | `youtube_shorts` | 1920×1080 + 1080×1920 副本 | 16:9 横版 + 9:16 Shorts | 英文标题 + 教练英文名 + 诗词 + 双语 CTA |
| **抖音** | `douyin` | 1080×1920 | 9:16 竖版 | 教练居中裁切 + 换脸 + 弹幕 + 节拍闪光 |
| **小红书** | `xiaohongshu` | 1080×1440 | 3:4 竖版 | 教练居中裁切 + 换脸 + 弹幕 |
| **通用 Shorts** | `shorts` | 1920×1080 | 16:9 横版 | 全装饰（intro/outro/PIP/mascot/danmaku/能量条）|

> **关键原则**: 16:9 源视频是唯一的原始素材。竖版 (9:16/3:4) 全部从横版裁切而来，**不做 H2V 转换**。裁切以领操人水平位置 (`lead_cx`) 为中心。

---

## 二、Preset 详细配置

### 2.1 `shorts` (通用 Shorts — 全装饰横版)

```
h2v_convert: false     ← H2V 禁止使用，16:9 源保持 16:9
body_warp: false       ← H2V 关后必须关 body_warp（否则 h2v_size=None 崩溃）
intro_outro: true      ← 片头片尾全开
pip: true              ← 画中画右下
mascot: true           ← 吉祥物左下
danmaku: true          ← 弹幕
energy_bar: true       ← 能量条
face_swap: true        ← 换脸（必须在 mascot 之后、danmaku 之前跑）
color_grade: true      ← CLAHE+sharpen
```

### 2.2 `douyin` (抖音 — 简洁竖版)

```
h2v_convert: false
body_warp: false
intro_outro: false     ← 关掉！16:9 渲染的文字在 9:16 裁切后会被切断
pip: false             ← 关掉！9:16 画面小
mascot: false          ← 关掉！画面小
energy_bar: false      ← 关掉！画面小
danmaku: true          ← 保留弹幕
beat_flash: true       ← 保留节拍
face_swap: true        ← 换脸
watermark.show_seal: false  ← 关掉汉印，画面小不放装饰
```

### 2.3 `xiaohongshu` (小红书 — 简洁 3:4)

```
同 douyin，但 output: 1080×1440 (3:4)
```

### 2.4 `youtube_shorts` (YouTube Shorts)

```
同 shorts 全装饰 + formats: ['9x16'] 多格式副本
```

---

## 三、Pipeline Stage 执行顺序

### 3.1 全部 Stage（按顺序，不可随意调换）

| 序号 | Stage | 文件 | 默认 | 关键依赖 |
|------|-------|------|------|---------|
| 0 | preflight | `00_preflight.py` | ✅ | 校验视频分辨率 ≥720p |
| 1 | pose_detect | `01_pose_detect.py` | ✅ | YOLO GPU 检测关键点 |
| 2 | pre_deblock | `00_pre_deblock.py` | ❌ | |
| 3 | stabilize | `02_stabilize.py` | ❌ | |
| 4 | **h2v_convert** | `03_h2v_convert.py` | ❌ | **禁止使用**，见红线 #1 |
| 5 | body_warp | `05_body_warp.py` | ❌ | 依赖 h2v_size，H2V 关闭时不可用 |
| 6 | ken_burns | `04_ken_burns.py` | ❌ | |
| 7 | face_warp | `08_face_warp.py` | ❌ | |
| 8 | **color_grade** | `06_color_grade.py` | ✅ | CLAHE+sharpen，**整个 Pipeline 第一个真正处理的 stage** |
| 9 | **skin_smooth** | `21_skin_smooth.py` | ✅ | 输入链必须含 `color_path`，磨皮在 CLAHE 增强后做 |
| 10 | skin_tone_filter | `22_skin_tone_filter.py` | ❌ | |
| 11 | denoise | `23_denoise.py` | ❌ | |
| 12 | audio | `09_audio.py` | ❌ | |
| 13 | beat_flash | `17_beat_flash.py` | ✅ | 节拍闪光，**zoom_factor=1.0 关闭缩放** |
| 14 | highlight | `18_highlight.py` | ✅ | |
| 15 | energy_bar | `19_energy_bar.py` | 看平台 | 计算 lead_cx 供 export 裁切 |
| 16 | **intro_outro** | `20_intro_outro.py` | 横版✅ 竖版❌ | 16:9 全幅渲染，竖版裁切后文字被切 |
| 17 | watermark | `24_watermark.py` | ✅ | 水印文字 + 汉印（可关） |
| 18 | **mascot** | `29_mascot.py` | 横版✅ 竖版❌ | 吉祥物小老虎 |
| 19 | **face_swap** | `37_face_swap.py` | ✅ | **必须在 mascot 后、danmaku 前** |
| 20 | blush | `25_blush.py` | ❌ | |
| 21 | face_beautify | `26_face_beautify.py` | ❌ | |
| 22 | face_beautify2 | `27_face_beautify2.py` | ❌ | 死代码（workers 从未实现） |
| 23 | rife | `28_rife_interpolate.py` | ❌ | |
| 24 | speed_ramp | `32_speed_ramp.py` | ❌ | |
| 25 | danmaku | `34_danmaku.py` | ✅ | 弹幕 |
| 26 | intensity_burst | `35_intensity_burst.py` | ✅ | 爆燃大字 |
| 27 | film_look | `33_film_look.py` | ❌ | 发雾排查时暂关 |
| 28 | pip | `31_pip.py` | 横版✅ 竖版❌ | 画中画，用 main_path 实际尺寸 |
| 29 | bgm_beat | `30_bgm_beat.py` | ❌ | |
| 30 | qin_cold_open | `36_qin_cold_open.py` | ❌ | |
| 31 | **export** | `07_export.py` | ✅ | 最终编码 + 教练裁切 + Shorts |
| 32 | face_enhance | `30_face_enhance.py` | ❌ | |

### 3.2 关键 Stage 顺序（不可调换）

```
color_grade → skin_smooth → beat_flash → energy_bar → intro_outro → watermark
→ mascot → face_swap → danmaku → intensity_burst → pip → export
```

**为什么这个顺序**:
1. `color_grade` 先跑 CLAHE+sharpen 提升对比度
2. `skin_smooth` 在增强后的画面上磨皮（输入链含 `color_path`）
3. `mascot` 生成吉祥物路径
4. `face_swap` **紧跟 mascot 后**，改写 `mascot_path` 为换脸版——后续 danmaku/pip/export 都用换脸版
5. `export` 最后：片头+主体+片尾拼接 + 教练居中的 9:16 裁切

---

## 四、红线（禁止事项）

### 1. H2V (h2v_convert) 永久禁用 ❌

**原因**: 
- H2V 阶段把 16:9 源强制裁切成 9:16，导致"视频只占中间一小块"
- 用户原话: "H2V 总是出各种问题"、"原视频就是宽幅，不需要转竖屏"
- 竖屏裁切现在由 **export 阶段**根据 `lead_cx` 以教练为中心裁切

**替代方案**: export 阶段读 `ctx.get("lead_cx", 0.5)` 定位裁切窗口

### 2. body_warp 必须跟随 h2v 状态

H2V 关闭时 `body_warp` 必须关闭，否则 `h2v_size=None` 导致 `crop_w, crop_h = h2v_size` 解包崩溃。

### 3. face_swap Stage 顺序不可动

`face_swap` 必须在 `mascot` 之后、`danmaku` 之前。  
原因: face_swap 改写 `mascot_path` → 后续 stage 用换脸版。

### 4. 不要轻易修改 `07_export.py` 的 auto-adjust 逻辑

export 已禁用"自动方向调整"（之前会把 9:16 改回 16:9）。  
用户通过 `--preset` 显式指定输出尺寸，以 config 为准。

### 5. skin_smooth 输入链必须含 `color_path`

否则 CLAHE+sharpen 白跑，正片发雾。

### 6. 16:9 源 → 9:16 输出用 "增大+裁切"，不是 letterbox

`force_original_aspect_ratio=increase` + `crop`——填满画面，裁掉多余。
**不是** `force_original_aspect_ratio=decrease` + `pad`——那会导致画面缩到中间一小块。

### 7. FFmpeg 编码参数统一用 `yuv420p`

`yuv444p` 在 FFmpeg 8.1 Gyan build 上有兼容性问题（只给版本 banner 不给错误）。

### 8. 竖版视频不要 intro_outro

16:9 全幅渲染的片头片尾文字在 9:16 裁切后会被切断。  
竖版用 watermark 文字代替品牌展示。

### 9. 竖版视频装饰元素精简

9:16 幅面小，去掉 mascot（吉祥物）、energy_bar（能量条）、pip（画中画）、汉印（seal）。
保留：换脸 + 弹幕 + 节拍闪光 + 爆燃大字 + watermark 文字。

### 10. 永不删 output 目录成品

CLAUDE.md 红线 #1。

---

## 五、关键设计决策

### 5.1 教练居中裁切

**背景**: 原来的 H2V 做"领操人追踪 + 裁切"，但 H2V 问题太多被禁用。

**新方案**: export 阶段根据 `lead_cx`（energy_bar stage 追踪的领操人水平位置）计算裁切窗口。

```
lead_cx = clamp(ctx.get("lead_cx", 0.5), 0.25, 0.75)
crop_w = in_h * 9/16          # 从 16:9 源裁 9:16 窗口
crop_x = lead_cx * in_w - crop_w/2
filter: crop=W:H:X:0,scale=1080:1920
```

**安全钳**: `lead_cx` 限制在 [0.25, 0.75]，防止追踪错误导致教练被裁出画面。

### 5.2 换脸 (face_swap)

- 依赖 `tools/face_swap.py`（InsightFace），处理 6 位教练的美颜照
- 美颜照位置: `tools/coach_gfpgan.png`（预处理好的 GFPGAN 增强照）
- face_swap 改写 `mascot_path`，后续 stage 自动用换脸版
- 换脸内部故障时保留 `_tmp_vid.mp4` 供上层 stage fallback

### 5.3 去雾三板斧

| 修复 | 效果 |
|------|------|
| CLAHE 真正生效 | `needs_grade` 检查 `clahe=True` 时不被误判跳过 |
| skin_smooth 吃 CLAHE | 输入链首加 `color_path` |
| 关闭 beat_flash zoom | `zoom_factor=1.0` 消除镜头推拉感 |

### 5.4 自动优化

| 功能 | 触发条件 | 效果 |
|------|---------|------|
| auto CLAHE 跳过 | 画面 LAB L > 128 | 白天户外自动跳过 CLAHE |
| 磨皮自动降压 | face_swap 启用 + strength > 0.1 | 降到 0.05（面部已有美颜照）|

### 5.5 视频模糊问题排查

**不是 film_look 造成的**（film_look: false）。原因链:
1. skin_smooth 双边滤波柔化
2. color_grade 的 `needs_grade` bug 导致 CLAHE 不生效（已修）
3. skin_smooth 输入链缺 `color_path`（已修）
4. 720p 源放大到 1080p（用原生 1080p 源可解决）

### 5.6 弹幕字号

- 16:9 横版: `font_size: 50`（config.yaml 默认）
- 9:16 竖版: `font_size: 36`（shorts.yaml）/ `48`（douyin.yaml）
- 竖版弹幕建议缩小，避免遮挡教练

### 5.7 汉印 (Seal)

- 左上角红色"胭脂虎"印章
- 16:9 横版: 全程显示，alpha=0.70, size=130
- 9:16 竖版: `watermark.show_seal: false` 关掉
- 实现: `lib/seal.py` → `stages/24_watermark.py:190` 调 `overlay_seal()`

### 5.8 YouTube Shorts 特性

`_make_shorts.py` 生成 30 秒精华，叠加:
- 英文标题 (e.g. "DAILY AEROBIC WORKOUT")
- 教练英文名 + 英文副标题
- 中文诗词 (4 行，每教练专属)
- 双语 CTA ("点赞 LIKE & SUBSCRIBE 关注")
- 繁体中文支持 (`seo.traditional: true`)

---

## 六、常见问题速查

| 问题 | 原因 | 修复 |
|------|------|------|
| 视频只占中间一小块 | H2V 复用了旧裁切缓存 | engine 禁用 stage 时跳过缓存扫描 |
| 9:16 输出变成 16:9 | auto-adjust 方向检测错误 | 禁用了 auto-adjust |
| 人物被拉长 | `scale=` 直接拉伸 | 改用 `force_original_aspect_ratio=increase,crop` |
| 竖版片头片尾文字被切 | intro 在 16:9 全幅渲染→裁切 | 竖版关 intro_outro |
| 戴操人被裁出画面 | lead_cx 追踪错误 | clamp [0.25, 0.75] |
| 正片发雾 | skin_smooth 没用 CLAHE 输出 | 输入链加 color_path |
| CLAHE 不生效 (0.0s) | `needs_grade` 把 `True==1.0` 跳过 | clahe 单独检查 |
| 换脸不进最终视频 | face_swap 在 pip 后跑 | 移到 mascot 后 |
| FFmpeg 崩溃不给错误 | FFmpeg 8.1 + yuv444p + -v info | 改 yuv420p + -v error |
| `body_warp NoneType` 崩溃 | H2V 关闭时 h2v_size=None | body_warp 同步关闭 |
| `process_isolate` 子进程 GPU crash | 子进程栈溢出 | 关闭 process_isolate |

---

## 七、文件索引

| 文件 | 职责 |
|------|------|
| `main.py` | CLI 入口，Stage 注册与顺序 |
| `pipeline/engine.py` | Pipeline 引擎，增量恢复，_scan_existing_outputs |
| `pipeline/config.py` | 配置加载，deep_merge，known_keys 校验 |
| `pipeline/process_stage.py` | 进程隔离包装器（已关闭）|
| `pipeline/manifest.py` | 增量恢复 Manifest |
| `stages/07_export.py` | **核心**: 最终编码 + 教练裁切 + 多格式分发 + Shorts |
| `stages/06_color_grade.py` | CLAHE+sharpen + auto CLAHE 光线检测 |
| `stages/21_skin_smooth.py` | 磨皮 + 换脸自动降压 |
| `stages/17_beat_flash.py` | 节拍闪光 (zoom 已关闭) |
| `stages/19_energy_bar.py` | 能量条 + lead_cx 追踪 |
| `stages/20_intro_outro.py` | 片头片尾（仅横版用）|
| `stages/24_watermark.py` | 水印文字 + 汉印 |
| `stages/29_mascot.py` | 吉祥物小老虎 |
| `stages/37_face_swap.py` | 换脸 Stage |
| `stages/34_danmaku.py` | 弹幕 |
| `stages/31_pip.py` | 画中画 |
| `stages/35_intensity_burst.py` | 爆燃大字 |
| `tools/face_swap.py` | InsightFace 换脸核心 |
| `lib/coach_profiles.py` | 教练画像，诗词，英文标题 |
| `lib/seal.py` | 汉印生成 |
| `lib/utils.py` | 领操人追踪，中文字体渲染 |
| `_make_shorts.py` | YouTube Shorts 30s 生成 |

---

## 八、历史 Commit 里程碑

```
41972f8  fix: 抖音/小红书竖版关 intro_outro
8da3ea2  fix: 9:16 以教练 lead_cx 为中心裁切
1e78fcd  refactor: 抖音/小红书简洁化
cbdb5f9  fix: 9:16 letterbox防拉伸 + color_grade FFmpeg容错
01c41b2  fix: export auto-adjust禁用 + _make_shorts重接 + watermark容错
aa06867  feat: 注册抖音/小红书/YouTube Shorts 三平台预设
987ab2e  fix: 汉印全程显示 + 透明度+放大
fadf0a8  feat: auto skin_smooth降压 + auto CLAHE 光线检测
f72603b  fix: 去雾三板斧 — CLAHE生效 + skin_smooth链 + 关闭节拍缩放
94d6d46  fix: color_grade CLAHE+sharpen 去雾
219cbc6  fix: face_swap 移到 mascot 之后
3d987a6  fix: 换脸输出丢失 + 视频模糊
d6b1b42  feat: 注册 face_swap stage
41bf427  fix: engine 禁用 stage 不复用旧缓存
```

---

## 九、快速命令参考

```bash
# 抖音竖版 (9:16, 简洁)
python main.py process "source_videos/xxx.mp4" --preset douyin --full-video

# 小红书 (3:4, 简洁)
python main.py process "source_videos/xxx.mp4" --preset xiaohongshu --full-video

# YouTube Shorts (16:9 + 9:16 副本)
python main.py process "source_videos/xxx.mp4" --preset youtube_shorts --full-video

# 通用 Shorts (16:9 全装饰)
python main.py process "source_videos/xxx.mp4" --preset shorts --full-video

# YouTube 长视频 (16:9)
python main.py process "source_videos/xxx.mp4" --preset youtube --full-video
```

---

## 十、上传与发布流程

### 10.1 平台上传方式

| 平台 | 上传方式 | 工具 | 状态 |
|------|---------|------|------|
| **YouTube** | API 自动上传 | `youtube_upload` Python 库 | ✅ 完整 |
| **抖音** | 网页自动上传 | Playwright / Chrome DevTools MCP | ⚠️ 需浏览器工具 |
| **小红书** | 网页自动上传 | Playwright / Chrome DevTools MCP | ⚠️ 需浏览器工具 |

### 10.2 YouTube 完整上传流程

```bash
# 1. 生成视频
python main.py process "xxx.mp4" --preset youtube --full-video

# 2. 查看 SEO 元数据
cat output/日期/xxx_seo.json

# 3. 一键上传（从 SEO 文件读标题/描述/标签）
python publish.py upload output/日期/xxx_final_16x9.mp4 --seo output/日期/xxx_seo.json

# 4. 或加入发布队列定时发布
python publish.py add output/日期/xxx_final_16x9.mp4 --coach 枫林红 --type long
python publish.py run
```

### 10.3 抖音/小红书上传（浏览器自动化）

抖音和小红书无公开 API，需通过网页版创作者中心上传。项目已集成 Playwright + Chrome DevTools MCP 工具。

**流程概要**:
1. Pipeline 生成视频 + 封面 + SEO 元数据
2. 通过 Playwright 打开抖音/小红书创作者中心
3. 自动填表：标题、描述、标签、封面图
4. 上传视频文件
5. 提交发布

**需要的 MCP 工具**: `mcp__playwright__*` 或 `mcp__chrome-devtools__*`

**关键文件**:
| 文件 | 职责 |
|------|------|
| `publish.py` | CLI 入口，发布调度 |
| `lib/publisher.py` | 发布队列引擎，JSON 持久化 |
| `lib/upload_utils.py` | YouTube API 上传 + SEO 构建 |
| `auto_publish.py` | 全自动: Pipeline → 封面 → 上传 |

### 10.4 发布队列

`lib/publisher.py` 基于 JSON 文件 (`publish_queue.json`) 的持久化队列:

```
长视频: 周一/周四 16:00 (观众高峰前 2-3h 发布)
Shorts: 每天 07:00 + 12:30 (通勤 + 午休高峰)
周末:   10:00 + 19:00
```

### 10.5 各平台发布注意事项

| 平台 | 标题限制 | 标签限制 | 封面要求 | 其他 |
|------|---------|---------|---------|------|
| YouTube | ≤100 字符 | 不限 | 16:9 1280×720 | 可定时发布 |
| YouTube Shorts | ≤100 字符 | #Shorts 必加 | 9:16 | ≤60 秒 |
| 抖音 | ≤55 字符 | 建议 5-8 个 | 3:4 或 9:16 | 手机端发布 |
| 小红书 | ≤20 字符 | 建议 3-5 个 | 3:4 最佳 | 标题短小精悍 |

---

> **维护提醒**: 每次修改 Pipeline 逻辑后，请同步更新本文档。特别关注"红线"部分——那是反复踩坑总结的教训，不要再犯。

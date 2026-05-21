# 胭脂虎健身视频管线 — 移交文档

## 一句话概述

视频丢进素材目录 → `python auto_publish.py` → 全自动处理上传，零人工介入。

## 目录结构

```
F:\wkspace\fitness-video-pipeline\
├── auto_publish.py          ★ 日常唯一入口
├── main.py                  完整管线 CLI（14 阶段特效）
├── batch_publish.py         旧入口（已修复 OpenCV，保留备用）
├── config.yaml              管线配置（启用的阶段列表）
├── coaches.yaml             教练映射（艳青→胭脂虎、丽丽→腰女…）
├── day_counter.json         日计数器（每个教练的打卡 day）
├── auto_publish_state.json  已处理文件记录
├── auto_publish.log         运行日志
├── stages/                  管线阶段（35 个模块）
├── presets/                 风格预设
├── output/                  输出目录（按日期分文件夹）
│   └── 2026-05-19/          丽丽7 的产出在这里（待上传）
└── assets/
    └── bgm.mp3              BGM 文件
```

## 日常用法

```bash
cd F:\wkspace\fitness-video-pipeline

# 一次性：扫描素材目录，处理所有新视频
python auto_publish.py

# 定时扫描：每 60 秒扫一次
python auto_publish.py --loop 60

# 持续守护：一直跑不退出
python auto_publish.py --watch
```

## 素材目录

```
C:\Users\18091\Desktop\短视频素材\
├── *.mp4          ← 新视频放这里，自动识别教练
├── _processed\    ← 处理完自动移到这里
├── logo/横幅       ← 品牌素材（不动）
└── 秦腔*.mp4 等   ← 宣传片素材（不动）
```

## 完整处理流程

```
新视频放入 短视频素材\
    ↓
auto_publish.py 扫描到新文件
    ↓ 等 2 秒确认文件稳定
Step 1: 完整管线（main.py process，约 25 分钟）
    pre_deblock → pose_detect → denoise → beat_flash →
    highlight → energy_bar → intro_outro → watermark →
    mascot → danmaku → intensity_burst → film_look →
    pip → bgm_beat → export (GPU h264_nvenc)
    ↓
Step 2: Hook 叠加（前 4 秒"Day N 暴汗打卡 | 跟XX一起练"）
    ↓
Step 3: 缩略图（大号文字叠加）
    ↓
Step 4: Shorts（竖屏 9:16 中心裁剪 15s）
    ↓
Step 5: YouTube 上传（主视频 + Shorts, private, 频道=fitness）
    ↓
源文件移到 _processed\，状态写入 auto_publish_state.json
```

## 教练映射（coaches.yaml / COACH_MAP）

| 文件名含 | 昵称 |
|------|------|
| 艳青 | 胭脂虎 |
| 丽丽 | 腰女 |
| 建玲 | 三宝妈 |
| 郭海军 | 老兵不老 |
| 枫林红 | 霸道总裁 |
| 小红豆 | 红娘子 |
| 李刚 | 托塔天王 |
| 小飞侠 | 节拍战神 |

文件名包含关键字即自动识别，日计数器自动递增。

## 关键配置

### 日计数器 (day_counter.json)
```json
{"艳青": 2, "枫林红": 5, "丽丽": 2, "郭海军": 3}
```
每个教练下一个视频的 Day 序号。上传成功后自动 +1。

### 已处理记录 (auto_publish_state.json)
标记哪些文件已处理过，避免重复。新会话第一次跑前，确认现有素材都已在 processed 列表中，否则会全部重新处理。

## 已修复的问题

### OpenCV OpenH264 崩溃
- **现象**: `Failed to load OpenH264 library: openh264-1.8.0-win64.dll`
- **原因**: OpenCV 4.13.0 的 FFMPEG DLL 编译时带了 `--enable-libopenh264`
- **修复**: `main.py` 第 25 行加了 `os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_LIST", "MSMF")`，优先用 Windows Media Foundation 编码，不依赖 OpenH264
- **修复文件**: `main.py`、`batch_publish.py` 的顶部和子进程 env

### 大文件上传 SSL 断连
- **现象**: `SSLEOFError(8, 'EOF occurred in violation of protocol')`
- **原因**: 代理 `127.0.0.1:7897`（Clash）在大文件传输时掐连接
- **临时方案**: 用 `curl`（Schannel SSL）绕过 Python OpenSSL 上传；或等代理稳定
- **长期**: 在 `youtube_upload.py` 用 `resumable=True` 分块上传（需修复 googleapiclient 兼容性）

## 上传/YouTube 配置

- **上传模块**: `F:\wkspace\ComfyUI\custom_nodes\youtube_upload.py`
- **频道 token**: `youtube_token_yanzhi.pickle`
- **OAuth client**: `client_secret.json`
- **隐私**: 默认 private，需手动改公开
- **ComfyUI Python**: `F:\wkspace\ComfyUI\venv\Scripts\python.exe`（管线子进程用这个）

## 当前待处理

| 视频 | 状态 |
|------|------|
| 丽丽7.mp4 | 管线已处理完毕，输出在 `output\2026-05-19\`，等 YouTube 配额恢复后上传 |

丽丽7 的输出文件：
- `丽丽7_deblocked_final_16x9_hook.mp4`（带 hook 的最终版）
- `丽丽7_sm.mp4`（压缩版 20MB）
- `丽丽7_deblocked_final_16x9_hook_shorts.mp4`
- `丽丽7_deblocked_final_16x9_hook_thumb.jpg`

丽丽7 已从 state 的 processed 中移除，day_counter 丽丽回退到 2，下次跑 auto_publish.py 会重新处理+上传。

## 不要做的事

- 不要删 `output/` 下的中间文件（管线用增量恢复，删了会重新跑）
- 不要改 `stages/` 下的模块（除非明确要修改特效）
- 不要在管线运行时手动杀进程（可能留半成品文件）
- 不要同时跑两个 auto_publish.py（日计数器会冲突）

## 需要大模型介入的情况

- 新增教练（改 COACH_MAP 和 state 文件）
- 修改标题/描述模板
- 调整管线阶段开关（改 config.yaml）
- 新增特效阶段
- 修复管线 bug

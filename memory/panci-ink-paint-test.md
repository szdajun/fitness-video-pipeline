---
name: panci-ink-paint-failed-and-abandoned
description: 【2026-07-12 废弃】小红豆 4 句判词 AI 插画路线全失败 — 古风水墨风格冲突/多肢/SDXL下不到, 用户拍板废弃
metadata:
  type: project
---

# 判词 AI 插画路线 — 2026-07-12 废弃

## 状态
**已废弃 (用户拍板)**. AI 插画路线不适合本项目 (主管线真人 vs AI 风格冲突 + SD1.5 多肢 + SDXL 国内下不到). 主管线零影响零回归, 180 tests 全绿.

## 时间线 (钉死)

| 阶段 | 用户原话 | 行动 | 结果 |
|------|----------|------|------|
| 起 | "水墨画我看了, 感觉和健身视频太不协调了, 而且最后一张眼睛有点问题" | 看 4 张图 | 确认: 4 张风格撕裂 + 黑底 SD1.5 渲弱 |
| 1 | "试 2 换视觉风格" | 用户拍板换风格 | 走线条插画路线 |
| 2 | "手绘运动插画" | 选子风格 | 干净线条 + 高饱和能量色 (子风) |
| 3 | "只上 1 个总插画 + 判词 4 句字幕" | 选长度 | 单图路线 |
| 4 | "下底模 (Recommended)" | 选底模来源 | Anything v5 gated → 改 Anything v3 FP16 |
| 5 | "Anything v3 FP16 (2.13GB) (Recommended)" | 下 1.987GB | AdamOswald1/Anything-Preservation 公开镜像 |
| 6 | "挺可爱的, 能否将颜色改变为黑白色调" | 改 prompt 加 monochrome | 出黑白版 |
| 7 | "腿 3 条, 胳膊一条, 吓人" | SD1.5 多肢 bug | 改站姿 → 还是多肢 → 改半身图稳定 |
| 8 | "一个腿抬起来了, 支撑腿只看到一半" | 9:16 全身被裁 | 加 far shot + cropped negative → 还是被裁 |
| 9 | "三条腿了" | SD1.5 老问题反复 | 改半身图 (胸口以上) → 终于稳 |
| 10 | "这个只有上半身, 是这样要求的吗" | 用户问姿势 | 解释半身图是 SD1.5 bug 折衷方案 |
| 11 | "重下 SDXL/换底模" | 用户拍板换 SDXL | HF 全 401 + CivitAI 401 (要 token) + 国内镜像站跳回 HF = 全死 |
| 12 | "废弃" | 拍板废弃插画路线 | 删产物/脚本/底模 |
| 13 | "ComfyUI 留着有用的" | 拍板保留 ComfyUI | 主体 + ckpts + custom_nodes 全留, 只清插画相关产物 |

## 教训 (钉死, 不要再走)

### 1. AI 风格插画 vs 主管线真人视频 = 视觉语言冲突
- 古风水墨 (静态/写意/古装) vs 现代健身 (动态/现代/广场) = 撕裂
- 二次元萌系 vs 真人跟练视频 = 不协调
- **不要再走 AI 插画路线**. 主管线已经是真人视频 (face_swap), 加插画不如加汉印/水印/能量条这些"已融入"的元素

### 2. SD1.5 9:16 全身 + 复杂姿态 = 多肢 bug 老问题
- SD1.5 Anything v3 在 (踢腿+展臂+抱头) 任何组合都会出 3-4 条腿/胳膊
- 修法 1: 改半身图 (胸口以上) — 稳, 但人物不完整
- 修法 2: 改静态站姿 (双脚并拢+单臂上举) — 50% 概率稳
- 修法 3: 加 negative `extra limbs, three legs, four arms` — 减弱不根治
- 修法 4: 换 SDXL — 国内下不到 (HF/CivitAI 全 401)
- **未来如果再做**: 用 SDXL 在线 (CivitAI online) 或 Comfy cloud, 别本地

### 3. HF 限流 + CivitAI token 双重死结
- HF 主仓: `andite/anything-v5.0`, `Linaqruf/animagine-xl-v3.1`, `ProGamerGov/AnyLoRA`, `Yuno779/AnythingXL` 全部 401 (限流 / gated)
- hf-mirror.com 镜像站: 全 308 跳回 HF 本体 = 没用
- CivitAI `/api/download/models/{id}` 直接: 401 (需 token)
- CivitAI 在线: 可达, 可浏览, 但下载需登录 token
- **唯一出路**: 用户手动从某渠道下好后给本地路径

### 4. Anything v3 FP16 = 2.0GB 实际大小 (不是 2.13GB)
- 标称 2.13GB 是 v3 full, FP16 pruned 实际 1.987GB
- 功能无影响, 注意 ckpt 文件大小描述

## 当前 ComfyUI 状态 (用户拍板"留有用的")

### 留的
- `F:\wkspace\ComfyUI\` 主体完整
- `models/checkpoints/DreamShaper_8_pruned.safetensors` 2.0GB — 主管线长期依赖
- `models/checkpoints/ltx-video-2b-v0.9.5.safetensors` 6.0GB — 未来视频生成
- `models/configs/anything_v3.yaml` — 占用极小, 留着 (孤儿)
- `custom_nodes/ComfyUI-LTXVideo` + `AnimateDiff-Evolved` + `Impact-Pack/Subpack` + `VideoHelperSuite` + `Manager` — 主管线未来用
- `custom_nodes/ComfyUI-CogVideoXWrapper` — IMPORT FAILED 是没装依赖不是坏, 完整仓库, 留着
- `custom_nodes/daily.py` + `content_gen.py` + `hosts.py` + `heygem_*` + `face_restore.py` + `download_models.py` + `post_process.py` + `batch_producer.py` + `client_secret.json` + `published_scripts.json` + `example_node.py.example` — ComfyUI 自带, 跟本项目无关, 全留

### 清掉的 (本轮)
- `tools/panci_paint/xiaohongdou/` (4 张水墨图原图) — 旧 commit 3884f00 留存, 工作目录删
- `tools/panci_paint/xiaohongdou_v2/main.png` (本轮半身图最终版) — 删
- `scripts/gen_panci_paint_v2.py` (本轮出图脚本) — 删
- `scripts/_parse_csdn.py` (本轮调研用, 后来没用上) — 留 (untracked)
- `ComfyUI/models/checkpoints/anything-v3-fp16-pruned.safetensors` 1.987GB — 删 (用户拍板)
- `ComfyUI/custom_nodes/pipeline.py` (1KB) — 删 (本轮之前的孤儿, IMPORT FAILED)
- `ComfyUI/custom_nodes/auto_daily.py` — 删 (同上)
- `_temp/wawa_probe/` — 删 (本轮调研临时帧)
- `_temp/civitai_search.json` — 删 (本轮搜索结果)
- `output/2026-07-12/_test_xhd_panci.mp4` — 删 (本轮 8s 测试视频)

### 留的 (跟本轮无关)
- `output/2026-07-12/小红豆1_2_merged_*` 三件套 — 上轮跑通的产品, 抖音待传
- `gen_panci_paint.py` (旧 4 张水墨图脚本) — 在 git 历史 commit 3884f00
- ComfyUI server 已停

## 守门 (TODO)
- 无. 主管线零影响, 没改任何代码

## 重启信号 (不要主动重启)
1. 用户拍板 OR
2. 主管线改用真动态视频路线 (LTX-Video) 实战 (现在 LTX-Video 还在) OR
3. AI 插画路线有明确收益目标 (例如小红豆卡通立绘 IP)

不主动重启.

## 关联
- 主管线 0 代码改动
- 180 tests 全绿零回归
- commit 3884f00 @ feat(panci_paint): 小红豆 4 句判词水墨插画生成 + 8s 测试视频 — git 历史留存 (不在工作目录)
---
name: panci-ink-paint-test
description: 【2026-07-12】判词水墨插画测试 — DreamShaper_8 SD1.5 出图 + 防暴露修复 + 测试视频合成
metadata:
  type: project
---

# 判词水墨插画 — 2026-07-12 测试产物

## 状态
**仅测试阶段, 未接入主管线**. 用户拍板"先测试" → 走轻量路径, 4 张静态图 + 8s 测试视频.

## 产物
- 4 张原图: `tools/panci_paint/xiaohongdou/{1,2,3,4}.png` (512×896 SD1.5 出图, ~30s/张)
- 测试视频: `output/2026-07-12/_test_xhd_panci.mp4` (8s, 4×2s Ken Burns 慢推)
- 生成脚本: `scripts/gen_panci_paint.py` (走 ComfyUI HTTP API)
- 测试视频合成: `scripts/test_panci_overlay.py`

## 技术路径
1. 启动 ComfyUI server (cd ComfyUI && python main.py --listen 0.0.0.0 --port 8188)
2. workflow = `ComfyUI/custom_nodes/workflows/text_to_image.json` (用 DreamShaper_8 SD1.5 checkpoint)
3. 4 prompt 各 1 张, 512×896 竖版, 25 steps, euler_ancestral + karras, seed 固定
4. ffmpeg zoompan 把单图变 2s 视频 (1.0x→1.08x 缓推) → concat 4 段 = 8s

## 关键坑

### 防暴露 (CLAUDE.md 弹幕/字幕内容审查基线)
**首跑**: 图 2 (婚纱感) + 图 4 (高开叉露大腿) 触发"性暗示"基线 ⚠️.
**修复**:
- 加 negative: `nudity, lingerie, bikini, cleavage, bare shoulders, exposed chest, wedding dress, white dress, slit, high slit, thigh gap, navel, sexy, sensual, suggestive, revealing outfit, see-through, short skirt, mini skirt, bare legs, bare thighs, high boots, knee high slit, side slit, leg visible, midriff, action pose, dynamic pose, dancing wildly`
- prompt 里 "dancing wildly" / "flowing" / "dynamic" 等暗示词改成 "gentle" / "graceful" / "modest"
- 重跑后 4 张全部合规 ✅

### diffusers single_file 不兼容 SD1.5
- ComfyUI venv (torch 2.12+cu130, diffusers 0.32+) 调 `StableDiffusionPipeline.from_single_file()` 报 `CLIPTextModel has no attribute text_model` (新版 transformers 改 CLIPTextModel 内部结构)
- 解: 走 **ComfyUI HTTP server + workflow API**, 不直接调 diffusers
- ComfyUI 自己加载 checkpoint + KSampler 没问题

## 视觉评估
- 图 1 红豆生来俏模样: 红衣女子 + 红梅枝 + 白墙 ✅ 含蓄水墨工笔
- 图 2 香汗淋漓透红妆: 红衣舞动 + 飘袖 ✅ 红梅 + 烟水意境
- 图 3 娇喘微微惹人怜: 红衣侧脸 + 流苏 + 牡丹窗光 ✅ 最佳, 真有工笔仕女图感
- 图 4 花枝乱颤舞霓裳: 黑红裙 + 飘带 + 优雅动作 ✅

## 下一步候选 (待用户拍板)
1. 接主管线 (新 stage `45_panci_paint.py`, opening 期间 4×2s 切图)
2. 试 SDXL 出图 (更大模型, 风格更稳, 显存需 ≥10GB, RTX 4070 12GB 可行)
3. 走"真动态"路径 (Pika/Runway 图生视频, 每段 4s, 投入大)
4. 给其他教练做 (蜂王/李娜/铁娘子 4 句判词各异)

## 已知限制
- DreamShaper_8 是 SD1.5 衍生, 风格倾向"性感动态", 即便 negative 控住仍偶尔出黑丝/短裙
- 1080×1920 不在 SD1.5 原生分辨率, 当前 512×896 上采样后单图细节略糊
- 没有 SDXL base checkpoint (本机只有 DreamShaper 8 = SD1.5 衍生, 2GB)

## 守门 (TODO)
- 没加. 主线零影响 (无主管线接入), 仅作素材工具
- 如接入主管线需加: 4 张图存在性守门 + 内容审查 negative 不变 + duration=8s

## 关联
- 主管线 0 代码改动
- 未来接 `stages/45_panci_paint.py` 时, 复用 `gen_panci_paint.py` + coach_profiles.PANCI
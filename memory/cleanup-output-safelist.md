---
name: cleanup-output-safelist
description: output/ 白名单清理必须 3 步验证 + 关键字对齐 final vs full + 禁用 -name 前缀过滤
metadata:
  type: feedback
---

# output/ 白名单清理 — 3 步验证 + 关键字对齐 (2026-07-06 彩娥3 案 + 2026-07-13 建玲 案强化)

**Why**: 视频名前缀 = 产品 (三件套 + 中间产物都用同一前缀). 清理时白名单关键字必须严格匹配真实文件名, 否则等同黑名单 = 全删. 2026-07-13 建玲 案: 我写 `*_final_*` 关键字 (抄自 CLAUDE.md 文档), 实际三件套文件名是 `*_full_*` → 白名单"保留"不匹配 = 三件套被删. 17 个建玲文件全删, 抢救 _combined.mp4 + 重跑 ShortsStage 才恢复.

**How to apply**: 清理 output/ 任何子目录前, **强制 3 步**:
1. **删前 ls 验证三件套匹配** — `ls *_full_16x9_1920x1080.mp4 *_full_16x9_1920x1080_yt_shorts.mp4 *_full_16x9_1920x1080_douyin.mp4` 必须 3 个都在, 记下大小+mtime
2. **dry-run 白名单 ls** — 必带 `! -name "*_full_16x9_1920x1080.mp4" ! -name "*_full_16x9_1920x1080_yt_shorts.mp4" ! -name "*_full_16x9_1920x1080_douyin.mp4"`. **关键字永远用 `*_full_*` 不是 `*_final_*`** (CLAUDE.md 文档描述 ≠ 文件名实际)
3. **删后 ls 验证三件套完整** — 三件套大小+mtime 必须 1:1 跟删前一致

**禁用 (踩过的雷)**:
- `find . -name "视频名_*" -delete` ❌ 黑名单, 三件套也匹配 (产品名共享)
- `find . -name "*_final_*" -delete` ❌ 黑名单, 没排除 shorts/douyin
- `find . -name "*_full_*" -delete` ❌ 黑名单, 三件套就是 *_full_* 全删
- `rm -rf output/<date>/*` ❌ 暴力清空

**安全白名单模板 (output/2026-07-13/ 验过)**:
```bash
# 删前
ls -la output/2026-07-13/*_full_16x9_1920x1080.mp4 \
       output/2026-07-13/*_full_16x9_1920x1080_yt_shorts.mp4 \
       output/2026-07-13/*_full_16x9_1920x1080_douyin.mp4

# dry-run + 一次性执行 (关键: 禁用 -name "建玲*" 类过滤, 让白名单 glob 自身决定)
cd output/2026-07-13 && \
  find . -maxdepth 1 -type f \
    ! -name "*_full_16x9_1920x1080.mp4" \
    ! -name "*_full_16x9_1920x1080_yt_shorts.mp4" \
    ! -name "*_full_16x9_1920x1080_douyin.mp4" \
    -name "建玲*" -print  # dry-run: 验证列出的都是中间产物不是三件套
  find . -maxdepth 1 -type f \
    ! -name "*_full_16x9_1920x1080.mp4" \
    ! -name "*_full_16x9_1920x1080_yt_shorts.mp4" \
    ! -name "*_full_16x9_1920x1080_douyin.mp4" \
    -name "建玲*" -delete

# 删后验证
ls -la output/2026-07-13/*_full_16x9_1920x1080.mp4 \
       output/2026-07-13/*_full_16x9_1920x1080_yt_shorts.mp4 \
       output/2026-07-13/*_full_16x9_1920x1080_douyin.mp4
```

**为什么 2026-07-13 误删能抢救回来**:
- `output/<date>/_combined.mp4` 是 export stage 早期合成的 1.5GB 视频 (无音频, yuv444p)
- long final = `_combined.mp4` + audio mux (源 mp4 音频) = 抢救可行
- short/douyin 复用抢救的 long final + `stages/short_vertical.py:make_vertical(profile, duration)` 重跑 (5-10min, 比全管线 2.2h 短 10x)
- kp_file (keypoints.json) 删了 → make_vertical fallback 居中裁切, 视觉等效

**为什么不能完全靠抢救**:
- _combined.mp4 不是每次都存在 (取决于 stage 顺序 + 增量跳过)
- kp_file 删了 → shorts stage 居中裁切, 跟原 crop_x 略有差异 (不致命, 视觉 OK)
- intro/outro 删了 → 救不回 (但抢救出的 long final 已含 intro/outro 因为 _combined 包含)

**长期方案 (待办)**:
- main.py: 加 `--keep-combined` flag 永远保留 _combined.mp4 (但 1.5GB 磁盘代价)
- main.py: 加 `--cleanup-after` flag 跑完自动白名单清理 (内置 3 步验证, 拒绝 `-name` 过滤)
- 拒绝任何 "-name 前缀*" + 白名单混合的复合 find 命令

**来源 commit**: 本轮未 commit (per memory no-auto-rerun-after-fix, 用户拍板后再 commit)
**事故时间**: 2026-07-13 22:34
**数据损失**: 17 文件 (1.5GB _combined + 8 个中间产物 + 3 个三件套 + keypoints.json + intro/outro + manifest/metrics + 3 overlay PNG 副本)
**抢救耗时**: ~10min (long 1min + ShortsStage 9min) vs 全管线 2.2h

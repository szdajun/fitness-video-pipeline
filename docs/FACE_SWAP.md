# Face Swap 流水线经验总结

> 2026-06-26 反复折腾后定稿。**所有规则都已被 31 个 pre-commit 测试守门**，下次不要重新折腾。

## 性能数据

| 版本 | 102s 视频 face_swap 耗时 | 总耗时 | 备注 |
|---|---|---|---|
| v0 (2026-06-25 郭海军1) | **7.3 小时** | 8.5h | GFPGAN CPU 跑（7.27h 占 85%）|
| v1 (2026-06-26 关 GFPGAN) | 21 min | 40 min | 14× 提速 |
| v2 (cx 中心选脸) | 21 min | 40 min | 视觉更稳定 |
| v3 (+ color_match) | 21 min | 1h12min | 色彩融入场景（color_match 仅 +15s，多的是冷启动）|
| v4 (NVENC) | 21 min | **~30 min** | export 275s → 142s |

**关键发现**：face_swap 7.3h 不是 inswapper 慢（它只占 21min），而是 **GFPGAN 跑 CPU 修"假脸"**。

## 6 个坑 + 解决

### 坑 1: GFPGAN 跑 CPU 修"换上去的假脸"

**症状**：face_swap 跑 7.3h。

**根因**：GFPGAN 默认在 CPU 跑（`torch.cuda.is_available() == False` 在子进程里），3070 帧 × 200ms = 10min CPU 计算，加上模型启动 = 7h。

**更深的问题**：GFPGAN 修的是 inswapper 128×128 输出（**已经是假脸**），没有意义。

**修复**：`stages/37_face_swap.py: gfpgan_strength = 0`（默认 0）。可手动 `--gfpgan-strength 0.5` 启用，但默认关。

### 坑 2: min_face_area=0.02 过滤远景领操人

**症状**：face_swap 跑了但 0 张脸被换。

**根因**：健身视频全身镜头领操人脸只占 0.5-1%，0.02 阈值过滤掉所有人。

**修复**：`tools/face_swap.py: swap_face()` 默认 `min_face_area=0.001`。

### 坑 3: insightface 漏检仰头/侧脸

**症状**：ROI 内 `app.get()` 找不到郭海军的脸，只检测到远处小脸。

**根因**：buffalo_l 检测器对仰头/逆光的脸不敏感。

**缓解**（不完美）：
- bbox 用 pose 关键点（鼻子+肩膀），保证框在领操人身上
- ROI 调大到 ≥160px 让脸完整在框内
- 用 cx 接近 ROI 中心的策略而非 max area

**未解决**：领操人远景 + 仰头 + 多人群场景，算法无法保证换脸成功。

### 坑 4: ROI 内"最大脸"不稳定

**症状**：换脸一会儿换到领操人，一会儿换到别人脸。

**根因**：`max(candidates, key=area)` 选了"最大脸"，但远处大婶面积可能 > 领操人。

**修复**：`tools/face_swap.py: swap_face()` 用 `min(dist_to_roi_center, -area, -det_score)` 选 cx 最近的。

### 坑 5: 换脸后色彩融入场景差

**症状**：换上去的脸**发白发亮**，跟场景光照不匹配。

**根因**：inswapper 128×128 输出保留源脸光照，GFPGAN 重建源脸是均匀光照。

**修复**：color_match_face (LAB Reinhard) 把换后脸 L/A/B 通道均值方差拉向原场景。`swap_face` lead_bbox 分支也调（之前只全图检测分支有）。

### 坑 6: ffmpeg CPU 编码慢

**症状**：export stage 275s (102s 视频)。

**根因**：libx264 CPU 编码。

**修复**：`config.yaml: output.encoder: nvenc` + `prefer_gpu: true`。h264_nvenc 加速 48%。

## 关键架构决策

### 为什么用 pose 而不是全图检测？

| 策略 | 优点 | 缺点 |
|---|---|---|
| 全图 `app.get(frame)` | 简单 | 漏检领操人 |
| pose 定位 + ROI 检测 | 框定领操人 | pose keypoints 不准（仰头时鼻子 y 偏高）|
| MediaPipe FaceMesh | 小 ROI 鲁棒 | 不能直接给 inswapper（输出 landmark 不是 bbox）|

**当前选择**：pose 定位 + ROI 检测，配置见 `get_lead_bbox_from_pose()`。

### 为什么不一直开 GFPGAN 美颜？

源脸**已经过 GFPGAN 重建**（存在 `tools/{coach}_face_gfpgan.png`），inswapper 输出的 128×128 脸本身清晰。换脸后再过 GFPGAN = 修假脸，没必要。

### 为什么不用 cx 最大的脸做 lead？

cx 最大的人 ≠ 领操人。cx 最接近 0.5 + body 最大 = 折衷。

## 配置速查

### config.yaml (output 段)
```yaml
output:
  encoder: nvenc          # 不用 auto, 显式指定
  prefer_gpu: true
  crf: 14                 # 质量 (越小越清晰, 默认 14 已够)
```

### face_swap 块（config 或 preset）
```yaml
face_swap:
  gfpgan_strength: 0      # 必须 0, 不要开
  min_face_area: 0.001    # 必须 ≤ 0.001
  color_match_strength: 0.8  # 默认 0.8, 不要关
```

## 命令参考

```bash
# 单视频
python main.py process "input.mp4" --preset youtube --full-video --coach "郭海军"

# 批量
python main.py batch -i "source_dir" -o "output_dir" --preset youtube --segment 60

# 单独参数覆盖
python main.py process "input.mp4" --preset youtube --gfpgan-strength 0 --no-color-grade
```

## 调试技巧

**问题**：换脸不生效，看不出来。
1. 截帧对比 source vs output（同时间点）：
   ```bash
   ffmpeg -ss 30 -i source.mp4 -frames:v 1 before.jpg
   ffmpeg -ss 30 -i output.mp4 -frames:v 1 after.jpg
   ```
2. 看 `_keypoints.json`：用 `get_lead_bbox_from_pose()` 验证 bbox 是否覆盖领操人。
3. 看 ROI 内 `app.get(roi)` 返回的脸数 + det_score（低 < 0.6 会失败）。

**问题**：换脸到错的脸上。
- 检查 pose 检测是否选错 lead（cx 居中但身体小 → 远处人）
- 检查 ROI 大小：160-250px 是甜蜜区

**问题**：色彩不融入。
- color_match_strength 调到 1.0 试试
- 检查 `orig_face_roi` 是否在 swap_face lead_bbox 分支有备份

## 后续改进（未做）

1. **SCRFD 检测器** — 替代 buffalo_l，更鲁棒
2. **NVENC 编码中间 stage** — 让 color_grade/face_swap 等都用 GPU
3. **PNG 序列 → 直接 frame 传递** — 减少磁盘 I/O
4. **face_swap pipeline cache** — 第二次跑用 cache（已部分实现，但每次跑还是冷启动）

## 测试清单

```
tests/
├── test_upload_title.py             # 标题模板 (13 tests)
├── test_upload_manifest.py          # manifest 写入 (4 tests)
├── test_upload_manifest_dedup.py    # manifest 不重复 (3 tests)
├── test_face_swap_no_gfpgan.py      # gfpgan_strength=0 (3 tests)
├── test_face_swap_min_face_area.py  # min_face_area ≤ 0.001 (3 tests)
└── test_face_swap_lead_selection.py # cx 中心选脸 (2 tests)
```

**31 tests, all green.** Run: `.git/hooks/pre-commit`
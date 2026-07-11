---
name: shorts-hook-disabled-2026-07-12
description: 【2026-07-12 用户拍板取消】竖屏 hook 高燃预览开场 (抖音+Shorts 都不加, opt-in 仍可用). 用户原话"竖屏的产品, 最前面的 hook, 感觉很乱, 不如取消了" + "抖音和 youtube 都关"
metadata:
  type: feedback
---

# 竖屏 hook 高燃预览开场 — 已取消 (2026-07-12)

## 状态
**抖音 + YouTube Shorts 默认都不再加 hook**. 算法 + 字幕 + 4 步编码链路完整可用, `--with-hook` CLI flag 保留 opt-in (单文件/单次任务临时启用).

## 用户原话
1. "竖屏的产品, 最前面的 hook, 感觉很乱, 不如取消了"
2. "抖音和 youtube 都关"

## 改动清单 (全部落 commit, 187 tests 全绿)

### 代码层
- `main.py:143-148`: CLI help 文案标"【已废弃 2026-07-12】", 默认行为不变 (用户没传 = 不开)
- `stages/39_shorts.py:118`: `cfg.get("shorts_hook", True)` → `False` (默认改关)
- `stages/short_vertical.py:739`: `hook_enabled: bool = False` (已经是 False 默认, 不变)
- `stages/short_vertical.py:900-911`: make_vertical 内部 gate `if hook_enabled and profile in ("yt_shorts", "douyin")` (不变, hook_enabled=False 自然跳过)

### 配置层
- `presets/vertical_native.yaml`: `shorts_hook: true` → `false`
- `presets/fengwang.yaml`: `shorts_hook: true` → `false`
- `shorts_hook_dur: 4` 两个 preset 都保留 (opt-in 还能用)

### 测试层 (守门防止回滚)
- `tests/test_short_vertical_hook.py` 新增 `TestHookDisabledByDefault` 4 个守门:
  1. `test_39_shorts_default_hook_enabled_is_false`: 扫 `stages/39_shorts.py` 必须有 `cfg.get("shorts_hook", False)`
  2. `test_vertical_native_yaml_hook_false`: 扫 yaml 必须 `shorts_hook: false`
  3. `test_fengwang_yaml_hook_false`: 同上 fengwang
  4. `test_main_py_with_hook_help_mentions_deprecated`: 扫 main.py 必须有"已废弃 2026-07-12"文案
- 原 10 个算法测试 (compute_hook_window 纯算法层) 不动, 仍可用

### 文档层
- `CLAUDE.md` 第 10 段标题加 **(2026-07-12 用户拍板取消默认开)** + 顶部状态说明
- `CLAUDE.md` 第 315 行 CLI flag 文案加粗提醒
- `CLAUDE.md` 第 384 行 CLI 默认值说明改
- `CLAUDE.md` 第 436 行 vertical_native preset 元素清单 + hook 行改 ❌ 砍掉
- `CLAUDE.md` 第 247 行 + `HANDOFF.md` (本轮新加的段) + memory 本文件 全部同步

## 不动的部分

- **算法 `compute_hook_window`**: 纯算法, 没改
- **字幕渲染 `render_short_overlay.render_preview`**: 没改, 未来 `--with-hook` 还能用
- **4 步编码链路 (step0/step1/step1.5/step2)**: 没改
- **音频 `anullsrc`+concat 修复**: 没改, 仍有效
- **🔥 emoji 字体 (Segoe UI Emoji)**: 没改
- **历史已发布视频 (蜂王1/李娜1/铁娘子1+2/小飞侠 等已上传 YT 的)**: 冻结不回改 (per memory `coach-rename-frozen-published`)

## 为什么
- 用户实际看到竖屏产品最前 4s 觉得"很乱" (字幕+静音+视觉跳变)
- 完播率/开场吸睛是好的, 但视觉体验冲突 > 营销价值
- 用户拍板完全关, 不留 opt-in 默认

## 未来重启用 (不主动)
1. 用户拍板恢复 (例如反馈"看完播率数据反而掉了" 或者 想给某个教练单开)
2. 重新设计 hook 视觉 (例如换静态海报 + 教练头像 + 判词标题, 不再用"🔥 高燃预警")

## 重启信号
- 用户拍板 OR
- 未来完播率/数据反馈说明 hook 实际帮上忙了 → 重启讨论
- **不要主动重启** per memory no-auto-rerun-after-fix

## 关联
- 主管线 187 tests 全绿零回归
- 历史产出 (蜂王1/李娜1 等已发布的) 冻结
- 抖音待传的你手工判断要不要传无 hook 版 (蜂王1 douyin / 李娜1 douyin / 海军1_2 douyin / 丽丽1_2 douyin / 建玲1_2 douyin / 铁娘子1_2 douyin / 小飞侠1_2 douyin 共 7 套)

---

## 相关历史
- 上线: 2026-07-07 commit c2f3613 feat(shorts): hook 默认开 (用户拍板'功能稳定后要默认开')
- 取消: 2026-07-12 本轮 (用户拍板取消, 本 memory 落地)
---
name: upload-reminder-tool
description: 【2026-07-13 上线】Windows 定时任务 + 命令行弹窗, 黄金时段 (10-14/19-23 北京) 提醒人工上传 YouTube/抖音. YouTube+抖音全停自动 (历史教训: YT 平台挂死自动上传视频), 弹窗是当前拍板的"半自动"方案.
metadata:
  type: project
---

# Upload Reminder 工具 (2026-07-13 用户拍板上线)

## 一句话

YouTube + 抖音全停自动上传 (per [[yt-long-video-publish-immediately]] + [[douyin-manual-upload]]), 用 Windows Task Scheduler + Python CLI 弹窗在黄金时段提醒用户手工传. **半自动** = 调度自动 + 上传人工.

## 触发机制

- **黄金时段** (复用 [[shorts-golden-hour-auto-publish-2026-07-12]]): 10-14 / 19-23 北京 UTC+8 (5月数据 1376/935/862 view 高峰)
- **6 个时点**: 10:00 / 12:00 / 14:00 / 19:00 / 21:00 / 23:00 (Task Scheduler 6 个独立任务, 后缀 _10/_12/_14/_19/_21/_23)
- **手动跑**: `uv run python tools/upload_reminder.py --skip-golden-check` (调试用)

## 文件清单

| 文件 | 角色 |
|------|------|
| `tools/upload_reminder.py` | CLI 入口 + 渲染 + 交互循环 (input 指令) |
| `lib/reminder_state.py` | `records/upload_reminder_log.json` 读写 + 状态 mutate |
| `tools/install_reminder_task.bat` | 双击注册 6 时点 |
| `tools/uninstall_reminder_task.bat` | 双击卸载 |
| `records/upload_reminder_log.json` | 状态 (path → record, git 入 .gitignore 或本地不传) |
| `tests/test_reminder_state.py` | 7 tests |
| `tests/test_reminder_scan.py` | 6 tests |
| `tests/test_reminder_golden.py` | 8 tests (含 7 个 _is_golden_hour 边界) |
| `docs/superpowers/specs/2026-07-13-upload-reminder-design.md` | 设计 spec |
| `docs/superpowers/plans/2026-07-13-upload-reminder.md` | 实施 plan |

## 交互指令 (命令窗口 input)

- `1,2,3` — 标记编号 1/2/3 已传 (空格或逗号分隔, 支持 `1-3` 区间)
- `o 1` — 系统默认播放器打开视频1
- `p 1` — 复制视频1路径到剪贴板
- `s 1` — 跳过 (下次还弹, 累加 remind_count)
- `r 1` — 恢复 (把 [OLD 3+次] 重置回待提醒)
- `a` — 全部标已传
- `h` — 查看帮助
- `q` — 退出 (不标已传, 保留状态)

## 漏处理容错

- 同一视频提醒 ≥3 次未标记 → `auto_archived: true`, 不再弹
- 顶部 [OLD 3+次未处理] 黄色块提示
- 用 `r 1` 手动恢复 (重置 remind_count=0, 删 auto_archived)
- log.json 损坏 → 自动备份 `.bak` + 返空 (不崩)

## 三件套扫描规则 (跟产物命名一致)

- 匹配: `*_full_16x9_1920x1080.mp4` (long) / `*_full_16x9_1920x1080_yt_shorts.mp4` (short) / `*_full_16x9_1920x1080_douyin.mp4` (douyin)
- 跳过: `*_full_16x9_1920x1080_full_16x9.mp4` 副本 (per CLAUDE §"上传只传 final_16x9")
- 跳过: 所有中间产物 (`*_color.mp4` / `*_watermark.mp4` / `*_faceswap*.mp4` / `*_energybar*.mp4` / `*_intro.mp4` / `*_outro.mp4` / `*_keypoints.json` 等)

## 关键设计选择 (钉死)

- ❌ **不读** `upload_manifest.json` 判 YouTube 已传 (用户拍板"全停自动", manifest 不可靠)
- ❌ **不自动**调 `tools/upload_youtube.py` 上传 (per 拍板)
- ❌ **不删**已传视频 (用户自己管空间)
- ❌ **不自动关窗** (等用户点 [X])
- ✅ 复用 `lib/upload_utils.py` 的 `_is_golden_hour` / `seconds_until_next_golden` / `build_title` (零侵入)
- ✅ Win10/11 ANSI 颜色 (ctypes 启用 VT100), 退化无色不报错
- ✅ 跨平台 open/clipboard (Win startfile/clip, Mac open/pbcopy, Linux xdg-open/xclip)

## 用户安装

双击 `tools/install_reminder_task.bat` (管理员权限) → 看到 6 个 [OK] 即装上. 卸载双击 `uninstall_reminder_task.bat`.

## 跟主管线关系

- **零侵入**: 不改 `stages/`, `main.py`, `pipeline/`, `lib/upload_utils.py`, `config.yaml`
- 新代码仅在 `tools/`, `lib/`, `tests/`, `memory/`, `CLAUDE.md`, `docs/superpowers/`
- 7 + 6 + 8 = 21 new tests, 总 244 + 21 = 265 (实际 244 + 7 + 6 + 8 = 265)

## 相关 memory

- [[yt-long-video-publish-immediately]] — YT 长视频立即发布 (用户拍板)
- [[douyin-manual-upload]] — 抖音手工传 (用户拍板)
- [[shorts-golden-hour-auto-publish-2026-07-12]] — 黄金时段算法
- [[upload-manifest-required]] — manifest 写规则 (本工具不依赖 manifest, 但风格一致)
- [[pre-commit-hook-venv-pipefail]] — 守门 hook (每次 commit 跑相关 tests)

---
name: upload-reminder-design
description: 设计 — Windows 定时任务 + 命令行弹窗，提醒用户人工上传 YouTube/抖音视频 (2026-07-13)
---

# Upload Reminder 工具设计 (2026-07-13)

## 一句话

**YouTube + 抖音全靠用户手工上传** (per `yt-long-video-publish-immediately` + `douyin-manual-upload`)，需要一个 Windows 定时任务在黄金时段自动弹出命令行窗口列出待传视频，用户输入编号标记已传。复用 `lib/upload_utils.py` 的黄金时段算法 (`_is_golden_hour` / `seconds_until_next_golden`)。

## 背景 (2026-07-13 用户拍板的事实)

- **自动上传全停**: YouTube long + Shorts + 抖音 全部手工传 (历史教训: 自动发被平台挂死)
- **黄金时段是拍板的**: 10-14 / 19-23 北京时间, 每天 5 月数据 1376/935/862 view 高峰 (per `shorts-golden-hour-auto-publish-2026-07-12`)
- **弹窗密度**: 每天 6 个时点进场提醒 (10:00 / 12:00 / 14:00 / 19:00 / 21:00 / 23:00 UTC+8)
- **容错**: 用户人不在错过 → 自动跟到下一个黄金时段, 不丢
- **范围**: 只提醒 `output/<date>/*.{final_16x9_1920x1080.mp4, yt_shorts.mp4, douyin.mp4}` 三件套
- **已传标记**: 命令行无 GUI checkbox, 用"输入编号"实现 (e.g. `1,2` 标记 1 和 2 已传)
- **快捷打开**: 输 `o 1` 用系统播放器打开视频1
- **漏处理**: 同一视频提醒 ≥3 次未标记 → 自动从待提醒列表移除 (可手动 `r 1` 恢复)

## 架构

4 个文件:

| 文件 | 角色 | 依赖 |
|------|------|------|
| `tools/upload_reminder.py` | CLI 入口, 渲染 + 交互 | lib/upload_utils, lib/reminder_state |
| `lib/reminder_state.py` | log.json 读写 + 状态管理 | 标准库 |
| `tools/install_reminder_task.bat` | 一键装 Windows Task Scheduler 6 时点 | schtasks.exe |
| `tools/uninstall_reminder_task.bat` | 一键卸载 | schtasks.exe |

**不修改**主管线代码, 不修改 `lib/upload_utils.py` 主体, 只 import 其黄金时段函数 (零侵入)。

## 数据流

```
[Task Scheduler 6 时点触发]
        ↓
[tools/upload_reminder.py]
        ↓
1. lib.upload_utils._is_golden_hour() 检查当前是否黄金期
   ├─ 不在 → 提示 + 询问是否继续 (默认 y)
   └─ 在 → 继续
2. lib.reminder_state.load() 读 log.json
3. 扫描 output/<date>/*.{final_16x9, yt_shorts, douyin}.mp4
4. 过滤: log.marked_uploaded_at 存在 → 跳过
5. 过滤: log.remind_count >= 3 → 跳过 (顶部加 [OLD 3+次] 黄字)
6. 渲染: 每条 编号 | 教练 | 类型 | 文件路径 | 大小 | 标题
7. input() 循环读指令:
   - `1,2,3` → 标记这些编号"已传"
   - `o 1` → os.startfile(视频1) 系统默认播放器
   - `p 1` → 复制路径到剪贴板 (pyperclip)
   - `s 1` → 跳过 (下次还弹, 不计数)
   - `r 1` → 恢复 (把 remind_count 重置 0)
   - `a` → 全部标已传
   - `q` → 退出 (不标记, 不关窗, 等你 X 关闭)
8. lib.reminder_state.save() 写回 log.json
9. 不自动关窗, 等你 X
```

## 状态文件结构

`records/upload_reminder_log.json`:

```json
{
  "F:\\wkspace\\fitness-video-pipeline\\output\\2026-07-12\\艳青1_2_merged_full_16x9_1920x1080.mp4": {
    "marked_uploaded_at": "2026-07-13T10:23:15",
    "remind_count": 0,
    "last_reminded_at": "2026-07-13T14:00:00"
  },
  "F:\\wkspace\\fitness-video-pipeline\\output\\2026-07-12\\艳青1_2_merged_full_16x9_1920x1080_douyin.mp4": {
    "marked_uploaded_at": null,
    "remind_count": 3,
    "last_reminded_at": "2026-07-13T14:00:00",
    "auto_archived": true
  }
}
```

**字段说明**:
- `marked_uploaded_at`: null = 未传; ISO timestamp = 已传
- `remind_count`: 累积提醒次数, ≥3 自动 `auto_archived: true` (不再弹)
- `last_reminded_at`: 最近一次弹窗时间

**路径用绝对路径 + 反斜杠转义** (`\\`), 与 manifest 风格一致 (per `upload-manifest-required`)。

## 渲染 (ANSI 彩色)

```
[胭脂虎健身团] 上传提醒 - 2026-07-13 14:23 北京
==================================================
当前黄金时段: ✓ 14:00-14:59 (均 view 1376, 5月数据)
今日剩黄金时段: 2 次 (19:00 / 21:00)
==================================================

待上传视频 (3 条, 已自动过滤已传 + 归档):

[1] LONG   艳青    274MB  2026-07-12
    路径: F:\...\艳青1_2_merged_full_16x9_1920x1080.mp4
    标题: 【胭脂虎】艳青力量燃脂操 | 塑腰臀跟练 | 细柳营健身

[2] SHORT  艳青    68MB   2026-07-12
    路径: F:\...\艳青1_2_merged_full_16x9_1920x1080_yt_shorts.mp4
    标题: 30秒暴汗燃脂 | 胭脂虎艳青 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡

[3] DOUYIN 艳青    280MB  2026-07-12 [OLD 3+次未处理]
    路径: F:\...\艳青1_2_merged_full_16x9_1920x1080_douyin.mp4
    标题: 【胭脂虎】艳青力量燃脂操 | 塑腰臀跟练 | 细柳营健身

==================================================
指令:
  1,2,3    标记为已传 (空格或逗号分隔)
  o 1      用系统播放器打开视频1
  p 1      复制视频1路径到剪贴板
  s 1      跳过 (下次还弹)
  r 1      恢复 (把 [OLD 3+次] 重置为待提醒)
  a        全部标记为已传
  h        查看帮助
  q        退出 (不标已传, 保留状态)
==================================================
> 
```

ANSI 颜色 (Windows Terminal / VSCode 终端支持):
- 标题黄 `[1]` 编号
- 路径灰
- `[OLD 3+次]` 黄底黑字
- 当前黄金时段 ✓ 绿色

**Win10/11 cmd.exe 默认不开启 ANSI** —— 用 `os.system('')` (空字符串) 调 `vt100 enable` 或 `color` 命令开 ANSI, 退化到无色不报错。

## 错误处理

| 场景 | 处理 |
|------|------|
| 不在黄金时段手动跑 | 顶部提示"当前不在黄金期, 还要继续吗? (y/n)" |
| output/ 为空 | 提示"今天无待传, 关闭窗口" + 等任意键退出 |
| log.json 损坏 | 备份到 .bak + 重生成空 log, 顶部 WARN |
| log.json 字段缺失 (e.g. 新加的字段) | `.get()` 默认值兜底, 不崩 |
| Windows Task Scheduler 未装 / 失败 | install 脚本检测 schtasks.exe 退出码, 提示用户 |
| 文件已被删除 (output/ 里找不到) | 标记为已传 (标记时检查文件存在) |
| `o 1` 打开失败 | 提示"打开失败, 文件不存在或无关联程序" |
| 路径含特殊字符 | 用 `pathlib.Path`, 不手拼字符串 |

## Windows Task Scheduler 调度

`install_reminder_task.bat` 用 `schtasks /create` 注册任务:
- 任务名: `FitnessVideoPipeline_UploadReminder`
- 触发器: 每天 6 个时点 (10:00 / 12:00 / 14:00 / 19:00 / 21:00 / 23:00)
- 操作: `python tools/upload_reminder.py` (用 `.venv\Scripts\python.exe`)
- 工作目录: 项目根 (`F:\wkspace\fitness-video-pipeline`)
- 描述: "弹窗提醒上传 YouTube/抖音视频, 黄金时段触发"

`uninstall_reminder_task.bat` 反向 `schtasks /delete`.

**用 .bat 不用 .ps1** —— 双击直接跑, 不需要执行策略调整。

## 测试

3 套纯函数测试 (不跑 input/output 交互):

1. `tests/test_reminder_state.py` (~5 tests)
   - `load()` 空文件 → `{}`
   - `save()` 后 `load()` 数据一致
   - `mark_uploaded()` 设置时间戳 + 重置 `remind_count`
   - `increment_remind()` 累加, 达 3 设 `auto_archived`
   - 损坏 JSON → 返回 `{}` + 备份

2. `tests/test_reminder_scan.py` (~5 tests)
   - 扫 `output/2026-07-12/` 找到 3 个三件套
   - 跳过中间产物 (`*_color.mp4`, `*_watermark.mp4` 等)
   - 跳过 `*_full_16x9.mp4` 副本 (CLAUDE §"上传只传 *final_16x9")
   - 过滤已 `marked_uploaded` 的
   - 过滤 `auto_archived` 的

3. `tests/test_reminder_golden.py` (~5 tests, 复用 `_is_golden_hour` 边界)
   - 13:59 北京 → False (边界)
   - 14:00 北京 → True
   - 14:59 北京 → True
   - 15:00 北京 → False
   - 22:59 北京 → True (夜间段)

总计 ~15 tests, 0 GPU / 0 网络, 跑 < 1s.

## 跟现有组件的复用

- **`lib/upload_utils.py:435 _is_golden_hour()`** — 直接 import
- **`lib/upload_utils.py:443 seconds_until_next_golden()`** — 直接 import
- **`lib/upload_utils.py:57 build_title()`** — 弹窗显示标题用
- **`records/upload_manifest.json`** — 不读 (用户说全手工, manifest 不可靠)
- **CLAUDE.md §"`_cleanup_intermediate.py` (6/29 的非本轮产物) 是否删除，等用户定"** — 不动

## 不做 (YAGNI)

- ❌ 自动调 `tools/upload_youtube.py` 上传 (per 拍板, 全停)
- ❌ GUI 弹窗 (per 拍板, 命令行即可)
- ❌ 读 `upload_manifest.json` 判 YouTube 已传 (per 拍板, 不依赖 manifest)
- ❌ 备份 log.json 到云端 / git (本地文件即可, gitignore `records/*.json` 已配)
- ❌ 多账号 (一个频道就够)
- ❌ 国际化 (中文)
- ❌ 自动关窗 (per 拍板, 等你 X)

## 交付清单

1. `tools/upload_reminder.py` (~150 行)
2. `lib/reminder_state.py` (~80 行)
3. `tools/install_reminder_task.bat` (~30 行)
4. `tools/uninstall_reminder_task.bat` (~10 行)
5. `tests/test_reminder_state.py` (~50 行)
6. `tests/test_reminder_scan.py` (~60 行)
7. `tests/test_reminder_golden.py` (~30 行)
8. 更新 `CLAUDE.md` §"独立工具 (tools/)" 加 1 行
9. 更新 `memory/MEMORY.md` 加 1 行指针
10. 用户手工跑 `install_reminder_task.bat` 装上 (脚本不自动装)

## 验收

- [ ] `uv run python tools/upload_reminder.py` 在黄金时段能弹窗, 列出待传
- [ ] 输入 `1,2` 标记已传, 退出后再跑, 1+2 不再弹
- [ ] 输入 `o 1` 打开视频1 (Win10/11 验证)
- [ ] 同一视频连续 3 次不标 → 第 4 次不再弹, 顶部 [OLD 3+次]
- [ ] `install_reminder_task.bat` 双击注册, Task Scheduler 看到 6 个触发器
- [ ] `uninstall_reminder_task.bat` 卸载干净
- [ ] `uv run pytest tests/test_reminder_*.py` 15/15 绿
- [ ] 244 + 15 = 259 tests 全绿零回归
- [ ] 不修改主管线 (stages/main.py/pipeline/) 任何代码 (新文件仅在 tools/lib/tests/memory/CLAUDE.md)

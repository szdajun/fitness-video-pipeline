---
name: shorts-golden-hour-auto-publish-2026-07-12
description: 【2026-07-12 用户拍板】Shorts 自动等黄金时段 (10-14 / 19-23 北京) 再发布, 长视频保持手工发. 数据驱动: 用户频道 195 视频 5 维度诊断
metadata:
  type: project
---

# Shorts 黄金时段自动发布 — 2026-07-12 用户拍板上线

## 状态
✅ 已落 commit. 199 tests 全绿 (187 → 199, +12 守门). 用户拍板: **"自动发, 人容易忘记"**.

## 用户拍板原话
1. "再就是发布时机和流量的关系分析一下"
2. "主要是每天何时发布最合适"
3. "两高峰任一优先" (10-14 / 19-23 任一)
4. "任意" (不挑剔具体 13 / 22)
5. "最好自动发, 人容易忘记"
6. "长视频我来发" (历史已证自动发挂死)
7. "现在长视频自动发了都得不到处理, 挂死了" (确认规则)

## 数据驱动 (用户频道真实 195 视频分析)

### 5 维度诊断
1. **教练 IP**: 8+ 教练分散精力, 新 IP 首爆 (三宝妈 2521/雷震子 891/节拍战神 1843) 比老 IP 高
2. **时间趋势**: 2026-05 黄金期 1344/视频 → 2026-07 跌到 384, **频道在衰减**
3. **Shorts vs Long**: ROI 14.7x, 96 个 Long 只拿 6.2% 流量, **全面转 Shorts**
4. **hashtag**: 2026-07 之后用的 #暴汗燃脂瘦全身 等全军覆没 (25 view/视频), 黄金期 #kpop (2939) #每天坚持运动打卡 (2426) #dance (1892) 是天花板
5. **标题模式**: 当前模板 (488 view) 比早期"性感词+多 hashtag"模板 (2042 view) 差 4 倍

### 时长分布 (用户回答"短视频 30 秒是不是短了")
| 时长 | 数量 | 均 view | 爆款率 (>1500) |
|------|------|---------|----------------|
| 0-15s | 1 | 2939 | 100% 🏆 |
| 15-30s | 11 | 1043 | 36.4% |
| **30-45s** | **78** | **941** | 20.5% (甜区) |
| 45-60s | 5 | 157 | 0% |
| 60-90s | 38 | 104 | 0% (死亡) |
| 90-180s | 46 | 40 | 0% |
| 180s+ | 16 | 60 | 0% |

→ 30 秒**不算短**, 用户甜区 30-45s. 用户拍板"不动时长".

### 发布时段 (黄金时段数据)
**5 月黄金期按小时均 view** (北京 UTC+8):
- **13-14**: 1376 view (n=5, 单时段天花板)
- **22-23**: 935 view (n=10, 睡前刷)
- **19-20**: 862 view (n=8, 晚饭)
- **11-12**: 1739 view (n=1, 样本太小)
- **20-21**: 798 view (n=9)
- **0-1**: 856 view (n=4)
- **6-7**: 526 view (n=1)

**避开低谷**:
- 8-10: 98 view (n=2, 早晨匆忙)
- 14-18: 199-468 view (下午)

→ 黄金窗口 **10-14 / 19-23** 双高峰.

## 改动 — lib/upload_utils.py

### 新增函数
```python
def _is_golden_hour(now=None) -> bool:
    """10-14 或 19-23 北京时间 (半开区间)."""
    
def seconds_until_next_golden(now=None) -> int:
    """到下一个黄金时段开始的秒数."""
    
def wait_for_golden_hour(check_interval_sec=60, progress_print_min=5):
    """客户端 sleep 到黄金时段 (不用 publishAt, 绕开 YT 长视频挂死 bug)."""
```

### upload_pair 改动
- 加 `wait_for_short_golden_hour: bool = True` 默认参数
- short 上传分支**先** `wait_for_golden_hour()` 再 `upload_video(publish_at=None)`
- long 上传分支**不变** (保持"立即发布"规则)

## 守门 — 199 tests 全绿

`tests/test_upload_golden_hour.py` 新增 12 tests:

### 单元测试 (8 tests)
- `TestIsGoldenHour` (4):
  - `test_in_golden_window_10_to_14`: 10:00, 11:30, 13:59 → True
  - `test_in_golden_window_19_to_23`: 19:00, 20:30, 22:59 → True
  - `test_outside_golden_window`: 8:00, 9:59, 14:00, 15:30, 16:00, 17:30, 18:59, 23:00, 0:30, 5:00 → False
  - `test_boundary_excluded`: 14:00 False (半开), 23:00 False, 19:00 True, 18:59 False

- `TestSecondsUntilNextGolden` (4):
  - `test_in_golden_returns_zero`: 13:30, 22:00 → 0
  - `test_morning_before_10_waits_to_10`: 8:00 → 7200s, 9:30 → 1800s
  - `test_afternoon_14_to_18_waits_to_19`: 14:00 → 18000s, 15:30 → 12600s, 18:59 → 60s
  - `test_evening_after_23_waits_to_next_day_10`: 23:00 → 39600s, 0:30 → 34200s, 5:00 → 18000s

### 集成测试 (4 tests)
- `test_upload_pair_default_wait_for_short_golden_hour`: 默认 True
- `test_no_publishat_used_in_short_path`: short 分支不传 publish_at
- `test_long_path_also_publishes_immediately`: long 分支不传 publish_at (per CLAUDE 钉死)
- `test_golden_hour_does_not_use_publishat`: wait_for_golden_hour 函数体内不引用 publish_at

## 为什么不用 publishAt

per memory `yt-long-video-publish-immediately` (CLAUDE 钉死规则):
- YT 宽幅长视频 scheduled (publishAt) 延迟发布会**挂死在平台得不到处理** (HD processing 卡死)
- 用户 2026-07-12 实证: "现在长视频自动发了都得不到处理, 挂死了"
- 解决: Shorts 走**客户端 sleep + 立即发**, 不用 publishAt — 绕过 bug

## 已知限制

1. **主管线常驻**: `wait_for_golden_hour()` 调用 `time.sleep(60)` 循环, 主管线进程要等几小时. 如果中途 kill, 上传不发生. 应对: auto_publish.py --loop / --watch 模式常驻.
2. **错过黄金时段就等下一个**: 如果 22:00 完成跑批, 等到明天 10:00 (12 小时延迟). 这是设计 — 避开 publishAt bug.
3. **时区**: 北京 UTC+8 写死, 不支持其它时区频道.

## 重启信号 (不主动)
1. 用户拍板改时段 (例如数据证明 19-23 更好 → 收窄到 13 + 22)
2. YT 修了长视频 scheduled bug → 可以统一用 publishAt
3. 频道扩展到其它时区

## 关联
- 主管线 199 tests 全绿零回归
- memory yt-long-video-publish-immediately (历史教训)
- CLAUDE.md 第 261 行 "Long 立即发布" + 第 263 行 "Shorts 黄金时段自动发布"
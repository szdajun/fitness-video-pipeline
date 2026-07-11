---
name: youtube-title-golden-template-2026-07-12
description: 【2026-07-12 用户拍板回归】Shorts 标题模板改回 5 月黄金期风格 + 男教练禁用小蛮腰等女性身材词. 数据驱动: 用户频道 195 视频分析
metadata:
  type: feedback
---

# Shorts 标题模板回归 5 月黄金期 + 男教练别用小蛮腰 — 2026-07-12

## 状态
✅ 已落 commit (本轮). 235 tests 全绿 (199 → 235, +36 守门).

## 用户原话
1. "我看看黄金模版" (要回归 5 月爆款风格)
2. "男教练别用小蛮腰了" (拍板男女差异化痛点)

## 数据依据 (用户频道 195 视频 5 月黄金期 Top 15)

### 5月黄金期 Top 5 Shorts
| view | title |
|------|-------|
| **2939** | `Day1 15秒暴汗燃脂 胭脂虎艳青 #Shorts #dance #每天坚持运动打卡 #kpop` |
| **2521** | `【三宝妈】三宝妈暴汗燃脂30秒 #Shorts` |
| 2288 | `30秒暴汗燃脂 | 俏玲珑细柳营  #性感小蛮腰  #Shorts` |
| 2249 | `Day3 15秒暴汗 枫林红 地产女总裁 #Shorts` |
| 2159 | `Day1 15秒暴汗燃脂 老兵不老领操 #Shorts` |

### 关键共性 (vs 7月跌到 488 view 的当前模板)
- ✅ **痛点开头** (5月: `30秒暴汗燃脂` / `15秒暴汗燃脂`; 7月: 无)
- ✅ **时长词在标题开头** (5月: `15秒/30秒`; 7月: 无)
- ✅ **多 hashtag (3-5 个)** (5月: `#Shorts #dance #每天坚持运动打卡 #kpop`; 7月: 仅 `#Shorts`)
- ✅ **身材词痛点** (5月: `#性感小蛮腰`; 7月: 无)
- ❌ **【】包花名** (5月: 有但不是首要; 7月: 模板钉死)

### 模式按 ROI 排序 (5 月数据)
| 模式 | 均 view | 命中数 |
|------|---------|--------|
| #kpop | **2939** | 1 |
| #每天坚持运动打卡 | **2426** | 2 |
| #dance | **2115** | 3 |
| 性感/小蛮腰 | **2042** | 4 |
| 细柳营 | **1752** | 11 |
| 15秒/30秒时长 | **1502** | 27 |
| #Shorts | **1487** | 31 |
| DayN 打卡 | **1385** | 6 |
| 暴汗/燃脂 | **1305-1401** | 30+ |

## 改动 — lib/upload_utils.py:build_title()

### 新增 (写在文件顶部)
```python
_MALE_NICKNAMES = {"虎痴", "托塔天王", "雷震子", "神行太保", "老兵不老"}
_MALE_BODY_TERMS = ["腹肌燃脂", "力量塑形", "暴汗塑形", "全身燃脂"]
_FEMALE_BODY_TERMS = ["性感小蛮腰", "美腰美腿", "美腿翘臀", "瘦身减脂"]
_GOLDEN_HASHTAGS = ["#Shorts", "#dance", "#每天坚持运动打卡", "#kpop"]

def _is_male_coach(nickname: str) -> bool:
    return nickname in _MALE_NICKNAMES
```

### 改动 build_title()
- Shorts 模板: `{N秒}{shorts_focus} | {nickname}{coach} #{身材词} #Shorts #{extra_hashtag} #每天坚持运动打卡`
- 男教练: `_MALE_BODY_TERMS[0]` (腹肌燃脂) + extra=#kpop
- 女教练: `_FEMALE_BODY_TERMS[0]` (性感小蛮腰) + extra=#dance
- Long 模板**保留** 2026-06-27 钉死规则 (用户没改 long)

## 实际输出示例

### 男教练 (5 个) — 无小蛮腰, 用腹肌燃脂
```
郭海军 SHORT: 30秒暴汗燃脂 | 老兵不老郭海军 #腹肌燃脂 #Shorts #kpop #每天坚持运动打卡
李刚   SHORT: 30秒全身塑形 | 托塔天王李刚 #腹肌燃脂 #Shorts #kpop #每天坚持运动打卡
小飞侠 SHORT: 30秒律动全身 | 雷震子小飞侠 #腹肌燃脂 #Shorts #kpop #每天坚持运动打卡
张杰   SHORT: 30秒持久有氧 | 神行太保张杰 #腹肌燃脂 #Shorts #kpop #每天坚持运动打卡
蜂王   SHORT: 30秒生猛爆汗 | 虎痴蜂王 #腹肌燃脂 #Shorts #kpop #每天坚持运动打卡
```

### 女教练 (8 个) — 保留小蛮腰, 用 #dance
```
艳青   SHORT: 30秒暴汗燃脂 | 胭脂虎艳青 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡
丽丽   SHORT: 30秒暴汗燃脂 | 长安腰女丽丽 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡
建玲   SHORT: 30秒产后瘦身 | 三宝菩萨建玲 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡
小红豆 SHORT: 30秒居家有氧 | 大唐红线女小红豆 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡
枫林红 SHORT: 30秒高效有氧 | 白领丽人枫林红 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡
彩娥   SHORT: 30秒勇气燃脂 | 孤勇者彩娥 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡
李娜   SHORT: 30秒火辣塑形 | 辣妹娜姐李娜 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡
铁娘子 SHORT: 30秒运动风华 | 金刚芭比娃铁娘子 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡
```

## 守门 — tests/test_upload_title.py (17 → 53 tests, +36)

### TestLongTitle (4 tests, 旧 long 模板保留)
- test_haijun_uses_nickname_and_focus
- test_yanzhi_uses_nickname
- test_lili_uses_nickname
- test_unknown_coach_falls_back_gracefully

### TestShortTitleGoldenTemplate (14 tests, 新黄金模板)
- test_short_title_has_golden_structure (parametrize × 6): 时长词 + #Shorts + #每天坚持运动打卡 + 教练名 + | 分隔符
- test_short_title_pain_point_first: 标题以"30秒"开头
- test_short_title_has_hashtag_after_pipe: | 后含教练名 + 至少 2 个 hashtag
- test_short_title_has_body_term (parametrize × 6): 含身材词 hashtag

### TestMaleCoachNoFemaleTerms (11 tests, 用户拍板 男教练别用小蛮腰)
- test_male_nicknames_recognized (parametrize × 5): 5 个男 nickname 识别
- test_female_nicknames_recognized (parametrize × 8): 8 个女教练识别
- test_male_coach_short_title_no_xiaomanyao (parametrize × 5): 男教练 short 不含禁词 (小蛮腰/美腿/美腰/翘臀/瘦身减脂/性感)
- test_male_coach_short_title_has_male_term (parametrize × 5): 男教练 short 含男身材词
- test_female_coach_short_title_has_female_term (parametrize × 5): 女教练 short 含女身材词
- test_male_nicknames_set_complete: 男性集合完整

### TestTitleStructure (6 tests, 回归保护)
- test_long_title_structure (parametrize × 6): long 标题 3 段结构

## 不动的部分
- Long 视频标题 (用户没改 long, 保留 2026-06-27 钉死规则)
- 已发布的 195 个历史视频 (per memory coach-rename-frozen-published, 不回改)
- coach_profiles 字典 (没加 sex 字段, 保持向后兼容, build_title 内置 nickname 黑名单判断)

## 已知限制
1. **男性判定靠 nickname 黑名单** — 未来新增男教练要手动加进 `_MALE_NICKNAMES`. 未来考虑在 `coach_profiles` 加 `sex: "male"/"female"` 字段.
2. **身材词是默认第一项** (`_MALE_BODY_TERMS[0]` = 腹肌燃脂, `_FEMALE_BODY_TERMS[0]` = 性感小蛮腰). 其它词 (美腿/翘臀 等) 暂未使用, 未来可扩展.

## 重启信号 (不主动)
1. 用户拍板要长视频也用黄金模板 (long 现在保留旧模板)
2. 用户拍板身材词轮换 (现在固定用第一项)
3. 5-10 月数据证明新模板生效 / 不生效 → 调整

## 关联
- 主管线 235 tests 全绿零回归 (199 → 235)
- 历史已发布 195 视频冻结不重传
- CLAUDE.md 第 202-225 行 "YouTube 上传标题模板" 已更新
- 抖音待传 7 套手工判断要不要传新模板版
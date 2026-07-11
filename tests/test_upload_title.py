"""测试 YouTube 标题模板 — 2026-07-12 回归黄金期模板

数据依据: 用户频道 195 视频, 5 月黄金期均 view 1500+, 7 月跌到 488.
黄金模板 (Top 1 2939 view, Top 2 2288 view):
- Shorts: {N秒}{shorts_focus} | {nickname}{coach} #{身材词} #Shorts #dance/#kpop #每天坚持运动打卡
- Long:   【{nickname}】{coach}{focus}操 | {focus}跟练 | 细柳营健身 (旧模板保留)

男教练禁用女性身材词 (用户 2026-07-12 拍板).
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.upload_utils import build_title, _is_male_coach, _MALE_NICKNAMES


class TestLongTitle:
    """长视频标题保留 CLAUDE.md 2026-06-27 钉死模板 (用户没改 long)."""

    def test_haijun_uses_nickname_and_focus(self):
        """郭海军: nickname=老兵不老, focus=刚劲塑形"""
        assert build_title("郭海军", "2026-07-12", "long") == \
            "【老兵不老】郭海军刚劲塑形操 | 刚劲塑形跟练 | 细柳营健身"

    def test_yanzhi_uses_nickname(self):
        """艳青: 应有 nickname"""
        title = build_title("艳青", "2026-07-12", "long")
        assert title.startswith("【")
        assert "艳青" in title
        assert title.endswith("细柳营健身")

    def test_lili_uses_nickname(self):
        """丽丽: 应有 nickname"""
        title = build_title("丽丽", "2026-07-12", "long")
        assert title.startswith("【")
        assert "丽丽" in title

    def test_unknown_coach_falls_back_gracefully(self):
        """未知教练不应崩溃"""
        title = build_title("新教练", "2026-07-12", "long")
        assert "新教练" in title
        assert "细柳营健身" in title


class TestShortTitleGoldenTemplate:
    """Shorts 黄金期标题模板 (2026-07-12 用户拍板回归 5 月爆款风格)."""

    @pytest.mark.parametrize("coach", ["郭海军", "艳青", "丽丽", "建玲", "小红豆", "枫林红"])
    def test_short_title_has_golden_structure(self, coach):
        """黄金模板结构: {N秒}{shorts_focus} | {nickname}{coach} #{身材词} #Shorts ... #每天坚持运动打卡"""
        title = build_title(coach, "2026-07-12", "short", duration_sec=30)
        # 必须含 30秒 (时长)
        assert "30秒" in title, f"{coach}: 缺时长词 30秒: {title}"
        # 必须含 #Shorts
        assert "#Shorts" in title, f"{coach}: 缺 #Shorts: {title}"
        # 必须含 #每天坚持运动打卡 (5月黄金 hashtag)
        assert "#每天坚持运动打卡" in title, f"{coach}: 缺黄金 hashtag: {title}"
        # 必须含 coach 名
        assert coach in title, f"{coach}: 缺教练名: {title}"
        # 必须含分隔符 |
        assert " | " in title, f"{coach}: 缺 | 分隔符: {title}"

    def test_short_title_pain_point_first(self):
        """痛点 (shorts_focus) 在开头 — 5月 Top 1 风格"""
        title = build_title("艳青", "", "short", duration_sec=30)
        # 标题应该以 "30秒" 开头 (时长 + 痛点开头)
        assert title.startswith("30秒"), f"痛点未在开头: {title}"

    def test_short_title_has_hashtag_after_pipe(self):
        """| 后是教练+身材词+hashtag 块 — 黄金模板特征"""
        title = build_title("丽丽", "", "short", duration_sec=30)
        parts = title.split(" | ")
        assert len(parts) == 2, f"应有 2 段 (前/后 |), 实际: {title}"
        # 前段: 时长 + 痛点 (e.g. "30秒暴汗燃脂")
        # 后段: {nickname}{coach} #{身材词} #Shorts ... (e.g. "长安腰女丽丽 #性感小蛮腰 #Shorts #dance #每天坚持运动打卡")
        assert "丽丽" in parts[1], f"第二段缺教练名: {parts[1]}"
        # 第二段必须含至少 2 个 hashtag
        import re
        hashtag_count = len(re.findall(r'#\S+', parts[1]))
        assert hashtag_count >= 2, f"第二段 hashtag 太少 ({hashtag_count}): {parts[1]}"

    @pytest.mark.parametrize("coach", ["郭海军", "艳青", "丽丽", "建玲", "小红豆", "枫林红"])
    def test_short_title_has_body_term(self, coach):
        """必须有身材词 hashtag (痛点)"""
        title = build_title(coach, "", "short", duration_sec=30)
        # 检查含 #身材词 模式
        import re
        has_body_hashtag = bool(re.search(r'#(腹肌|力量塑形|暴汗塑形|全身燃脂|性感小蛮腰|美腰美腿|美腿翘臀|瘦身减脂)', title))
        assert has_body_hashtag, f"{coach}: 缺身材词 hashtag: {title}"


class TestMaleCoachNoFemaleTerms:
    """用户 2026-07-12 拍板: 男教练不用小蛮腰等女性身材词."""

    @pytest.mark.parametrize("nickname", list(_MALE_NICKNAMES))
    def test_male_nicknames_recognized(self, nickname):
        """5 个男教练 nickname 必须被识别"""
        assert _is_male_coach(nickname) is True, f"{nickname}: 应识别为男教练"

    @pytest.mark.parametrize("female_coach", ["艳青", "丽丽", "建玲", "小红豆", "枫林红", "彩娥", "李娜", "铁娘子"])
    def test_female_nicknames_recognized(self, female_coach):
        """女教练 nickname 不在男性集合"""
        # 取 coach 的 nickname
        from lib.coach_profiles import COACH_PROFILES
        nickname = COACH_PROFILES[female_coach].get("nickname", female_coach)
        assert _is_male_coach(nickname) is False, f"{female_coach} (nickname={nickname}): 应识别为女教练"

    @pytest.mark.parametrize("coach", ["郭海军", "李刚", "小飞侠", "张杰", "蜂王"])
    def test_male_coach_short_title_no_xiaomanyao(self, coach):
        """男教练 Shorts 标题不能含 '小蛮腰'/'美腿'/'美腰' 等女性身材词"""
        title = build_title(coach, "", "short", duration_sec=30)
        forbidden = ["小蛮腰", "美腿", "美腰", "翘臀", "瘦身减脂", "性感"]
        for word in forbidden:
            assert word not in title, f"{coach}: 含禁词 '{word}': {title}"

    @pytest.mark.parametrize("coach", ["郭海军", "李刚", "小飞侠", "张杰", "蜂王"])
    def test_male_coach_short_title_has_male_term(self, coach):
        """男教练 Shorts 必须用男身材词 (腹肌/力量塑形/暴汗塑形)"""
        title = build_title(coach, "", "short", duration_sec=30)
        male_terms = ["腹肌燃脂", "力量塑形", "暴汗塑形", "全身燃脂"]
        has_male = any(t in title for t in male_terms)
        assert has_male, f"{coach}: 缺男性身材词: {title}"

    @pytest.mark.parametrize("coach", ["艳青", "丽丽", "建玲", "小红豆", "枫林红"])
    def test_female_coach_short_title_has_female_term(self, coach):
        """女教练 Shorts 用女性身材词 (性感小蛮腰/美腰美腿等)"""
        title = build_title(coach, "", "short", duration_sec=30)
        female_terms = ["性感小蛮腰", "美腰美腿", "美腿翘臀", "瘦身减脂"]
        has_female = any(t in title for t in female_terms)
        assert has_female, f"{coach}: 缺女性身材词: {title}"

    def test_male_nicknames_set_complete(self):
        """男性 nickname 集合必须包含 5 月数据里所有男教练."""
        # 5月 Top 15 Shorts 里的男教练花名: 老兵不老(郭海军)/托塔天王(李刚)/雷震子(小飞侠)
        required = {"老兵不老", "托塔天王", "雷震子", "神行太保", "虎痴"}
        assert required.issubset(_MALE_NICKNAMES), \
            f"男性集合缺: {required - _MALE_NICKNAMES}"


class TestUniversalHashtags:
    """2026-07-12 用户拍板: Shorts 加 3 个常驻通用 hashtag (参考小马达频道 5-7 hashtag/视频).

    数据依据: 用户频道 5月 #kpop 2939 / #每天坚持运动打卡 2426 / #dance 2115 是黄金 hashtag.
    通用 hashtag (健身操/全身燃脂/居家健身) 跟小马达标题风格对齐.
    """

    def test_short_title_has_at_least_5_hashtags(self):
        """Shorts 标题至少含 5 个 hashtag (黄金模板: #身材词 + #Shorts + #extra + #每天坚持运动打卡 + 3 常驻通用 = 6)"""
        import re
        for coach in ["郭海军", "艳青", "丽丽", "建玲", "小红豆", "枫林红"]:
            title = build_title(coach, "", "short", duration_sec=30)
            count = len(re.findall(r'#\S+', title))
            assert count >= 5, f"{coach}: hashtag 太少 ({count}), 标题: {title}"

    @pytest.mark.parametrize("coach", ["郭海军", "艳青", "丽丽", "建玲", "小红豆", "枫林红"])
    def test_short_title_has_3_universal_hashtags(self, coach):
        """3 个常驻通用 hashtag: #健身操 + #全身燃脂 + #居家健身"""
        title = build_title(coach, "", "short", duration_sec=30)
        for tag in ["#健身操", "#全身燃脂", "#居家健身"]:
            assert tag in title, f"{coach}: 缺常驻 hashtag {tag}: {title}"

    def test_male_short_has_at_least_6_hashtags(self):
        """男教练 hashtag 数 ≥ 6 (1 身材词 + 1 Shorts + 1 dance/kpop + 1 #每天 + 3 通用)"""
        import re
        for coach in ["郭海军", "李刚", "小飞侠", "张杰", "蜂王"]:
            title = build_title(coach, "", "short", duration_sec=30)
            count = len(re.findall(r'#\S+', title))
            assert count >= 6, f"{coach} (男): hashtag {count}: {title}"

    def test_female_short_has_at_least_6_hashtags(self):
        """女教练 hashtag 数 ≥ 6 (1 身材词 + 1 Shorts + 1 dance/kpop + 1 #每天 + 3 通用)"""
        import re
        for coach in ["艳青", "丽丽", "建玲", "小红豆", "枫林红"]:
            title = build_title(coach, "", "short", duration_sec=30)
            count = len(re.findall(r'#\S+', title))
            assert count >= 6, f"{coach} (女): hashtag {count}: {title}"


class TestTitleStructure:
    """所有长视频标题必须有统一结构 (回归保护)."""

    @pytest.mark.parametrize("coach", ["郭海军", "艳青", "丽丽", "建玲", "小红豆", "枫林红"])
    def test_long_title_structure(self, coach):
        title = build_title(coach, "2026-07-12", "long")
        parts = title.split(" | ")
        assert len(parts) == 3, f"{coach}: 标题应有 3 段, 实际: {title}"
        assert parts[-1] == "细柳营健身"
        assert parts[0].startswith("【"), f"{coach}: 第一段缺【 前缀: {parts[0]}"
        assert "】" in parts[0], f"{coach}: 第一段缺】 闭合: {parts[0]}"
        assert coach in title, f"{coach}: 标题中找不到教练名: {title}"
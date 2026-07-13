"""Tests for 雷震子 (小飞侠) 三件套文案 — profile 自定义 description 字段.

回归保护:
- long_description / douyin_description / thumbnail_suggestion 字段能被 get_coach 读到
- generate_description 优先用自定义 long_description, 没设时 fallback 通用模板
- 雷震子文案不串台到其他教练 (李刚/艳青 等文案不污染)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def test_get_leizhenzi_has_custom_description_fields():
    """雷震子 profile 应有 long_description + douyin_description + thumbnail_suggestion."""
    from lib.coach_profiles import get_coach

    coach = get_coach("雷震子")
    assert coach["nickname"] == "雷震子"
    assert coach.get("long_description"), "雷震子缺 long_description"
    assert coach.get("douyin_description"), "雷震子缺 douyin_description"
    assert coach.get("thumbnail_suggestion"), "雷震子缺 thumbnail_suggestion"


def test_get_leizhenzi_long_description_mentions_rhythm():
    """雷震子 long_description 必须含律动/节奏特征 (不串台到李刚的力量燃脂)."""
    from lib.coach_profiles import get_coach

    desc = get_coach("雷震子")["long_description"]
    assert "律动" in desc or "节奏" in desc or "节拍" in desc, (
        f"雷震子 long_description 应含律动/节奏, 实: {desc!r}"
    )
    # 不应含李刚专属关键词
    assert "托塔" not in desc and "塑形" not in desc, (
        f"雷震子 long_description 不应串台到李刚 (托塔/塑形), 实: {desc!r}"
    )


def test_get_leizhenzi_douyin_description_is_concise():
    """抖音文案应简短 + 有话题标签."""
    from lib.coach_profiles import get_coach

    desc = get_coach("雷震子")["douyin_description"]
    # 抖音文案不应过长
    assert len(desc) < 200, f"抖音文案应 < 200 字符, 实 {len(desc)}: {desc!r}"
    # 应含 hashtag
    assert "#" in desc, f"抖音文案应含 hashtag, 实: {desc!r}"


def test_get_leizhenzi_thumbnail_suggestion_short():
    """缩略图建议应短 (4-8 字)."""
    from lib.coach_profiles import get_coach

    sug = get_coach("雷震子")["thumbnail_suggestion"]
    assert 4 <= len(sug) <= 12, f"缩略图建议 4-12 字符, 实 {len(sug)}: {sug!r}"


def test_other_coaches_unaffected():
    """新增字段不应污染其他教练 — 李刚没设自定义描述时 generate_description 走模板."""
    from lib.coach_profiles import get_coach, generate_description

    coach = get_coach("李刚")
    # 李刚没设 long_description → 应走模板路径 (不报错)
    desc = generate_description(coach, {"channel": "细柳营健身"}, duration="45分钟")
    assert "李刚" in desc
    assert "托塔天王" in desc
    # 李刚的通用模板不会出现雷震子专属词
    assert "雷震" not in desc and "节拍" not in desc, (
        f"李刚 description 不应含雷震子词, 实: {desc!r}"
    )


def test_get_coach_by_real_name_also_works():
    """小飞侠 (真名) 也能 get 到雷震子 profile."""
    from lib.coach_profiles import get_coach

    coach = get_coach("小飞侠")
    assert coach["nickname"] == "雷震子"
    assert coach.get("long_description")
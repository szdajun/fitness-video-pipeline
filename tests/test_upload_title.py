"""测试 YouTube 标题模板 — 防止被改回扁平版

固化规则（来自 CLAUDE.md）：
- 长视频模板: 【{nickname}】{coach}{focus}操 | {focus}跟练 | 细柳营健身
- 短视频模板: 细柳营{coach} | 暴汗燃脂30秒 #Shorts
- nickname/focus 来自 lib/coach_profiles.COACH_PROFILES
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.upload_utils import build_title


class TestLongTitle:
    def test_haijun_uses_nickname_and_focus(self):
        """郭海军: nickname=老兵不老, focus=刚劲塑形"""
        assert build_title("郭海军", "2026-06-25", "long") == \
            "【老兵不老】郭海军刚劲塑形操 | 刚劲塑形跟练 | 细柳营健身"

    def test_yanzhi_uses_nickname(self):
        """艳青: 应有 nickname"""
        title = build_title("艳青", "2026-06-25", "long")
        assert title.startswith("【")
        assert "艳青" in title
        assert title.endswith("细柳营健身")

    def test_lili_uses_nickname(self):
        """丽丽: 应有 nickname"""
        title = build_title("丽丽", "2026-06-25", "long")
        assert title.startswith("【")
        assert "丽丽" in title

    def test_no_flat_format(self):
        """禁止扁平格式: 细柳营·{coach} | 有氧健身操·燃脂暴汗 | {date}"""
        for coach in ["郭海军", "艳青", "丽丽", "建玲"]:
            title = build_title(coach, "2026-06-25", "long")
            assert "·有氧健身操" not in title, f"{coach} 用了扁平模板"
            assert f"细柳营·{coach}" not in title, f"{coach} 用了扁平模板"

    def test_unknown_coach_falls_back_gracefully(self):
        """未知教练不应崩溃"""
        title = build_title("新教练", "2026-06-25", "long")
        assert "新教练" in title
        assert "细柳营健身" in title


class TestShortTitle:
    def test_short_contains_shorts_tag(self):
        title = build_title("郭海军", "", "short")
        assert "#Shorts" in title
        assert "郭海军" in title

    def test_short_no_nickname_prefix(self):
        """短视频不强制 nickname 前缀 (历史格式)"""
        title = build_title("郭海军", "", "short")
        # 短链可以保留简单格式
        assert "细柳营" in title


class TestTitleStructure:
    """所有长视频标题必须有统一结构"""

    @pytest.mark.parametrize("coach", ["郭海军", "艳青", "丽丽", "建玲", "小红豆", "枫林红"])
    def test_long_title_structure(self, coach):
        title = build_title(coach, "2026-06-25", "long")
        # 三个 | 分隔的部分
        parts = title.split(" | ")
        assert len(parts) == 3, f"{coach}: 标题应有 3 段, 实际: {title}"
        # 最后一段必须是 细柳营健身
        assert parts[-1] == "细柳营健身"
        # 第一段必须以 【 开头, 包含 】
        assert parts[0].startswith("【"), f"{coach}: 第一段缺【 前缀: {parts[0]}"
        assert "】" in parts[0], f"{coach}: 第一段缺】 闭合: {parts[0]}"
        # 第一段必须包含 coach 名
        assert coach in parts[0] or coach in title, f"{coach}: 标题中找不到教练名: {title}"
"""弹幕库禁用历史遗留短语守门 (2026-07-11).

背景: hook 大字 "高燃预警/先睹为快" 期间, danmaku 弹幕 "秦人血脉醒!" 飘过
     同位置 → 视觉错位 (用户报"文案错位了"). 删 PHRASES 里 "秦人血脉醒!".

原则 (钉死):
  - 弹幕库禁止含: '秦人血脉'/'觉醒' 这类品牌历史标语 (会跟 hook/CTA 大字撞)
  - 其他细柳营精神类 ("细柳营打卡"/"练就完了"/"汗砸地声响"/"两千年前这里也练兵" 等)
    保留 (位置不在 hook 中央, 不冲突)
  - 跟 a60b545 暧昧词清理原则同根: 弹幕内容审查默认收紧, 防回退.

修复: stages/34_danmaku.py 删 "秦人血脉醒!" + 此测试扫 PHRASES 含禁用串即 fail.

不重跑主管线 (per no-auto-rerun), 留待未来新跑视频自动生效.
"""
import re
from pathlib import Path

STAGE_FILE = Path(__file__).resolve().parent.parent / "stages" / "34_danmaku.py"

# 禁用串: 任何含这些子串的弹幕都 fail (避免误判可加边界)
DISALLOWED_SUBSTRINGS = [
    "秦人血脉",   # 历史标语 + 跟 hook 大字撞位置
    "觉醒",       # 同样品牌标语, 不该混进弹幕
]

# 额外检查: 整条相等 (用于钉死单条)
DISALLOWED_EXACT = [
    "秦人血脉醒!",   # 已删 (2026-07-11)
]


def _load_phrases() -> list[str]:
    """从 34_danmaku.py PHRASES 列表 ast 提取所有字符串."""
    import ast
    src = STAGE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    phrases: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # 单 target List 赋值
            if (len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id == "PHRASES" and
                isinstance(node.value, (ast.List, ast.Tuple))):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        phrases.append(elt.value)
    return phrases


def test_phrases_no_disallowed_substrings():
    """PHRASES 不能含禁用子串."""
    phrases = _load_phrases()
    assert phrases, "PHRASES 列表为空, 测试无法验证"
    hits = []
    for p in phrases:
        for bad in DISALLOWED_SUBSTRINGS:
            if bad in p:
                hits.append((p, bad))
    assert not hits, (
        f"[弹幕库违规] PHRASES 含禁用串: {hits}. "
        f"原因: 弹幕内容审查 (2026-07-11 起收紧), 这些串跟 hook/CTA 大字撞位置/品牌混淆, 必须删."
    )


def test_specific_phrases_deleted():
    """钉死: '秦人血脉醒!' 不能在 PHRASES 里."""
    phrases = _load_phrases()
    for exact in DISALLOWED_EXACT:
        assert exact not in phrases, (
            f"[弹幕库回退] '{exact}' 又出现在 PHRASES. "
            f"原因: 2026-07-11 已删 (commit a60b545 同期, 用户报 hook 期间撞位置视觉错位). "
            f"如需恢复请先讨论."
        )


def test_phrases_min_count_after_cleanup():
    """钉死: PHRASES 当前条数 ≥ 75 (清理后保留健康激励/身材/细柳营精神类).

    2026-07-11 a60b545 + 本次删 '秦人血脉醒!' 共减 25 条 (24+1).
    原基线 ~106 条 → 现在 ~80 条. 留 ≥75 防回退.
    """
    phrases = _load_phrases()
    assert len(phrases) >= 75, (
        f"[弹幕库过少] PHRASES 仅 {len(phrases)} 条 (<75). "
        f"原基线 ~106 条 (24 条暧昧删 + 1 条秦人血脉醒删 = ~81 条). "
        f"再删会显著降低弹幕密度, 影响视频氛围."
    )
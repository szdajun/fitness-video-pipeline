"""弹幕进 final 修复守门测试 (2026-07-06).

背景: 主管线弹幕从未叠加到 final. 旧 main.py 顺序 danmaku → burst → export, 但
     burst fallback 链 (84d39a2 钉死) 让 burst 接力 mascot_path (face_swap 别名) 优先于
     danmaku_path → burst 跳过已叠弹幕的 danmaku 输出 → final 缺弹幕.

     抽帧实证: 艳青1/3_4、小飞侠1_2、丽丽4_5_6、枫林红2_3、郭海军1_2 全部 final 无弹幕.

修复: 调换 main.py 顺序 burst → danmaku → export. danmaku 接力 burst 输出
     (face_swap + 爆燃), 弹幕画在 burst 文字上, export 接力 danmaku 输出.
     final = face_swap + 爆燃 + 弹幕 全部齐.
     配套:
       - 34_danmaku fallback 链加 burst_path (接力 burst 输出)
       - 07_export fallback 链把 danmaku_path 提到 burst_path 之前 (接力弹幕)

这些测试用 AST 静态分析验证修复不能回退.
"""
import ast
import re
from pathlib import Path


STAGES_DIR = Path(__file__).resolve().parent.parent / "stages"
MAIN_PATH = Path(__file__).resolve().parent.parent / "main.py"


def _parse_chain_from_or(stage_file: Path, var_names: set[str]) -> list[str]:
    """从 stage 文件 ast 提取指定变量名的 fallback chain (or 表达式)."""
    src = stage_file.read_text(encoding="utf-8")
    tree = ast.parse(src)
    chain = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in var_names):
            if isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
                for v in node.value.values:
                    # str(ctx.input_path) 兜底
                    if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                            and v.func.id == "str"):
                        if (v.args and isinstance(v.args[0], ast.Attribute)
                                and v.args[0].attr == "input_path"):
                            chain.append("__input_path__")
                            continue
                    if not isinstance(v, ast.Call):
                        continue
                    func = v.func
                    if isinstance(func, ast.Attribute) and func.attr == "get":
                        if v.args and isinstance(v.args[0], ast.Constant):
                            chain.append(v.args[0].value)
                    elif isinstance(func, ast.Attribute) and func.attr == "input_path":
                        chain.append("__input_path__")
    return chain


# === burst (35) ===

def test_burst_mascot_before_danmaku_preserved():
    """stages/35 burst mascot_path 必须在 danmaku_path 之前 (commit 84d39a2 钉死, 不能改)."""
    chain = _parse_chain_from_or(STAGES_DIR / "35_intensity_burst.py", {"input_path"})
    assert "mascot_path" in chain, f"burst 链缺 mascot_path: {chain}"
    assert "danmaku_path" in chain, f"burst 链缺 danmaku_path: {chain}"
    assert chain.index("mascot_path") < chain.index("danmaku_path"), (
        f"burst 链 mascot_path 必须在 danmaku_path 之前 (face_swap 输出优先级最高, "
        f"84d39a2 钉死). 当前链: {chain}"
    )


def test_burst_smart_crop_first():
    """burst fallback 链 smart_crop_path 必须在最前 (2026-06-20 修复)."""
    chain = _parse_chain_from_or(STAGES_DIR / "35_intensity_burst.py", {"input_path"})
    assert chain and chain[0] == "smart_crop_path", (
        f"smart_crop_path 必须在 burst fallback 链最前. 实测第一项: "
        f"{chain[0] if chain else None} (完整: {chain})"
    )


def test_burst_fallback_has_input_path_tail():
    """burst fallback 链末尾必须 str(ctx.input_path) 兜底 (防全 None 崩)."""
    chain = _parse_chain_from_or(STAGES_DIR / "35_intensity_burst.py", {"input_path"})
    assert "__input_path__" in chain, (
        f"burst fallback 链末尾必须 str(ctx.input_path) 兜底, 实测: {chain}"
    )


# === danmaku (34) ===

def test_danmaku_fallback_includes_burst_path():
    """stages/34 danmaku fallback 链必须含 burst_path (2026-07-06 修复, 接力 burst 输出)."""
    chain = _parse_chain_from_or(STAGES_DIR / "34_danmaku.py", {"input_path"})
    assert "burst_path" in chain, (
        f"danmaku fallback 链必须含 burst_path (接力 burst 输出画弹幕). "
        f"实测: {chain}. "
        f"缺这个会导致弹幕叠到 watermark/energybar 链, 跳过 burst 文字."
    )


def test_danmaku_burst_before_mascot():
    """danmaku fallback 链 burst_path 必须在 mascot_path 之前 (burst 是后跑的接力链)."""
    chain = _parse_chain_from_or(STAGES_DIR / "34_danmaku.py", {"input_path"})
    if "burst_path" in chain and "mascot_path" in chain:
        bi = chain.index("burst_path")
        mi = chain.index("mascot_path")
        assert bi < mi, (
            f"danmaku 链 burst_path @ {bi} 应在 mascot_path @ {mi} 之前 "
            f"(main.py 2026-07-06 调换后 burst 在 danmaku 之前跑, danmaku 接力 burst). "
            f"链: {chain}"
        )


# === export (07) ===

def test_export_danmaku_before_burst():
    """stages/07 export fallback 链 danmaku_path 必须在 burst_path 之前 (2026-07-06 修复).

    旧链 burst > danmaku → export 选 burst_path (无弹幕) → final 缺弹幕.
    新链 danmaku > burst → export 选 danmaku_path (弹幕 + 爆燃) → final 完整.
    """
    chain = _parse_chain_from_or(STAGES_DIR / "07_export.py", {"processed_path"})
    assert "danmaku_path" in chain, f"export 链缺 danmaku_path: {chain}"
    assert "burst_path" in chain, f"export 链缺 burst_path: {chain}"
    di = chain.index("danmaku_path")
    bi = chain.index("burst_path")
    assert di < bi, (
        f"export 链 danmaku_path @ {di} 必须在 burst_path @ {bi} 之前 (弹幕进 final). "
        f"实测链: {chain}"
    )


# === main.py stage 顺序 (核心修复) ===

def test_main_burst_before_danmaku():
    """main.py 中 intensity_burst 必须在 danmaku 之前 (2026-07-06 修复, 治本).

    调换理由: 旧顺序 danmaku → burst → export, danmaku 跑过但 burst fallback
    钉死 mascot > danmaku, burst 接力 face_swap 跳过 danmaku 输出, final 缺弹幕.
    新顺序 burst → danmaku → export: burst 接力 face_swap 输出, danmaku 接力 burst
    输出画弹幕到 burst 文字上, export 接力 danmaku 输出. final = 换脸 + 爆燃 + 弹幕.
    """
    src = MAIN_PATH.read_text(encoding="utf-8")
    burst_idx = src.find('engine.add_stage("intensity_burst"')
    danmaku_idx = src.find('engine.add_stage("danmaku"')
    assert burst_idx != -1, "main.py 缺 intensity_burst add_stage"
    assert danmaku_idx != -1, "main.py 缺 danmaku add_stage"
    assert burst_idx < danmaku_idx, (
        f"main.py intensity_burst add_stage @ {burst_idx} 必须在 danmaku @ {danmaku_idx} 之前. "
        f"旧顺序 danmaku → burst 让 burst 接力跳过 danmaku 输出, final 缺弹幕. "
        f"如要回退, 必须同步改 34_danmaku fallback 链 / 07_export fallback 链."
    )
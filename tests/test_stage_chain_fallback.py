"""
test_stage_chain_fallback.py — 防回归测试

背景: 2026-07-05 bug fix (commit 84d39a2):
  stages/35 intensity_burst 的 fallback 链漏 mascot_path, 导致 burst 接力
  watermark/danmaku 链绕过 face_swap 结果 → final 输出回退到原脸
  颜青3+4 验证: faceswap-out→selfie cos=0.696 (真换),
                  final→selfie cos=0.185 (=原脸)

钉死原则:
  - face_swap (stages/37) 在 main.py L402 add_stage
  - 之后所有 stage 的 fallback 链必须含 mascot_path
    (因为 face_swap.run() 末尾 ctx.set("mascot_path", out_path)
    下游靠 mascot_path 接脸结果, 见 CLAUDE 2026-06-29 修复)
  - stages/35 burst 特殊: mascot_path 必须在 danmaku_path 之前
    (face_swap 输出优先级最高, danmaku 接力前先吃换脸)
  - 任何 stage 的 fallback 链可以在 mascot_path 之前加 face_swap_path
    (不是必须, mascot_path 已 alias, 但显式更安全)
"""

import ast
from pathlib import Path

import pytest


# face_swap 在 main.py L402, 这些 stage 都跑在它之后
POST_FACE_SWAP_STAGES = {
    "smart_crop": "stages/38_smart_crop.py",
    "danmaku":    "stages/34_danmaku.py",
    "intensity_burst": "stages/35_intensity_burst.py",
    "film_look":  "stages/33_film_look.py",
    "pip":        "stages/31_pip.py",
    "bgm_beat":   "stages/30_bgm_beat.py",
    "qin_cold_open": "stages/36_qin_cold_open.py",
    "export":     "stages/07_export.py",
}


def _extract_input_path_chain(filepath: Path) -> list[str]:
    """从 stage 文件 ast 提取 input_path fallback chain 的 ctx key 列表"""
    src = filepath.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # 通用变量名 (各 stage 用不同变量名, 但都是 ctx.get() 链)
    _INPUT_VARS = {
        "input_path", "processed_path", "src", "main_path",
        "input_video", "target", "input_src",
    }

    ctx_keys = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in _INPUT_VARS):
            if isinstance(node.value, ast.BoolOp) and isinstance(node.value.op, ast.Or):
                for v in node.value.values:
                    if (isinstance(v, ast.Call)
                            and isinstance(v.func, ast.Attribute)
                            and v.func.attr == "get"
                            and len(v.args) >= 1
                            and isinstance(v.args[0], ast.Constant)):
                        ctx_keys.append(v.args[0].value)
    return ctx_keys


@pytest.mark.parametrize("stage_name,rel_path", list(POST_FACE_SWAP_STAGES.items()))
def test_post_face_swap_stage_chain_includes_mascot_path(stage_name, rel_path):
    """face_swap 之后的所有 stage fallback 链必须含 mascot_path (face_swap 接力别名)"""
    src_path = Path(__file__).resolve().parent.parent / rel_path
    chain = _extract_input_path_chain(src_path)

    assert "mascot_path" in chain, (
        f"{stage_name} ({rel_path}) fallback 链缺 mascot_path. "
        f"face_swap (main.py L402) 在这个 stage 之前跑, 完成后会 "
        f"ctx.set('mascot_path', faceswap_output). 下游必须接力, 否则绕过换脸结果. "
        f"\n参考 bug 84d39a2 (颜青3+4 final→selfie cos 跌回 source). "
        f"\n当前链: {chain}"
    )


def test_burst_mascot_before_danmaku():
    """stages/35 intensity_burst mascot_path 必须在 danmaku_path 之前 (commit 84d39a2 钉死)"""
    src_path = Path(__file__).resolve().parent.parent / "stages/35_intensity_burst.py"
    chain = _extract_input_path_chain(src_path)
    assert "mascot_path" in chain
    assert "danmaku_path" in chain
    assert chain.index("mascot_path") < chain.index("danmaku_path"), (
        f"burst 链 mascot_path 必须在 danmaku_path 之前 (face_swap 输出优先级最高). "
        f"\n当前链: {chain}"
    )


def test_face_swap_stage_sets_mascot_path_alias():
    """face_swap run() 内必须 ctx.set('mascot_path', out_path), 下游 fallback 依赖"""
    src_path = Path(__file__).resolve().parent.parent / "stages/37_face_swap.py"
    src = src_path.read_text(encoding="utf-8")
    # 容许两种引号风格
    assert ('ctx.set("mascot_path"' in src
            or "ctx.set('mascot_path'" in src), (
        f"stages/37 face_swap 必须 ctx.set(mascot_path=out_path), "
        f"是下游接力别名. CLAUDE 2026-06-29 钉死 (颜青3+4 bug 84d39a2 同源)."
    )


def test_face_swap_runs_before_export():
    """main.py add_stage 顺序: face_swap < export. 修改 main.py 时必须保持这个顺序, 否则下游接力断"""
    main_path = Path(__file__).resolve().parent.parent / "main.py"
    src = main_path.read_text(encoding="utf-8")
    fs_idx = src.find('engine.add_stage("face_swap"')
    ex_idx = src.find('engine.add_stage("export"')
    assert fs_idx != -1 and ex_idx != -1, "main.py 缺 face_swap 或 export add_stage"
    assert fs_idx < ex_idx, (
        f"main.py 中 face_swap add_stage 必须在 export 之前. 当前 face_swap={fs_idx}, export={ex_idx}. "
        f"修改 main.py 时如发现 export 在前, 必须调整顺序或显式 ctx.set('mascot_path', ...) 接力."
    )

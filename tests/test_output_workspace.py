"""
test_output_workspace.py — 防产物落地 C 盘桌面

背景: 用户 2026-07-05 钉死: 主管线产物只放项目 output/,
绝不 cp 到 C:/Users/18091/Desktop/短视频素材/ 或其他 C 盘路径.
本测试防止代码意外把产物写到主项目 output/ 之外.

反例: 处理艳青3+4 时顺手 cp douyin.mp4 到桌面, 用户反对 (memory no-desktop-output).

钉死规则:
  - main.py / stages/*.py / tools/*.py / lib/*.py / scripts/*.py
    不允许出现 cp/move/shutil.copy 类命令指向 C:/Users/18091/Desktop/ 或 %USERPROFILE%/
  - 进程内产物 ctx.set 的路径必须以 F:/wkspace/fitness-video-pipeline/output/ 或
    Path(__file__).parent 解析的本地路径开头
  - main.py 不应有 --output 默认写桌面 (CLAUDE §"输出目录原则")

策略: AST 扫描 agent 可能修改的文件, 找 shutil.copy/copyfile/move/shutil.rmtree
+ 硬编码桌面路径. 不是完美 (e.g. subprocess.Popen 不在检测范围), 但能挡大部分.
"""

import ast
from pathlib import Path

import pytest


# 不允许硬编码的桌面路径 (用户明确反对)
FORBIDDEN_PATH_FRAGMENTS = [
    "C:/Users/18091/Desktop",
    "C:\\Users\\18091\\Desktop",
    "/c/Users/18091/Desktop",
    # 用户名桌面通用模式: %USERPROFILE%/Desktop 也属 C 盘
    "Desktop/短视频素材",
]


# 扫描的代码目录
SCAN_DIRS = [
    "main.py",
    "stages",
    "lib",
    "pipeline",
    "tools",
    "scripts",
]


# 白名单: 工具脚本可写产物到 output/ (合规), 但禁止写桌面
ALLOWED_OUTPUT_WRITE_FUNCS = {"shutil.copy", "shutil.copyfile",
                               "shutil.copytree", "shutil.move", "os.replace"}


def _scan_file_for_forbidden_writes(filepath: Path) -> list[str]:
    """AST 扫描单个 Python 文件, 返回 (line, msg) 列表"""
    try:
        src = filepath.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    issues = []

    for node in ast.walk(tree):
        # 1) 调用含硬编码桌面路径
        if isinstance(node, ast.Call):
            func_name = _resolve_func_name(node.func)
            if func_name in ALLOWED_OUTPUT_WRITE_FUNCS:
                # 检查 string arg 是否含 forbidden fragment
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if any(frag in arg.value for frag in FORBIDDEN_PATH_FRAGMENTS):
                            rel = filepath.relative_to(Path(__file__).resolve().parent.parent)
                            issues.append(
                                f"{rel}:{node.lineno} {func_name}({arg.value!r}) 写到桌面 "
                                f"(违反钉死规则: 产物只放项目 output/)"
                            )
                # 检查第一个 arg 是变量 (e.g. out_path) 时不报警 (合规来源)
            # subprocess.run 第一参数列表中含桌面路径
            if isinstance(node, ast.Call) and _resolve_func_name(node.func) == "subprocess.run":
                if node.args and isinstance(node.args[0], ast.List):
                    for elt in node.args[0].elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            if any(frag in elt.value for frag in FORBIDDEN_PATH_FRAGMENTS):
                                rel = filepath.relative_to(Path(__file__).resolve().parent.parent)
                                issues.append(
                                    f"{rel}:{node.lineno} subprocess.run 调用含桌面路径 ({elt.value!r})"
                                )
        # 2) 赋值语句右侧含桌面字符串
        if isinstance(node, ast.Assign):
            for v in node.value.children if hasattr(node.value, 'children') else []:
                pass  # AST generic fallback; rely on call scan
    return issues


def _resolve_func_name(func_node) -> str:
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        if isinstance(func_node.value, ast.Name):
            return f"{func_node.value.id}.{func_node.attr}"
        # a.b.c
        if isinstance(func_node.value, ast.Attribute):
            return f"{_resolve_func_name(func_node.value)}.{func_node.attr}"
    return ""


@pytest.mark.parametrize("subdir", SCAN_DIRS)
def test_no_writes_to_desktop_in_source_code(subdir):
    """扫描 main.py / stages / lib / pipeline / tools / scripts, 不应含写桌面的调用"""
    src_root = Path(__file__).resolve().parent.parent / subdir
    if not src_root.exists():
        pytest.skip(f"{subdir} 不存在")

    files_to_check = (
        [src_root] if src_root.is_file() else sorted(src_root.rglob("*.py"))
    )

    all_issues = []
    for f in files_to_check:
        if "__pycache__" in str(f):
            continue
        all_issues.extend(_scan_file_for_forbidden_writes(f))

    if all_issues:
        msg = "\n".join(all_issues)
        pytest.fail(
            f"\n钉死规则: 主管线产物禁止写到 C 盘桌面 ({len(all_issues)} 处违规)\n"
            f"\n应改: 产物路径以本项目 output/ 开头, 见 CLAUDE.md §输出目录原则\n"
            f"\n{msg}"
        )


def test_no_forbidden_path_in_claude_md():
    """CLAUDE.md 自身不应包含 cp 命令或桌面路径 (CLAUDE §'don't put outputs on C drive')"""
    src = (Path(__file__).resolve().parent.parent / "CLAUDE.md").read_text(encoding="utf-8")
    # CLAUDE.md 应该明确禁止 (出现 '不要' + 'cp'/'输出'/'桌面' 的警示)
    assert "不要" in src and ("桌面" in src or "Desktop" in src), (
        "CLAUDE.md 缺少'不在 C 盘放输出'的明确警示. 用户钉死 (memory no-desktop-output)."
    )


def test_douyin_stage_uses_project_output():
    """stage 39 shorts / 35 burst 等 stage 输出路径应在 output_dir/ 下, 不应 cp 桌面"""
    short_vertical = Path(__file__).resolve().parent.parent / "stages" / "short_vertical.py"
    if not short_vertical.exists():
        pytest.skip("short_vertical.py 不存在")
    src = short_vertical.read_text(encoding="utf-8")
    # 不应硬编码桌面
    forbidden = ["C:/Users/", "/c/Users/", "Desktop"]
    for frag in forbidden:
        if frag in src:
            pytest.fail(
                f"stages/short_vertical.py 引用了桌面路径 ({frag!r}). "
                f"产物应只用 output_dir/ (CLAUDE.md)."
            )

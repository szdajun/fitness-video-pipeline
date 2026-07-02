"""测试 bg_swap 工具泛化定稿决策不被回退 (镜像 test_face_swap_no_gfpgan 的源码断言范式).

固化规则 (来自 docs/BG_SWAP.md / memory:bg-swap-tool-influencer / 2026-07-01 泛化定稿):
- matte 默认开 (RVM 高精度抠像治本, 不回退 seg)
- grounding / shadow_strength 内置默认 0 (opt-in, 健身/舞蹈预设才开 0.18)
- ffmpeg 可移植: 不再裸硬编码 C:/Users/.../ffmpeg.exe, 走 _resolve_ffmpeg() 解析链
- 源/背景/教练/输出都是 required CLI 参数, 不绑死某台机器的 Desktop 路径
- _grounding (接地感) / load_bgswap_preset (预设) 函数存在
- 三个预设文件 (fitness 实测 / clean 基线 / dance 起步) 存在且含 bg_swap 段
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
BG_SWAP = ROOT / "tools" / "bg_swap.py"
PRESETS = ROOT / "presets"


def _src():
    return BG_SWAP.read_text(encoding="utf-8")


# ---- 内置默认值 (preset 缺省时的兜底) ----

def test_matte_builtin_default_on():
    """RVM 抠像默认开 (治本), 不应改回 seg 默认"""
    src = _src()
    m = re.search(r'add_argument\(\s*["\']--matte["\'].*?default=preset\.get\(\s*["\']matte["\']\s*,\s*(True|False)\s*\)',
                  src, re.S)
    assert m, "找不到 --matte 的 default=preset.get(...)"
    assert m.group(1) == "True", f"--matte 内置默认应为 True (RVM 治本), 实际: {m.group(1)}"


def test_grounding_builtin_default_zero():
    """接地感默认关 (opt-in); 健身/舞蹈预设才开 0.18"""
    src = _src()
    m = re.search(r'add_argument\(\s*["\']--grounding["\'].*?default=preset\.get\(\s*["\']grounding["\']\s*,\s*([\d.]+)\s*\)',
                  src, re.S)
    assert m, "找不到 --grounding 的 default=preset.get(...)"
    assert float(m.group(1)) == 0.0, f"--grounding 内置默认应为 0.0 (opt-in), 实际: {m.group(1)}"


def test_shadow_builtin_default_zero():
    """硬阴影默认关 (6 轮实测凸显脚地两层, 失败)"""
    src = _src()
    m = re.search(r'add_argument\(\s*["\']--shadow-strength["\'].*?default=preset\.get\(\s*["\']shadow_strength["\']\s*,\s*([\d.]+)\s*\)',
                  src, re.S)
    assert m, "找不到 --shadow-strength 的 default=preset.get(...)"
    assert float(m.group(1)) == 0.0, f"--shadow-strength 内置默认应为 0.0, 实际: {m.group(1)}"


# ---- 可移植性 ----

def test_ffmpeg_portable_resolver():
    """ffmpeg 走 _resolve_ffmpeg() 解析链 (CLI/env/PATH/兜底), 不裸硬编码"""
    src = _src()
    assert "def _resolve_ffmpeg(" in src, "应有 _resolve_ffmpeg() 可移植解析函数"
    assert "FFMPEG = _resolve_ffmpeg()" in src, "模块级 FFMPEG 应由 _resolve_ffmpeg() 赋值"
    # 不应再有裸模块级硬编码 (已知好路径仅作 fallback 留在函数内, 允许)
    assert not re.search(r'^FFMPEG\s*=\s*r["\']C:', src, re.M), \
        "不应有裸模块级 FFMPEG = r'C:/...' 硬编码"


def test_no_source_or_bg_hardcode():
    """源/背景/教练/输出都是 required CLI 参数, 不绑死 Desktop 路径"""
    src = _src()
    for flag in ('"--video"', '"--bg"', '"--coach"', '"--output"'):
        assert flag in src, f"应有 {flag} CLI 参数"
    # argparse 声明这些为 required=True
    for line_flag in ("--video", "--bg", "--coach", "--output"):
        m = re.search(rf'add_argument\(\s*["\']{re.escape(line_flag)}["\'].*?required=True',
                      src, re.S)
        assert m, f"{line_flag} 应 required=True (不绑死路径)"
    # 源码里不该出现绝对桌面素材路径硬编码
    assert "Desktop" not in src, "源码不应硬编码 Desktop 绝对路径"


# ---- 关键函数存在 ----

def test_grounding_function_exists():
    """接地感增强函数 _grounding 必须存在 (C 方案定稿)"""
    assert re.search(r'^def _grounding\(', _src(), re.M), "应有 def _grounding( 接地感函数"


def test_preset_loader_exists():
    """预设加载器 load_bgswap_preset 必须存在"""
    assert re.search(r'^def load_bgswap_preset\(', _src(), re.M), "应有 def load_bgswap_preset("


# ---- core-matte 撑实胳膊 (坑 9, 治 RVM 软抠对胳膊低 alpha 的虚化/渗出) ----

def test_core_bolster_builtin_default_on():
    """core-matte 默认开 (pose 骨架包络撑实 RVM 软抠漏的胳膊 core; 治虚化/渗出主力)"""
    src = _src()
    m = re.search(r'add_argument\(\s*["\']--core-bolster["\'].*?default=preset\.get\(\s*["\']core_bolster["\']\s*,\s*([\d.]+)\s*\)',
                  src, re.S)
    assert m, "找不到 --core-bolster 的 default=preset.get(...)"
    assert float(m.group(1)) == 1.0, f"--core-bolster 内置默认应为 1.0 (开), 实际: {m.group(1)}"


def test_pose_core_matte_function_exists():
    """core+edge matte split 的 pose 骨架包络函数必须存在 (撑实胳膊 core)"""
    assert re.search(r'^def _pose_core_matte\(', _src(), re.M), "应有 def _pose_core_matte("


def test_pink_thresh_passthrough_dest():
    """回归守门: render() 的 pink_sat 应读 args.pink_thresh_sat (正确 dest), 非 args.pink_sat.
    2026-07-02 此 bug (plan B2 passthrough 拼错 dest) 让所有 bg_swap 渲染 AttributeError 崩."""
    src = _src()
    assert "pink_sat=args.pink_thresh_sat" in src, \
        "pink_sat 应读 args.pink_thresh_sat (--pink-thresh-sat 的 dest)"
    assert not re.search(r'pink_sat=args\.pink_sat\b', src), \
        "pink_sat 不能读 args.pink_sat (裸名 dest 错, 会让 render 崩)"


# ---- 预设文件 ----

def test_preset_files_exist():
    """三个预设文件存在: fitness (实测) / clean (基线) / dance (起步)"""
    for name in ("fitness", "clean", "dance"):
        p = PRESETS / f"bgswap_{name}.yaml"
        assert p.exists(), f"预设缺失: {p.name}"


def test_preset_yaml_has_bg_swap_section():
    """每个预设顶层有 bg_swap: 段 (load_bgswap_preset 读这一段)"""
    import yaml
    for name in ("fitness", "clean", "dance"):
        p = PRESETS / f"bgswap_{name}.yaml"
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        assert "bg_swap" in data, f"{p.name} 缺 bg_swap 顶层段"
        assert isinstance(data["bg_swap"], dict), f"{p.name} 的 bg_swap 应是 dict"


def test_fitness_preset_grounding_on():
    """健身预设 (已实测丽丽定稿) 必须开 grounding 0.18"""
    import yaml
    data = yaml.safe_load((PRESETS / "bgswap_fitness.yaml").read_text(encoding="utf-8"))
    cfg = data["bg_swap"]
    assert cfg.get("matte") is True, "fitness 预设应 matte: true"
    assert cfg.get("grounding") == 0.18, f"fitness 预设应 grounding 0.18, 实际 {cfg.get('grounding')}"
    assert cfg.get("shadow_strength") == 0.0, "fitness 预设 shadow_strength 应 0"


def test_clean_preset_all_enhancements_off():
    """clean 基线预设: 所有增强关 (只留 matte 抠像)"""
    import yaml
    cfg = yaml.safe_load((PRESETS / "bgswap_clean.yaml").read_text(encoding="utf-8"))["bg_swap"]
    for k in ("grounding", "parallax", "color_match", "light_wrap", "shadow_strength"):
        assert cfg.get(k) == 0.0, f"clean 预设 {k} 应 0.0, 实际 {cfg.get(k)}"

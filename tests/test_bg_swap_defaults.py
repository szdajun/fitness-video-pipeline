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

def test_core_bolster_builtin_default_off():
    """core-matte 默认关 (2026-07-02 反转: v3 全片实测骨架带每帧硬抬 alpha 让轮廓显脏, 用户
    '基本都这样'; 治渗出价值不抵边缘变脏, 弃用回 v2 软边. 需时手动 --core-bolster 1.0)"""
    src = _src()
    m = re.search(r'add_argument\(\s*["\']--core-bolster["\'].*?default=preset\.get\(\s*["\']core_bolster["\']\s*,\s*([\d.]+)\s*\)',
                  src, re.S)
    assert m, "找不到 --core-bolster 的 default=preset.get(...)"
    assert float(m.group(1)) == 0.0, f"--core-bolster 内置默认应为 0.0 (关), 实际: {m.group(1)}"


def test_pose_core_matte_function_exists():
    """core+edge matte split 的 pose 骨架包络函数必须存在 (撑实胳膊 core)"""
    assert re.search(r'^def _pose_core_matte\(', _src(), re.M), "应有 def _pose_core_matte("


# ---- arm-grow (坑 9.bis, 2026-07-03, 替代 arm-bolster) ----
# arm-bolster 只治核心管 (env scale 1.5), 用户看到的是核心管**外**的过渡环 (scale 1.5→3.0,
# RVM α 0.3-0.7 半透明带, 99.8% 帧有 >2000 渗出像素). D+grow = (a) inner (a>0.15) 填洞治斑驳
# + (b) 在 RVM 自信前景 (a>0.05) 内 grow N×3px 到真实边缘 + (c) max(rvm, smoothed). 模拟
# n=7488: 治愈 99.8% halo 2.5% (grow=1, 3px). 关键: grow 必须用 RVM α 门控, 否则扩到背景
# (A 方案 halo 389%).

def test_arm_grow_builtin_default_off():
    """--arm-grow 默认关 (opt-in). 治 RVM 对胳膊低 alpha 过渡环虚化时手动 --arm-grow 1."""
    src = _src()
    m = re.search(r'add_argument\(\s*["\']--arm-grow["\'].*?default=preset\.get\(\s*["\']arm_grow["\']\s*,\s*(\d+)\s*\)',
                  src, re.S)
    assert m, "找不到 --arm-grow 的 default=preset.get(...)"
    assert int(m.group(1)) == 0, f"--arm-grow 内置默认应为 0 (关), 实际: {m.group(1)}"


def test_pose_arm_core_matte_function_exists():
    """arm pose 包络函数必须存在 (复用为 D+grow 的臂区范围)"""
    assert re.search(r'^def _pose_arm_core_matte\(', _src(), re.M), \
        "应有 def _pose_arm_core_matte( (臂区, blaze 索引)"


def test_arm_core_matte_uses_blaze_indices():
    """_pose_arm_core_matte 必须用 BlazePose-33 索引 (11/12肩 13/14肘 15/16腕),
    非 COCO-17 (5/6/7/8/9/10). detect_pose 缓存格式是 blaze33; 用 COCO 索引读 = 错位画垃圾包络
    (2026-07-03 发现旧版双 bug 之一, 导致 bolster 从没生效)."""
    src = _src()
    m = re.search(r'def _pose_arm_core_matte\(.*?(?=\ndef )', src, re.S)
    assert m, "找不到 _pose_arm_core_matte 函数体"
    body = m.group(0)
    # 臂段连刃必须用 blaze 索引 11→13→15, 12→14→16 (肩→肘→腕)
    assert "(11, 13)" in body and "(13, 15)" in body, \
        "_pose_arm_core_matte 应画 blaze 臂段 (11,13) (13,15) (肩-肘, 肘-腕)"
    assert "(12, 14)" in body and "(14, 16)" in body, "_pose_arm_core_matte 应画右臂 blaze 段"
    # 不能误用 COCO 臂索引 (5,7)(7,9)(6,8)(8,10)
    assert "(5, 7)" not in body and "(7, 9)" not in body, \
        "_pose_arm_core_matte 不能用 COCO 索引 (5,7)(7,9) 读 blaze 数据 (错位)"


def test_arm_core_matte_torso_guard():
    """_pose_arm_core_matte 必须有按人躯干存在门控 (rvm_alpha + sho_thr): 多人源里背景路人
    α≈0 自动跳过, 只撑前景主角. 无门控会把背景路人胳膊也撑实 (clip3 单人无影响但生产多人会脏)."""
    src = _src()
    m = re.search(r'def _pose_arm_core_matte\(.*?(?=\ndef )', src, re.S)
    body = m.group(0)
    assert "rvm_alpha" in body, "_pose_arm_core_matte 应接受 rvm_alpha 参数做躯干门控"
    assert "sho_thr" in body, "_pose_arm_core_matte 应有 sho_thr (躯干 α 阈值) 门控"


def test_arm_grow_uses_fill_holes():
    """arm-grow 核心算法 = binary_fill_holes 治斑驳 + dilate(α>0.05 门控 grow). 不写 fill_holes
    = 治不了 RVM 胳膊内斑驳孔洞 (C 方案 halo 16-39% 治愈 37-59% 就是因为没用 fill_holes).
    binary_fill_holes 仅补内孔, 不外扩 → halo 天然低."""
    src = _src()
    m = re.search(r'if arm_grow > 0 and mask is not None and persons:.*?arm_ok \+= 1',
                  src, re.S)
    assert m, "找不到 arm_grow 应用块 (if arm_grow > 0 ...)"
    block = m.group(0)
    assert "binary_fill_holes" in block, \
        "arm-grow 应有 binary_fill_holes 治胳膊 RVM α 斑驳孔洞 (不写=治不彻底)"
    assert "cv2.dilate" in block, "arm-grow 应有 cv2.dilate grow 到真实边缘"
    # 关键反直觉门控: solid_g & outer (RVM 感到前景的区) 不能缺, 否则扩到背景 (A 灾难)
    assert "& outer" in block or "solid_g = solid_g &" in block.replace(" ", ""), \
        "arm-grow 必须用 (RVM a>0.05) 门控防扩到背景 (否则 A 方案 halo 389%)"
    # in-place 配合 gc 治长视频 RAM
    assert "np.maximum(mask" in block and "out=mask" in block, \
        "arm-grow 应 in-place np.maximum(..., out=mask) 配 render 循环 gc.collect 治长视频 RAM"


def test_arm_grow_default_recommendation_grow_one():
    """--arm-grow 默认 0 关, 推荐值 1 (3px) — 模拟 n=7488: grow=1 halo 2.5% 治愈 99.8% 最优
    (grow=2 halo 3.0% / grow=3 halo 3.2%, 治愈都打平 99.8%). 不能再推荐 1.5 (那是旧
    arm-bolster 的 scale 推荐, 不是 grow 次数)."""
    src = _src()
    assert re.search(r'推荐\s*1', src), \
        "help 应明确推荐 --arm-grow 1 (3px), 非 1.5 (那是旧 arm-bolster 的 scale 推荐)"


def test_mask_mode_intersect_cli():
    """--mask-mode intersect CLI 必须存在, 默认 rvm (不破坏既有调用).
    intersect = RVM α × YOLO-seg person mask, 治 2026-07-03 RVM 远处半透真人 '鬼影' 问题
    (新版 RVM 把远处真人当前景画 = "3 人身后站一个不动的人"). YOLO 强制 CPU 避开 4 模型
    OOM (face-swap-cudnn-fix 三模型已用满 4GB onnx arena)."""
    src = _src()
    m = re.search(r'add_argument\(\s*["\']--mask-mode["\'].*?choices=\[([^\]]+)\].*?default=([\'"]\w+[\'"])',
                  src, re.S)
    assert m, "找不到 --mask-mode choices=... default=..."
    choices = m.group(1)
    assert "'rvm'" in choices and "'intersect'" in choices, \
        f"--mask-mode 应有 'rvm'(默认) + 'intersect'(YOLO 二次确认), 实际 choices: {choices}"
    assert "'rvm'" in m.group(2), f"--mask-mode 默认应是 'rvm' (不破坏既有调用), 实际: {m.group(2)}"


def test_yolo_intersect_render_branch_exists():
    """render() 必须有 mask_mode='intersect' 分支, 调 segment_person(yolo_seg_model, frame)
    拿 person mask 与 RVM α 取交集. 治鬼影实测 1 帧 OK: 3 真人完整, 鬼影消失, 边缘略硬 (RVM α 平滑)."""
    src = _src()
    m = re.search(r'def render\([^)]*\).*?(?=\ndef )', src, re.S)
    assert m, "找不到 render() 函数体"
    body = m.group(0)
    assert "mask_mode" in body, "render() 签名应有 mask_mode 参数"
    assert "intersect" in body, "render() 应有 mask_mode=='intersect' 分支"
    assert "segment_person" in body, "render() intersect 分支应调 segment_person (YOLO)"


def test_yolo_model_forced_cpu():
    """4 模型同进程 (RVM + buffalo_l + inswapper + YOLO) GPU 加载 buffalo_l 报 'bad allocation'
    (实测 4 模型 4GB onnx arena 不够). YOLO-seg 强制 CPU, 避开与 3 GPU 模型争 arena.
    yolov8n-seg 6.7MB CPU 推理 ~50ms/帧 (intersect 仅需 person mask, 不需高精度)."""
    src = _src()
    m = re.search(r'yolo_seg_model = YOLO\(args\.yolo_seg_model\).*?yolo_seg_model\.to\([\'"]cpu[\'"]\)',
                  src, re.S)
    assert m, "YOLO 必须 .to('cpu') 强制 CPU, 否则 4 模型同 GPU 加载 buffalo_l 'bad allocation' OOM"


def test_arm_motion_weight_removed():
    """arm_motion_weight (motion 门控) 已删: 实测静止/快动帧 bolster 收益无差 (臂内部不碰轮廓,
    全帧满抬也不脏), motion weight 反把最需治的快动帧压低. 不能加回."""
    assert not re.search(r'^def arm_motion_weight\(', _src(), re.M), \
        "arm_motion_weight 应已删 (motion 门控无益反害), 不能加回"


def test_arm_grow_imports_binary_fill_holes():
    """binary_fill_holes 来自 scipy.ndimage; arm-grow 用到必须 import (否则 NameError 崩)."""
    src = _src()
    assert re.search(r'^from\s+scipy\.ndimage\s+import\s+binary_fill_holes', src, re.M), \
        "缺少 from scipy.ndimage import binary_fill_holes (arm-grow 用到)"


def test_pink_thresh_passthrough_dest():
    """回归守门: render() 的 pink_sat 应读 args.pink_thresh_sat (正确 dest), 非 args.pink_sat.
    2026-07-02 此 bug (plan B2 passthrough 拼错 dest) 让所有 bg_swap 渲染 AttributeError 崩."""
    src = _src()
    assert "pink_sat=args.pink_thresh_sat" in src, \
        "pink_sat 应读 args.pink_thresh_sat (--pink-thresh-sat 的 dest)"
    assert not re.search(r'pink_sat=args\.pink_sat\b', src), \
        "pink_sat 不能读 args.pink_sat (裸名 dest 错, 会让 render 崩)"


# ---- pose 缓存映射完整性 (arm bolster 的前置依赖) ----

def test_coco2blaze_mapping_has_arm_joints():
    """student_closeup.detect_pose 输出 blaze33 缓存 (bg_swap _pose_arm_core_matte 读它).
    COCO2BLAZE 映射必须含肘(7/8→13/14)+腕(9/10→15/16)+膝(13/14→25/26).
    2026-07-03 发现旧映射漏这些 → 缓存里肘/腕/膝全 vis=0 → arm bolster 抓不到胳膊关节失效.
    不能回退到漏臂的映射."""
    src = (ROOT / "tools" / "student_closeup.py").read_text(encoding="utf-8")
    # COCO2BLAZE 字典必须存在且含臂+膝映射
    m = re.search(r'COCO2BLAZE\s*=\s*\{([^}]*)\}', src, re.S)
    assert m, "student_closeup 应有 COCO2BLAZE 映射字典"
    mapping = m.group(1)
    for pair in ("7: 13", "8: 14", "9: 15", "10: 16", "13: 25", "14: 26"):
        assert pair in mapping, f"COCO2BLAZE 缺臂/膝映射 {pair} (旧版漏这些致 bolster 失效)"


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

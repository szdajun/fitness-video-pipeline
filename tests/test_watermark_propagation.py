"""守门: 汉印/时间戳传播不变量 — 每个 watermark 之后的消费 stage 输入链必须含 watermark_path.

根因 (2026-07-08 张杰1_2 丢汉印/时间戳): 汉印(汉印水印)+时间戳在 **watermark stage (24)** 叠加.
要让它们一路传到 final, 每个 watermark 之后的 stage 输入链**必须含 watermark_path** —— 该 stage
要么直接读 watermark, 要么读一个"自身也含 watermark_path"的更晚 stage (归纳传递). 任何一个
post-watermark stage 漏了 watermark_path, 当更晚 stage 兜底缺席时就会跌穿到 energybar/color 等
watermark 之前的视频 → 丢汉印 → 下游全丢.

历史踩坑 (全同根因, 2026-07-08 一批修):
- 35_intensity_burst 漏 watermark_path (张杰案, 见 test_burst_chain_watermark)
- 38_smart_crop / 36_qin_cold_open / 25_blush / 26_face_beautify / 28_rife 漏 (input 链)
- 26_face_beautify 禁用 passthrough (line 165/176) 漏 → 禁用时 face_beautify_path=energybar 丢汉印
本测试钉死两条不变量, 防未来再加/改 stage 时漂移.

关联 memory: burst-fallback-chain-watermark, stage-order-add-stage-not-filenum.
"""
import re
from pathlib import Path

# 跑在 watermark (main.py:408) 之后的消费 stage (读上游视频产出本 stage 视频).
# 新增 post-watermark stage 记得加到这里 —— 不加就不被守门.
POST_WATERMARK_STAGES = [
    "25_blush.py", "26_face_beautify.py", "27_face_beautify2.py", "28_rife_interpolate.py",
    "29_mascot.py", "30_bgm_beat.py", "31_pip.py", "32_speed_ramp.py", "33_film_look.py",
    "34_danmaku.py", "35_intensity_burst.py", "36_qin_cold_open.py", "37_face_swap.py",
    "38_smart_crop.py", "39_shorts.py", "07_export.py",
]


def _src(name: str) -> str:
    return (Path("stages") / name).read_text(encoding="utf-8")


def test_every_post_watermark_stage_references_watermark_path():
    """核心不变量: 每个 post-watermark 消费 stage 必须引用 watermark_path.

    覆盖两种写法: ctx.get("watermark_path") (链式) 和 "watermark_path" (列表, 如 face_swap 的
    _TARGET_KEYS). 缺了 = 当更晚 stage 缺席时跌穿到 watermark 之前的视频 = 丢汉印/时间戳.
    """
    missing = []
    for name in POST_WATERMARK_STAGES:
        if '"watermark_path"' not in _src(name):
            missing.append(name)
    assert not missing, (
        f"这些 post-watermark stage 缺 watermark_path 引用 (会丢汉印/时间戳): {missing}. "
        f"汉印传播不变量: 每个 watermark 之后的消费 stage 输入链必须含 watermark_path."
    )


def _chains(src: str):
    """提取所有 or-连接的 ctx.get(..._path) 链 (一条链 = 一个连续的 A or B or C)."""
    return re.findall(
        r'ctx\.get\("[a-z_]+_path"\)(?:\s*or\s*ctx\.get\("[a-z_]+_path"\))+', src)


def test_no_chain_falls_to_energybar_before_watermark():
    """更强: 任一含 energybar_path 的 or-链, watermark_path 必须排在第一个 energybar_path 之前.

    否则该链兜底时会先跌到 energybar (无汉印) 而非 watermark. 这正是张杰 burst 案的精确模式
    (energybar 在 watermark 之前). 只检查含 energybar_path 的链 —— 不碰 export 多格式分发的
    原始横源路径 (那是故意用 stabilized/color/warped 不含 energybar, 见 07_export.py:707 注释).
    """
    offenders = []
    for name in POST_WATERMARK_STAGES:
        src = _src(name)
        for chain in _chains(src):
            keys = re.findall(r'ctx\.get\("([a-z_]+_path)"\)', chain)
            if "energybar_path" not in keys:
                continue
            eb = keys.index("energybar_path")
            wm = keys.index("watermark_path") if "watermark_path" in keys else None
            if wm is None or wm > eb:
                offenders.append(f"{name}: 链 {keys} (energybar@{eb}, watermark@{wm})")
    assert not offenders, (
        f"含 energybar_path 的链里 watermark_path 缺失或排在 energybar 之后 (会跌到无汉印视频): {offenders}"
    )

"""守门: burst fallback 链必须含 watermark_path (2026-07-08 张杰1_2 丢汉印/时间戳根因).

根因: face_swap 跳过 (教练无源照) 时 mascot_path/face_swap_path 全 None → burst 跌穿到
energybar_path (watermark 之前) → 爆燃输出无汉印 → 下游 danmaku/export 读 burst → final
丢汉印+时间戳. 加 watermark_path 在 energybar_path 之前, 保证 face_swap 缺席时 burst 也接力
含汉印的视频. 本测试钉死链顺序, 防回退.
"""
import re
from pathlib import Path


def test_burst_chain_has_watermark_before_energybar():
    src = Path("stages/35_intensity_burst.py").read_text(encoding="utf-8")
    # 抓 input_path = (...) 整条链 (含内部 ctx.get(...) 括号, 末尾以 str(ctx.input_path)) 收尾)
    m = re.search(r"input_path\s*=\s*\(([\s\S]*?str\(ctx\.input_path\))\)", src)
    assert m, "未找到 burst input_path fallback 链"
    chain = m.group(1)
    assert "ctx.get(\"watermark_path\")" in chain, "burst 链缺 watermark_path (会丢汉印)"
    # 顺序: watermark_path 必须在 energybar_path 之前 (否则 face_swap 缺席时跌穿丢汉印)
    wm = chain.index("watermark_path")
    eb = chain.index("energybar_path")
    assert wm < eb, "watermark_path 必须在 energybar_path 之前 (否则接力到 watermark 之前的视频)"
    # face_swap/mascot/danmaku 仍在前 (换脸/弹幕结果优先)
    assert chain.index("mascot_path") < wm
    assert chain.index("face_swap_path") < wm
    assert chain.index("danmaku_path") < wm

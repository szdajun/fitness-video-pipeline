"""测试 lib.seal.overlay_seal 真正实现 (非 stub)"""
import sys
import inspect
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_seal_is_not_stub():
    """overlay_seal 必须真正实现, 不能直接返回 frame"""
    from lib.seal import overlay_seal

    src = "def overlay_seal"
    code = inspect.getsource(overlay_seal)
    assert "return frame" not in code, \
        f"overlay_seal 还是 stub (直接返回 frame), 必须真实现"
    assert len(code) > 200, \
        f"overlay_seal 实现过短 ({len(code)} 字符), 可能只是 stub"


def test_seal_modifies_frame():
    """overlay_seal 必须修改 frame (不是返回原图)"""
    import numpy as np
    from lib.seal import overlay_seal

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (200, 200, 200)  # 灰色背景
    result = overlay_seal(frame, text="胭脂虎", pos="top-left", size=130)

    # 必须有像素变化 (左上角区域)
    diff = np.abs(result.astype(int) - frame.astype(int)).sum()
    assert diff > 1000, f"overlay_seal 没修改 frame (diff={diff})"


def test_seal_positions():
    """4 个位置都应该能渲染"""
    import numpy as np
    from lib.seal import overlay_seal

    for pos in ["top-left", "top-right", "bottom-left", "bottom-right"]:
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        result = overlay_seal(frame, text="胭脂虎", pos=pos, size=100, alpha=0.5)
        assert result.shape == frame.shape, f"{pos}: 输出 shape 错"
        assert (result != frame).any(), f"{pos}: 输出和输入相同"


def test_seal_prefers_ai_image():
    """当 tools/seal_ai.png 存在时, 应优先用它 (而不是代码版 PIL 绘制)"""
    import os
    from pathlib import Path
    seal_ai = Path(__file__).parent.parent / "tools" / "seal_ai.png"
    if not seal_ai.exists():
        return  # 跳过, AI 版还没生成
    import numpy as np
    from lib.seal import overlay_seal
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    result = overlay_seal(frame, text="X", pos="top-left", size=100, alpha=0.7)
    # AI 版应该有大量红色像素 (朱砂红)
    red = ((result[:, :, 2] > 150) & (result[:, :, 0] < 80)).sum()
    assert red > 100, f"应优先用 AI seal (红章), 但红色像素过少 ({red})"
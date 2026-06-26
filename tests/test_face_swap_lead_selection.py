"""测试 swap_face ROI 内 lead 选脸策略: 优先 cx 接近 ROI 中心"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.face_swap import swap_face
from unittest.mock import MagicMock


def test_swap_face_with_bbox_prefers_center_face():
    """ROI 内有多个脸时, 选中心脸 (cx 接近 ROI 中心), 不是最大脸"""
    app = MagicMock()
    swapper = MagicMock()
    swapper.get.return_value = "swapped_roi"

    # ROI 中心 (50, 50). 三张脸: 远大脸(0,0-90,90) area=8100,
    # 中心小脸(40,40-60,60) area=400, 边缘脸(80,80-100,100) area=400
    far_face = MagicMock(); far_face.bbox = [0, 0, 90, 90]; far_face.det_score = 0.9
    center_face = MagicMock(); center_face.bbox = [40, 40, 60, 60]; center_face.det_score = 0.7
    edge_face = MagicMock(); edge_face.bbox = [80, 80, 100, 100]; edge_face.det_score = 0.85
    app.get.return_value = [far_face, center_face, edge_face]

    target = MagicMock()
    target.shape = [(100, 100, 3)]
    target.__setitem__ = MagicMock()
    target.copy.return_value = target

    import numpy as np
    target = np.zeros((100, 100, 3), dtype=np.uint8)

    # 调用 swap_face, lead_bbox=(0,0,100,100) 即整个 ROI
    result = swap_face(swapper, MagicMock(), target, app, lead_bbox=(0, 0, 100, 100))

    # 验证 swapper.get 被调用时传入的是 center_face (cx=50, ROI中心)
    call_args = swapper.get.call_args
    face_passed = call_args[0][1]  # 第 2 个位置参数
    assert face_passed is center_face, f"应选中心脸, 实际选了 {face_passed}"


def test_swap_face_falls_back_when_no_face():
    """ROI 内无脸时, swapper 不被调用"""
    app = MagicMock()
    app.get.return_value = []
    swapper = MagicMock()
    import numpy as np
    target = np.zeros((100, 100, 3), dtype=np.uint8)
    result = swap_face(swapper, MagicMock(), target, app, lead_bbox=(0, 0, 100, 100))
    swapper.get.assert_not_called()


if __name__ == "__main__":
    test_swap_face_with_bbox_prefers_center_face()
    test_swap_face_falls_back_when_no_face()
    print("OK")
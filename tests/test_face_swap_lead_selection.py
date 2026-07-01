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

    # 三张脸 (基准 100×100 坐标, center cx=50 = ROI 几何中心):
    # far 大脸(0,0-90,90) area=8100, center 小脸(40,40-60,60) area=400,
    # edge 脸(80,80-100,100) area=400
    far_face = MagicMock(); far_face.det_score = 0.9
    center_face = MagicMock(); center_face.det_score = 0.7
    edge_face = MagicMock(); edge_face.det_score = 0.85
    base_bbox = {far_face: [0, 0, 90, 90],
                 center_face: [40, 40, 60, 60],
                 edge_face: [80, 80, 100, 100]}

    # swap_face lead_bbox 分支会把小 ROI (~100px) 上采样到 512 再检测 (2026-06-29
    # 远景小脸修复, 不可动). 真实检测器返回的 bbox 在【传入 roi 的坐标系】里, 故 mock
    # 须按 roi 尺寸缩放基准 bbox, 使 center 脸始终落在 roi 几何中心 (cx=rw/2),
    # 无论是否 upscale. (旧 mock 给固定 100×100 bbox, upscale 后 roi 中心 50→256,
    # 距离错位反选 edge_face.)
    def fake_get(roi, *a, **kw):
        rh, rw = roi.shape[:2]
        sx, sy = rw / 100.0, rh / 100.0
        for f, bb in base_bbox.items():
            f.bbox = [bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy]
        return list(base_bbox.keys())
    app.get.side_effect = fake_get

    import numpy as np
    target = np.zeros((100, 100, 3), dtype=np.uint8)

    # 调用 swap_face, lead_bbox=(0,0,100,100) 即整个 ROI
    result = swap_face(swapper, MagicMock(), target, app, lead_bbox=(0, 0, 100, 100))

    # 验证 swapper.get 被调用时传入的是 center_face (cx=ROI 中心, dist 最小)
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
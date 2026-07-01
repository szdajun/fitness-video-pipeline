"""lib/warp.py 体型变形引擎的纯函数不变量测试 (2026-07-02 债 C).

主管线 body_warp (stages/05) 的核心数学在 lib/warp.py, 之前无单测.
这里测不变量 (无需视频/GPU, 合成关键点):
- identity: 所有参数=1.0 → 位移图恒等于原网格 (默认不变形)
- shape: 输出位移图形状 == (img_h, img_w)
- waist_slim<1 → 腰部采样向身体中心收窄 (方向正确, 不反向)
- create_body_mask: vis<4 关键点 → 回退全 1 mask"""
import numpy as np

from lib.warp import create_displacement_map, create_body_mask

H, W = 480, 270  # 小尺寸竖屏, 快速


def _standing_kps():
    """合成 COCO-17 居中站立人 (坐标归一化 0-1, conf=1)"""
    kps = np.zeros((17, 3))
    kps[:, 2] = 1.0
    kps[0] = [0.50, 0.10, 1.0]    # nose
    kps[5] = [0.40, 0.20, 1.0]    # left shoulder
    kps[6] = [0.60, 0.20, 1.0]    # right shoulder
    kps[11] = [0.42, 0.55, 1.0]   # left hip
    kps[12] = [0.58, 0.55, 1.0]   # right hip
    kps[13] = [0.43, 0.75, 1.0]   # left knee
    kps[14] = [0.57, 0.75, 1.0]   # right knee
    kps[15] = [0.44, 0.95, 1.0]   # left ankle
    kps[16] = [0.56, 0.95, 1.0]   # right ankle
    return kps


_IDENTITY_CFG = {
    "leg_lengthen": 1.0, "waist_slim": 1.0, "overall_slim": 1.0,
    "leg_slim": 1.0, "chest_enlarge": 1.0, "neck_lengthen": 1.0,
}


def test_identity_warp_is_noop():
    """所有参数=1.0 时位移图恒等于原网格 (默认不变形)"""
    mx, my = create_displacement_map(_standing_kps().tolist(), H, W, _IDENTITY_CFG)
    yy, xx = np.mgrid[0:H, 0:W]
    assert np.allclose(mx, xx, atol=1e-3), "identity 参数下 map_x 应等于原 x 网格"
    assert np.allclose(my, yy, atol=1e-3), "identity 参数下 map_y 应等于原 y 网格"


def test_output_shape():
    """位移图形状必须 == (img_h, img_w)"""
    mx, my = create_displacement_map(_standing_kps().tolist(), H, W, _IDENTITY_CFG)
    assert mx.shape == (H, W)
    assert my.shape == (H, W)
    assert mx.dtype == np.float32 and my.dtype == np.float32


def test_waist_slim_narrows_toward_center():
    """waist_slim=0.8 → 腰部采样点向身体中心收窄 (方向不变反)"""
    cfg = {"waist_slim": 0.8}
    mx, _ = create_displacement_map(_standing_kps().tolist(), H, W, cfg)
    body_cx = 0.5 * W  # 居中人, 中心 x
    # 腰中心 y ≈ shoulder_y*0.35 + hip_y*0.65 = 96*0.35+264*0.65 ≈ 205
    y, x = 205, 160    # 离中心 25px, 在 body_half 内 (combined>0)
    d_before = abs(x - body_cx)
    d_after = abs(float(mx[y, x]) - body_cx)
    assert d_after < d_before, \
        f"waist_slim<1 应收窄 (d_after={d_after:.2f} 应 < d_before={d_before:.2f})"


def test_body_mask_shape_and_range():
    """身体 mask 形状对, 值域 [0,1]"""
    mask = create_body_mask(_standing_kps(), H, W)
    assert mask.shape == (H, W)
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_body_mask_low_visibility_fallback():
    """可见关键点 <4 → 回退全 1 mask (避免空凸包)"""
    kps = _standing_kps()
    kps[:, 2] = 0.1  # 全部 conf <0.3 → vis.sum()==0
    mask = create_body_mask(kps, H, W)
    assert mask.shape == (H, W)
    assert np.all(mask == 1.0), "vis<4 应回退全 1 mask"

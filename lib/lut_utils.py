"""3D LUT 工具: .cube 加载 + 三线性插值 + 内置调色预设"""

import numpy as np
from pathlib import Path


def load_cube(path: str):
    """解析 Adobe .cube 3D LUT 文件

    Returns:
        (lut_data, size) — lut_data 形状 (size, size, size, 3), float 0~1
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    size = None
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("TITLE"):
            continue
        if line.upper().startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if line.upper().startswith("DOMAIN_"):
            continue
        parts = line.split()
        if len(parts) == 3:
            data_lines.append([float(p) for p in parts])

    if size is None:
        raise ValueError("无效 .cube: 缺少 LUT_3D_SIZE")
    if len(data_lines) != size ** 3:
        raise ValueError(
            f".cube 数据行数({len(data_lines)}) ≠ {size}^3 = {size**3}"
        )

    lut = np.array(data_lines, dtype=np.float32).reshape(size, size, size, 3)
    return lut, size


def apply_lut(bgr: np.ndarray, lut: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """对 BGR 帧应用 3D LUT（三线性插值）

    Args:
        bgr: 输入帧 (H, W, 3), uint8
        lut: 3D LUT (size, size, size, 3), float 0~1
        intensity: 混合度 0~1, 1=完全 LUT, 0=原图

    Returns:
        处理后的 BGR 帧, uint8
    """
    size = lut.shape[0]
    # BGR → RGB, normalize to 0~1
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0

    # 坐标映射到 [0, size-1]
    idx = rgb * (size - 1)
    i0 = np.clip(np.floor(idx).astype(np.int32), 0, size - 2)
    i1 = i0 + 1
    frac = idx - i0.astype(np.float32)

    # 三线性插值 (vectorized)
    # 8 个格点
    c000 = lut[i0[..., 0], i0[..., 1], i0[..., 2]]
    c001 = lut[i0[..., 0], i0[..., 1], i1[..., 2]]
    c010 = lut[i0[..., 0], i1[..., 1], i0[..., 2]]
    c011 = lut[i0[..., 0], i1[..., 1], i1[..., 2]]
    c100 = lut[i1[..., 0], i0[..., 1], i0[..., 2]]
    c101 = lut[i1[..., 0], i0[..., 1], i1[..., 2]]
    c110 = lut[i1[..., 0], i1[..., 1], i0[..., 2]]
    c111 = lut[i1[..., 0], i1[..., 1], i1[..., 2]]

    f = frac  # (H, W, 3)
    # 沿 R 轴插值
    c00 = c000 * (1 - f[..., 0:1]) + c100 * f[..., 0:1]
    c01 = c001 * (1 - f[..., 0:1]) + c101 * f[..., 0:1]
    c10 = c010 * (1 - f[..., 0:1]) + c110 * f[..., 0:1]
    c11 = c011 * (1 - f[..., 0:1]) + c111 * f[..., 0:1]
    # 沿 G 轴插值
    c0 = c00 * (1 - f[..., 1:2]) + c10 * f[..., 1:2]
    c1 = c01 * (1 - f[..., 1:2]) + c11 * f[..., 1:2]
    # 沿 B 轴插值
    result = c0 * (1 - f[..., 2:3]) + c1 * f[..., 2:3]

    # 混合
    if intensity < 1.0:
        result = result * intensity + rgb * (1.0 - intensity)

    # RGB → BGR, back to uint8
    result = np.nan_to_num(result, nan=0.0)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result[..., ::-1]


def _gen_builtin_lut(name: str, size: int = 33):
    """程序化生成内置 LUT 数据

    Returns:
        (lut_data, size)
    """
    axis = np.linspace(0, 1, size, dtype=np.float32)
    r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
    lut = np.stack([r, g, b], axis=-1)

    if name == "cinematic":
        # S-curve contrast + teal shadows / warm highlights (橙青调)
        # 对比度 S 曲线
        lut = lut ** 0.85  # gamma-like
        # 阴影偏青 (减 R, 加 B)
        shadows = 1.0 - lut  # 阴影权重 (亮部=0, 暗部=1)
        shadows_strength = shadows ** 1.5
        lut[..., 0] -= shadows_strength[..., 0] * 0.08  # 阴影减红
        lut[..., 2] += shadows_strength[..., 2] * 0.06  # 阴影加蓝
        lut = np.clip(lut, 0, 1)
        # 高光偏暖 (加 R, 减 B)
        highlights = lut
        hl_strength = np.clip(highlights, 0, 1) ** 1.5
        lut[..., 0] += hl_strength[..., 0] * 0.04
        lut[..., 2] -= hl_strength[..., 2] * 0.03
        lut = np.clip(lut, 0, 1)
        # 饱和度微增
        gray = lut.mean(axis=-1, keepdims=True)
        lut = gray + (lut - gray) * 1.08

    elif name == "warm":
        # 暖色调 (golden hour)
        lut[..., 0] *= 1.06  # +R
        lut[..., 1] *= 1.02  # +G
        lut[..., 2] *= 0.92  # -B

    elif name == "cool":
        # 冷色调 (blue steel)
        lut[..., 0] *= 0.94
        lut[..., 1] *= 0.98
        lut[..., 2] *= 1.06

    elif name == "vintage":
        # 胶片褪色 (faded film)
        lut = lut * 0.85 + 0.08  # 降低对比 + 灰雾
        lut[..., 0] *= 1.03  # 偏暖
        lut[..., 2] *= 0.95

    elif name == "fuji":
        # Fujifilm Provia 模拟
        gray = lut.mean(axis=-1, keepdims=True)
        lut = gray + (lut - gray) * 1.1  # 微增饱和度
        lut[..., 2] += lut[..., 2] * 0.03  # 蓝通道微增
        lut = np.clip(lut, 0, 1)

    elif name == "bleach":
        # 漂白 bypass (高对比 + 低饱和)
        gray = lut.mean(axis=-1, keepdims=True)
        lut = gray + (lut - gray) * 0.6  # 降饱和
        lut = np.clip(lut ** 0.8, 0, 1)  # 增对比

    else:
        raise ValueError(f"未知内置 LUT: {name}")

    lut = np.clip(lut, 0, 1)
    return lut, size


_BUILTIN_CACHE = {}


def write_lut_cube(lut_data, intensity=1.0, size=33, path=None):
    """将 LUT 数据写入 Adobe .cube 文件（用于 FFmpeg lut3d 加速）

    Args:
        lut_data: (size, size, size, 3) float32 0~1
        intensity: 混合度, 1=完全LUT, 0=单位LUT
        size: LUT 网格大小
        path: 输出路径, None 则返回字符串

    Returns:
        写入的路径或字符串内容
    """
    # 混合 intensity（线性插值 identity ↔ lut）
    if intensity < 1.0:
        axis = np.linspace(0, 1, size, dtype=np.float32)
        r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
        identity = np.stack([r, g, b], axis=-1)
        lut_data = identity * (1.0 - intensity) + lut_data * intensity

    lines = [
        f"TITLE \"Generated LUT\"\n",
        f"LUT_3D_SIZE {size}\n",
        f"DOMAIN_MIN 0.0 0.0 0.0\n",
        f"DOMAIN_MAX 1.0 1.0 1.0\n",
    ]
    for ri in range(size):
        for gi in range(size):
            for bi in range(size):
                val = lut_data[ri, gi, bi]
                lines.append(f"{val[0]:.6f} {val[1]:.6f} {val[2]:.6f}\n")
    content = "".join(lines)

    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    return content


def get_builtin_lut(name: str):
    """获取内置调色预设

    支持: cinematic, warm, cool, vintage, fuji, bleach
    """
    if name not in _BUILTIN_CACHE:
        _BUILTIN_CACHE[name] = _gen_builtin_lut(name)
    return _BUILTIN_CACHE[name]

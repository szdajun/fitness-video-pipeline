"""源视频方向检测 (2026-07-10 竖屏源端到端通路)

目的: 抽 ffprobe EXIF/displaymatrix + cv2 像素兜底逻辑成纯函数, 让
- `main.py` 自动检测入口
- `stages/00a_normalize_orientation.py` 转码前判断
共用同一份检测代码, 避免重复。

Returns (detect_source_orientation):
    {
        "src_w": 1920,           # ffprobe 报的 width (raw)
        "src_h": 1080,           # ffprobe 报的 height (raw)
        "rotation": 90,          # 0/90/180/270, 0 = no rotation
        "is_vertical": True,     # 实际播放方向是 9:16 (考虑了 rotate/displaymatrix)
        "needs_normalize": True, # 需要转码锁像素 (有 rotate 或需要转置)
    }
"""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

# 复用 short_vertical.py 的 ffmpeg 候选路径思路镜像写 ffprobe
_FFPROBE_CANDIDATES = [
    r"C:\Users\18091\ffmpeg\ffprobe.exe",
    r"C:\Users\18091\ffmpeg\ffmpeg.exe",  # ffmpeg 也能 -version 测, 但 ffprobe 独立 exe
    "ffprobe",
]


def _resolve_ffprobe() -> str:
    """找一个可用的 ffprobe.exe. 失败 fallback 'ffprobe' (PATH)."""
    for p in _FFPROBE_CANDIDATES:
        try:
            r = subprocess.run([p, "-version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return p
        except Exception:
            continue
    return "ffprobe"


FFPROBE = _resolve_ffprobe()


def _probe_ffprobe_json(path: str) -> dict:
    """ffprobe -show_streams JSON, 失败返 {}."""
    try:
        r = subprocess.run(
            [
                FFPROBE, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height:stream_tags=rotate:stream_side_data=side_data_type,rotation",
                "-of", "json",
                str(path),
            ],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except Exception:
        pass
    return {}


def _normalize_rotation(v) -> int:
    """ffprobe rotate 可能是 90/-90/90.0, 统一到 0/90/180/270."""
    try:
        n = int(round(float(v))) % 360
        if n < 0:
            n += 360
        # -90 → 270 (displaymatrix 常见)
        if n == -90 % 360:
            n = 270
        return n
    except Exception:
        return 0


def _cv2_first_frame_size(path: str) -> Tuple[int, int]:
    """cv2.VideoCapture 首帧 shape 兜底."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return w, h
    except Exception:
        pass
    return 0, 0


def detect_source_orientation(path: str) -> dict:
    """检测源视频方向 (优先 ffprobe, 兜底 cv2).

    Returns:
        {
            "src_w": int,            # ffprobe 报的 width
            "src_h": int,            # ffprobe 报的 height
            "rotation": int,         # 0/90/180/270
            "is_vertical": bool,     # 实际播放方向是 9:16
            "needs_normalize": bool, # 需要转码锁像素
        }
    """
    width = height = 0
    rotation = 0

    data = _probe_ffprobe_json(path)
    streams = data.get("streams") or []
    if streams:
        st = streams[0]
        width = int(st.get("width") or 0)
        height = int(st.get("height") or 0)
        tags = st.get("tags") or {}
        rotation = _normalize_rotation(tags.get("rotate", 0))
        if not rotation:
            for sd in st.get("side_data_list") or []:
                if "rotation" in sd:
                    rotation = _normalize_rotation(sd.get("rotation"))
                    break

    # cv2 兜底
    if not width or not height:
        width, height = _cv2_first_frame_size(path)

    # 算 effective 方向 (考虑 rotate)
    if rotation in (90, 270):
        eff_w, eff_h = height, width
    else:
        eff_w, eff_h = width, height

    ratio = (eff_w / eff_h) if eff_h else 0
    is_vertical = bool(eff_h > eff_w and 0.50 <= ratio <= 0.65)
    needs_normalize = bool(rotation in (90, 180, 270))

    return {
        "src_w": width,
        "src_h": height,
        "rotation": rotation,
        "is_vertical": is_vertical,
        "needs_normalize": needs_normalize,
    }


def is_vertical_video(path: str) -> bool:
    """便利函数: detect_source_orientation(...)['is_vertical']."""
    try:
        info = detect_source_orientation(path)
        return bool(info.get("is_vertical"))
    except Exception:
        return False


def apply_transpose_filter(rotation: int) -> str:
    """rotation → ffmpeg -vf transpose filter 字符串.

    2026-07-10: **不再使用**! 教训=铁娘子3.mp4 实测, 加 `-noautorotate` + 手动
    transpose 会导致"双重旋转"=颠倒. 正确做法是**不加 noautorotate**, 让
    ffmpeg 默认自动应用 EXIF rotate (抽帧就是已正确的方向), 我们只做
    scale + `-metadata:s:v:0 rotate=0` 输出重置元数据.

    此函数保留仅为向后兼容 (测试 + 外部调用), 实际 normalize_orientation 不再调用.

    90  → 'transpose=1'        (90° CW)
    180 → 'transpose=1,transpose=1'
    270 → 'transpose=2'        (90° CCW)
    0   → ''                   (no-op)
    """
    if rotation == 90:
        return "transpose=1"
    if rotation == 270:
        return "transpose=2"
    if rotation == 180:
        return "transpose=1,transpose=1"
    return ""


def _is_already_vertical(src_w: int, src_h: int, rotation: int) -> bool:
    """像素已是 9:16 方向且不需要转置."""
    if rotation in (90, 270):
        # raw 是横屏但转 90/270 后是竖屏 → 像素没 baked, 需要 normalize
        return False
    return src_h > src_w
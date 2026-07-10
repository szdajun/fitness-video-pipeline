"""test_normalize_orientation.py (2026-07-10 竖屏源端到端通路)

守门:
- NormalizeOrientationStage 增量跳过 (normalized_path 已存在 → 复用)
- 横屏源 → 不调 ffmpeg (不需 normalize)
- 已是 9:16 像素 + rotation=0 → 不调 ffmpeg (passthrough)
- EXIF 旋转的源 → ctx.normalized_path 设到 _normalized/{stem}_normalized.mp4
"""
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

no_mod = importlib.import_module("stages.00_normalize_orientation")
NormalizeOrientationStage = no_mod.NormalizeOrientationStage


def _find_set(ctx, key, value):
    """找 ctx.set(key, value) 调用, 返 True/False."""
    for call in ctx.set.call_args_list:
        if call.args[0] == key and call.args[1] == value:
            return True
    return False


# ── apply_transpose_filter (re-test) ────────────────────────────────

def test_apply_transpose_filter_90():
    # 2026-07-10: apply_transpose_filter 保留向后兼容, 但实际不再使用 (ffmpeg 自动处理 rotate)
    assert no_mod.apply_transpose_filter(90) == "transpose=1"


def test_apply_transpose_filter_180():
    assert no_mod.apply_transpose_filter(180) == "transpose=1,transpose=1"


def test_apply_transpose_filter_270():
    assert no_mod.apply_transpose_filter(270) == "transpose=2"


def test_apply_transpose_filter_0():
    assert no_mod.apply_transpose_filter(0) == ""


# ── NormalizeOrientationStage 增量跳过 ─────────────────────────────

def test_stage_skip_when_normalized_path_exists(tmp_path):
    """ctx.normalized_path 已存在 → 直接复用, 不再 detect."""
    fake_normalized = tmp_path / "test_normalized.mp4"
    fake_normalized.write_text("fake")

    ctx = MagicMock()
    ctx.get.return_value = str(fake_normalized)
    ctx.input_path = str(tmp_path / "source.mp4")

    NormalizeOrientationStage().run(ctx)

    assert str(ctx.input_path) == str(fake_normalized)


# ── NormalizeOrientationStage 横屏源 passthrough ──────────────────

def test_stage_passthrough_horizontal_source(tmp_path):
    """横屏源 (1920x1080 + rotation=0) → 不需 normalize, ctx.normalized_path = src."""
    real_src = tmp_path / "horizontal.mp4"
    real_src.write_text("fake")

    with patch.object(no_mod, "detect_source_orientation") as mock_detect, \
         patch.object(no_mod, "path_exists", return_value=False):
        mock_detect.return_value = {
            "src_w": 1920, "src_h": 1080, "rotation": 0,
            "is_vertical": False, "needs_normalize": False,
        }
        ctx = MagicMock()
        # 用真实存在路径 (Path.exists() 不会 return False 守卫提早)
        ctx.input_path = real_src
        ctx.get.return_value = None

        NormalizeOrientationStage().run(ctx)

        expected = str(real_src)
        assert _find_set(ctx, "normalized_path", expected), \
            f"expected normalized_path={expected!r} in {ctx.set.call_args_list}"


# ── NormalizeOrientationStage 竖源 passthrough ─────────────────────

def test_stage_passthrough_native_vertical(tmp_path):
    """已是 9:16 像素 + rotation=0 → 不需 normalize."""
    real_src = tmp_path / "native_vertical.mp4"
    real_src.write_text("fake")

    with patch.object(no_mod, "detect_source_orientation") as mock_detect, \
         patch.object(no_mod, "path_exists", return_value=False):
        mock_detect.return_value = {
            "src_w": 1080, "src_h": 1920, "rotation": 0,
            "is_vertical": True, "needs_normalize": False,
        }
        ctx = MagicMock()
        ctx.input_path = real_src
        ctx.get.return_value = None

        NormalizeOrientationStage().run(ctx)

        expected = str(real_src)
        assert _find_set(ctx, "normalized_path", expected)


# ── NormalizeOrientationStage EXIF 旋转源 — 调 ffmpeg ──────────────

def test_stage_runs_ffmpeg_for_exif_rotation(tmp_path):
    """EXIF 旋转源 (1920x1080 + rotation=270) → 调 ffmpeg, normalized_path + 改 ctx.input_path.

    2026-07-10 教训: 不加 -noautorotate, 让 ffmpeg 自动处理 EXIF rotate;
    我们只做 scale=1080:1920 + -metadata rotate=0 重置元数据.
    """
    src = tmp_path / "exif_270.mp4"
    src.write_text("fake")

    with patch.object(no_mod, "detect_source_orientation") as mock_detect, \
         patch.object(no_mod, "path_exists", return_value=False), \
         patch.object(no_mod.subprocess, "run") as mock_run:
        mock_detect.return_value = {
            "src_w": 1920, "src_h": 1080, "rotation": 270,
            "is_vertical": True, "needs_normalize": True,
        }
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        ctx = MagicMock()
        ctx.input_path = str(src)
        ctx.get.return_value = None

        NormalizeOrientationStage().run(ctx)

        assert mock_run.called
        cmd = mock_run.call_args.args[0]
        # 2026-07-10: 不再调 transpose, 只 scale + 重置 rotate metadata
        vf = cmd[cmd.index("-vf") + 1]
        assert "scale=1080:1920" in vf
        assert "transpose" not in vf  # 关键: 没有手动 transpose (否则颠倒)
        # -metadata:s:v:0 rotate=0 重置元数据
        assert "rotate=0" in " ".join(cmd)
        # 不能有 -noautorotate (否则 ffmpeg 不自动旋转 + 不手动 rotate = 颠倒)
        assert "-noautorotate" not in cmd

        assert str(ctx.input_path).endswith("exif_270_normalized.mp4")
        assert "_normalized" in str(ctx.input_path)
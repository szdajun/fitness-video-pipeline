"""守门: tools/face_swap.py 的 ffmpeg 走 lib.utils.resolve_ffmpeg() 可移植解析,
不裸硬编码本机绝对路径 (2026-07-02 债1). 镜像 test_bg_swap_defaults::test_ffmpeg_portable_resolver."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
FS = ROOT / "tools" / "face_swap.py"


def _src():
    return FS.read_text(encoding="utf-8")


def test_face_swap_uses_resolve_ffmpeg():
    """face_swap 不裸硬编码 FFMPEG, 走 lib.utils.resolve_ffmpeg()"""
    src = _src()
    assert "from lib.utils import resolve_ffmpeg" in src, \
        "应 from lib.utils import resolve_ffmpeg"
    assert "FFMPEG = resolve_ffmpeg()" in src, \
        "模块级 FFMPEG 应由 resolve_ffmpeg() 赋值"
    # 不应再有裸模块级硬编码
    assert not re.search(r'^FFMPEG\s*=\s*r?["\']C:', src, re.M), \
        "不应有裸模块级 FFMPEG = 'C:/...' 硬编码"


def test_resolve_ffmpeg_in_lib_utils():
    """lib/utils.py 提供 resolve_ffmpeg 共享函数"""
    utils_src = (ROOT / "lib" / "utils.py").read_text(encoding="utf-8")
    assert "def resolve_ffmpeg(" in utils_src, "lib/utils.py 应有 resolve_ffmpeg()"


def test_resolve_ffmpeg_known_good_priority():
    """已知好路径优先于 PATH (Winget 版有 bug) — 逻辑写在 resolve_ffmpeg 内"""
    utils_src = (ROOT / "lib" / "utils.py").read_text(encoding="utf-8")
    assert "_KNOWN_GOOD_FFMPEG" in utils_src, "应有已知好路径常量"
    # 顺序: 已知好路径 isfile 检查在 shutil.which 之前
    m_known = utils_src.find("os.path.isfile(_KNOWN_GOOD_FFMPEG)")
    m_which = utils_src.find("shutil.which")
    assert 0 < m_known < m_which, "已知好路径应在 shutil.which(PATH) 之前 (优先级)"

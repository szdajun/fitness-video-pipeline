# -*- coding: utf-8 -*-
"""项目级 pytest conftest — 注入 ComfyUI custom_nodes 路径让测试能 import youtube_upload."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_COMFY = _ROOT.parent / "ComfyUI" / "custom_nodes"
if _COMFY.exists() and str(_COMFY) not in sys.path:
    sys.path.insert(0, str(_COMFY))
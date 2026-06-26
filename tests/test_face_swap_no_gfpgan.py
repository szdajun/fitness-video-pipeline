"""测试 face_swap 默认关 GFPGAN — 防止被改回

固化规则（来自 CLAUDE.md / 2026-06-26 决定）:
- face_swap.py 默认 gfpgan_strength=0 (完全关闭)
- 原因: 既然换了脸, 美颜修的就是假脸, 没意义; 且 CPU 跑 7h/视频
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FACE_SWAP_STAGE = Path(__file__).parent.parent / "stages" / "37_face_swap.py"


def test_gfpgan_default_is_zero():
    """face_swap stage 必须默认 gfpgan_strength=0"""
    src = FACE_SWAP_STAGE.read_text(encoding="utf-8")
    # 找 fs_cfg.get("gfpgan_strength", X) 的 X
    import re
    m = re.search(r'fs_cfg\.get\(\s*["\']gfpgan_strength["\']\s*,\s*([^)]+)\)', src)
    assert m, "找不到 gfpgan_strength 默认值"
    default = m.group(1).strip()
    assert default == "0", f"gfpgan_strength 默认值应为 0, 实际: {default}"


def test_no_gfpgan_import_or_call_when_disabled():
    """源码不应强制加载 GFPGAN 模型 (节省内存+启动时间)"""
    tools_src = (Path(__file__).parent.parent / "tools" / "face_swap.py").read_text(encoding="utf-8")
    # _load_gfpgan() 调用前应有 gfpgan_strength > 0 判断
    assert 'gfpgan_strength and gfpgan_strength > 0' in tools_src, \
        "tools/face_swap.py 应在调用 _load_gfpgan() 前判断 strength > 0, 否则 GFPGAN 模型被强制加载"


def test_face_swap_preserves_skip_when_no_face():
    """face_swap 检测不到脸时应整体跳过 (不影响下游)"""
    src = FACE_SWAP_STAGE.read_text(encoding="utf-8")
    tools_src = (Path(__file__).parent.parent / "tools" / "face_swap.py").read_text(encoding="utf-8")
    # 2026-06-26: 重构后, 主循环用 if not faces_before / skip_no_pose 跳过
    assert ("if not faces_before" in tools_src or "skip_no_pose" in tools_src
            or "if not lead_person" in tools_src), \
        "tools/face_swap.py 应有 'no face / no lead' 守卫, 无脸时整体跳过"

    # stage 应只在检测到脸时才设置 mascot_path
    assert "if not source_face:" in src or "找不到就 skip" in src, \
        "stage 应在找不到教练照片时 skip"


def test_face_swap_supports_pose_bbox():
    """swap_face 应支持 lead_bbox 参数, 用 pose 定位领操人"""
    tools_src = (Path(__file__).parent.parent / "tools" / "face_swap.py").read_text(encoding="utf-8")
    assert "lead_bbox" in tools_src, \
        "swap_face 必须支持 lead_bbox 参数, 用于 pose-driven lead 定位"
    assert "get_lead_bbox_from_pose" in tools_src, \
        "应有 get_lead_bbox_from_pose() 从 pose 关键点算 bbox"
    assert "find_lead_person" in tools_src, \
        "应有 find_lead_person() 从多人中选领操人"


def test_face_swap_skips_back_orientation():
    """背面朝镜头时应跳过换脸"""
    tools_src = (Path(__file__).parent.parent / "tools" / "face_swap.py").read_text(encoding="utf-8")
    assert "skip_back" in tools_src, \
        "process_video 主循环应有 skip_back 计数器"
    assert 'orientation == "back"' in tools_src, \
        "背面朝向时跳过换脸"


def test_stage_passes_keypoints_to_process_video():
    """stage 37 应把 keypoints_file 传给 process_video"""
    stage_src = FACE_SWAP_STAGE.read_text(encoding="utf-8")
    assert "keypoints_file" in stage_src, \
        "stages/37_face_swap.py 应把 keypoints_file 传给 process_video"
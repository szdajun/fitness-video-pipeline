"""Manifest 增量恢复系统

每个输入视频对应一个 manifest 文件，记录：
- input 指纹（path, mtime, size）
- config hash
- cache version
- 每个 stage 的运行结果和产出路径

恢复时校验 manifest 是否与当前运行兼容，避免误复用旧缓存。
"""

import json, hashlib, os
from pathlib import Path
from typing import Dict, Any, Optional

CACHE_VERSION = 1


def get_manifest_path(ctx) -> Path:
    """获取 manifest 文件路径"""
    return ctx.output_dir / f"{ctx.input_path.stem}_manifest.json"


def build_input_fingerprint(input_path: Path) -> Dict[str, Any]:
    """构建输入文件的指纹信息"""
    stat = input_path.stat()
    return {
        "path": str(input_path.resolve()),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def compute_config_hash(config: dict) -> str:
    """计算配置的 hash，用于判断配置变化时缓存是否有效"""
    # 只对关键参数取 hash，忽略临时字段
    relevant_keys = [
        "stages", "body_warp", "color_grade", "ken_burns", "face_beautify",
        "face_beautify2", "skin_smooth", "energy_bar", "intro_outro",
        "output", "preview", "full_video"
    ]
    hash_data = {}
    for key in relevant_keys:
        if key in config:
            hash_data[key] = config[key]

    # 加入 preset 名称（如果存在）
    if config.get("_preset_name"):
        hash_data["_preset_name"] = config["_preset_name"]

    content = json.dumps(hash_data, sort_keys=True, default=str)
    return hashlib.md5(content.encode()).hexdigest()


def load_manifest(ctx) -> Optional[Dict]:
    """加载 manifest 文件，如果不存在或无法解析返回 None"""
    path = get_manifest_path(ctx)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def is_manifest_compatible(manifest: dict, ctx) -> bool:
    """检查 manifest 是否与当前运行兼容"""
    # 检查 cache version
    if manifest.get("cache_version") != CACHE_VERSION:
        return False

    # 检查 input 指纹
    input_fingerprint = build_input_fingerprint(ctx.input_path)
    saved_input = manifest.get("input", {})
    if saved_input.get("path") != input_fingerprint["path"]:
        return False
    if saved_input.get("mtime") != input_fingerprint["mtime"]:
        return False
    if saved_input.get("size") != input_fingerprint["size"]:
        return False

    # 检查 config hash
    config_hash = compute_config_hash(ctx.config)
    if manifest.get("config_hash") != config_hash:
        return False

    return True


def save_manifest(ctx, manifest: dict):
    """保存 manifest 文件到输出目录"""
    path = get_manifest_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def record_stage_result(manifest: Dict, stage_name: str, outputs: Dict[str, Any]):
    """在 manifest 中记录某个 stage 的运行结果"""
    if "stages" not in manifest:
        manifest["stages"] = {}
    manifest["stages"][stage_name] = {
        "status": "done",
        "outputs": outputs,
    }


def restore_context_from_manifest(ctx, manifest: dict) -> bool:
    """从 manifest 恢复 ctx.data 内容"""
    stages = manifest.get("stages", {})
    restored_count = 0

    # pose_detect 恢复
    pose_stage = stages.get("pose_detect", {})
    if pose_stage.get("status") == "done":
        outputs = pose_stage.get("outputs", {})
        keypoints_path = outputs.get("keypoints_path")
        if keypoints_path and Path(keypoints_path).exists():
            ctx.set("keypoints_path", keypoints_path)
            # 加载 keypoints
            try:
                with open(keypoints_path, encoding="utf-8") as f:
                    raw = json.load(f)
                    keypoints = raw.get("keypoints", raw)
                    ctx.set("keypoints", keypoints)
            except Exception:
                pass
        video_info = outputs.get("video_info")
        if video_info:
            ctx.set("video_info", video_info)
        restored_count += 1

    # h2v_convert 恢复
    h2v_stage = stages.get("h2v_convert", {})
    if h2v_stage.get("status") == "done":
        outputs = h2v_stage.get("outputs", {})
        h2v_path = outputs.get("h2v_path")
        if h2v_path and Path(h2v_path).exists():
            ctx.set("h2v_path", h2v_path)
        cropped_kp_path = outputs.get("cropped_keypoints_path")
        if cropped_kp_path and Path(cropped_kp_path).exists():
            try:
                with open(cropped_kp_path, encoding="utf-8") as f:
                    ctx.set("cropped_keypoints", json.load(f))
            except Exception:
                pass
        h2v_size = outputs.get("h2v_size")
        if h2v_size:
            ctx.set("h2v_size", tuple(h2v_size))
        restored_count += 1

    # body_warp 恢复
    body_stage = stages.get("body_warp", {})
    if body_stage.get("status") == "done":
        outputs = body_stage.get("outputs", {})
        warped_path = outputs.get("warped_path")
        if warped_path and Path(warped_path).exists():
            ctx.set("warped_path", warped_path)
        restored_count += 1

    # color_grade 恢复
    color_stage = stages.get("color_grade", {})
    if color_stage.get("status") == "done":
        outputs = color_stage.get("outputs", {})
        color_path = outputs.get("color_path")
        if color_path and Path(color_path).exists():
            ctx.set("color_path", color_path)
        restored_count += 1

    # ken_burns 恢复
    kb_stage = stages.get("ken_burns", {})
    if kb_stage.get("status") == "done":
        outputs = kb_stage.get("outputs", {})
        kb_path = outputs.get("ken_burns_path")
        if kb_path and Path(kb_path).exists():
            ctx.set("ken_burns_path", kb_path)
        kb_ratio = outputs.get("ken_burns_ratio")
        if kb_ratio:
            ctx.set("ken_burns_ratio", kb_ratio)
        restored_count += 1

    # beat_flash 恢复
    bf_stage = stages.get("beat_flash", {})
    if bf_stage.get("status") == "done":
        outputs = bf_stage.get("outputs", {})
        bf_path = outputs.get("beatflash_path")
        if bf_path and Path(bf_path).exists():
            ctx.set("beatflash_path", bf_path)
        restored_count += 1

    return restored_count > 0


def init_manifest(ctx) -> Dict:
    """初始化一个新的 manifest"""
    return {
        "input": build_input_fingerprint(ctx.input_path),
        "config_hash": compute_config_hash(ctx.config),
        "cache_version": CACHE_VERSION,
        "stages": {},
    }
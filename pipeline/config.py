"""配置加载与管理"""

import yaml
from pathlib import Path
from copy import deepcopy

DEFAULT_CONFIG = {
    "stages": {
        "pose_detect": True,
        "stabilize": True,
        "h2v_convert": True,
        "ken_burns": False,
        "body_warp": True,
        "color_grade": True,
        "export": True,
    },
    "h2v": {
        "target_ratio": 9 / 16,
    },
    "body_warp": {
        "leg_lengthen": 1.0,
        "leg_slim": 1.0,
        "waist_slim": 1.0,
        "head_ratio": 1.0,
        "overall_slim": 1.0,
    },
    "color_grade": {
        "brightness": 0,
        "contrast": 1.0,
        "saturation": 1.0,
        "warmth": 0,
        "clahe": True,
    },
    "skin_tone_filter": {
        "pink_filter": 1.0,
        "warm_filter": 0.0,
        "cool_filter": 0.0,
        "soft_glow": 0.0,
    },
    "denoise": {
        "denoise_strength": 0,
        "denoise_mode": "fastNlMeans",
    },
    "ken_burns": {
        "mode": "smooth",
        "zoom_range": [1.0, 1.1],
    },
    "stabilize": {
        "shakiness": 5,
        "accuracy": 10,
    },
    "preview": False,
    "preview_seconds": 3,
}


def load_config(config_path: str = None) -> dict:
    """加载配置，合并默认值与用户配置"""
    cfg = deepcopy(DEFAULT_CONFIG)

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg:
            _deep_merge(cfg, user_cfg)

    return cfg


def load_preset(preset_name: str) -> dict:
    """加载预设配置"""
    preset_path = Path(__file__).parent.parent / "presets" / f"{preset_name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"预设不存在: {preset_name}")
    with open(preset_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict):
    """递归合并字典，override 的值覆盖 base"""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v

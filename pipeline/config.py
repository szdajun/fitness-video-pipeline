"""配置加载与管理"""

import logging
import yaml
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "stages": {
        "pre_deblock": False,
        "pose_detect": True,
        "stabilize": True,
        "h2v_convert": True,
        "ken_burns": False,
        "body_warp": True,
        "color_grade": True,
        "skin_tone_filter": True,
        "denoise": False,
        "watermark": False,
        "blush": False,
        "export": True,
        "face_enhance": False,
        "face_swap": False,
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
        "vignette_strength": 0,
        "vignette_radius": 0.8,
        "vignette_feather": 0.35,
        "film_grain_strength": 0,
        "film_grain_size": 2,
        "lut_path": "",
        "lut_intensity": 1.0,
        "lut_preset": "",
        "auto_exposure": 0,
        "ae_target": 128,
        "ae_speed": 0.05,
        "skin_protect": 0,
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
    "watermark": {
        "watermark_text": "",
        "watermark_position": "bottom-right",
        "watermark_size": 24,
        "watermark_color": (255, 255, 255),
        "watermark_alpha": 0.7,
        "watermark_margin": 20,
        "show_date": True,
    },
    "blush": {
        "blush_strength": 0.0,
        "brighten_strength": 0.0,
    },
    "ken_burns": {
        "mode": "smooth",
        "zoom_range": [1.0, 1.1],
        "track_smooth_window": 15,
        "track_zoom_range": [1.0, 1.08],
        "track_margin": 0.06,
    },
    "stabilize": {
        "shakiness": 5,
        "accuracy": 10,
    },
    "preview": False,
    "preview_seconds": 3,
    "cloud_enhance": {
        "enabled": False,
        "strength": 0.8,
        "mode": "cloud",
        "host": "",
        "port": 22,
        "user": "root",
        "password": "",
        "model_path": "",
    },
    "rife": {
        "enabled": False,
        "target_fps": 60,
        "gpu": True,
        "half_precision": True,
    },
    "pre_deblock": {
        "enabled": False,
        "quality": 4,
        "mode": 1,
    },
}


def _build_all_known_keys() -> set:
    """从 DEFAULT_CONFIG 和各处已知 key 自动构建完整 known keys 集合"""
    known = set()

    def _collect(d, top_level=True):
        for k, v in d.items():
            known.add(k)
            if isinstance(v, dict):
                _collect(v, top_level=False)
    _collect(DEFAULT_CONFIG)

    # 补充不在 DEFAULT_CONFIG 中但合法的 section / stage 名
    known.update({
        "process_isolate",  # 引擎子进程隔离开关 (config.yaml)
        "face_warp", "audio", "skeleton_overlay", "person_count",
        "lead_box", "lead_ghost", "face_blur", "motion_heatmap",
        "sync_score", "beat_flash", "highlight", "energy_bar",
        "intro_outro", "face_beautify", "face_beautify2", "export",
        "pre_deblock", "mascot", "bgm_beat",
        "output", "pose_backend", "pose_model", "pose_gpu",
        "full_video", "auto_preset", "skin_smooth", "face_enhance",
        "seo",
        # Missing top-level stage sections
        "pip", "speed_ramp", "film_look", "danmaku", "intensity_burst",
        "qin_cold_open",
    })
    # output section sub-keys
    known.update({
        "width", "height", "crf", "preset", "cut_ranges", "sharpen", "deblock",
        "formats",  # 多平台分发: ['9x16', '16x9'] 一次跑多格式输出
        "encoder", "resize_filter", "upscale_mode", "realesrgan_model",
        "realesrgan_scale", "realesrgan_tile", "realesrgan_gpu",
        "realesrgan_max_frames", "audio_bitrate", "video_fade_out",
    })
    # energy_bar section sub-keys
    known.update({
        "width", "margin_right", "margin_bottom", "height",
        "smoothing", "min_fill_ratio", "motion_scale",
    })
    # audio section sub-keys
    known.update({
        "target_lufs", "fade_in", "fade_out", "denoise", "ducking",
        "bg_music", "bg_volume",
    })
    # intro_outro section sub-keys
    known.update({
        "intro_duration", "outro_duration", "channel_name",
        "cta_text", "audio_fade_out", "fade_in_seconds",
        "fade_out_seconds", "location", "date",
    })
    # skin_smooth section sub-keys
    known.update({
        "strength", "d", "sigmaColor", "sigmaSpace", "downscale",
        "skin_detect",
    })
    # cloud_enhance sub-keys
    known.update({
        "enabled", "strength", "mode", "host", "port", "user", "password",
        "model_path",
    })
    # stabilize extends (beyond DEFAULT_CONFIG)
    known.update({
        "smoothing",
    })
    # face_beautify / face_beautify2 sub-keys
    known.update({
        "eye_brighten", "face_smooth", "eye_radius", "face_fill_light",
        "workers", "skin_smooth", "face_whiten", "face_slim", "eye_enlarge",
        "detect_interval",
    })
    # mascot sub-keys
    known.update({
        "size", "position", "bounce_range",
    })
    # bgm_beat sub-keys
    known.update({
        "bgm_file", "bg_volume", "beat_volume", "beat_width", "bgm_ratio",
    })
    # pip sub-keys
    known.update({
        "position", "scale", "margin", "border",
    })
    # speed_ramp sub-keys
    known.update({
        "slow_speed", "slow_window",
    })
    # film_look sub-keys
    known.update({
        "preset",
    })
    # danmaku sub-keys
    known.update({
        "font_size", "interval",
    })
    # qin_cold_open sub-keys
    known.update({
        "audio_file",
    })
    # color_grade — keys added mid-session (not yet in DEFAULT_CONFIG)
    known.update({
        "shadow", "auto_wb", "adaptive_contrast",
        "temporal_smooth", "highlight_protect", "highlight_threshold",
        "highlight_blur", "white_protect", "white_value_threshold",
        "white_sat_threshold", "white_protect_blur",
        "light_region_protect", "light_region_threshold",
        "light_region_min_area", "light_region_blur",
    })
    # seo sub-keys
    known.update({
        "enabled", "channel", "intensity", "audience", "tags",
        "traditional", "seo_keywords",
    })
    # face_swap sub-keys
    known.update({
        "every_n", "max_frames",
    })
    return known

_ALL_KNOWN_KEYS = _build_all_known_keys()


def _validate_config_keys(user_cfg: dict, prefix: str = ""):
    """递归校验 config key 拼写，未知 key 给出 warning"""
    for key in user_cfg:
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in _ALL_KNOWN_KEYS:
            logger.warning("未知配置项: '%s'", full_key)
        elif isinstance(user_cfg[key], dict):
            _validate_config_keys(user_cfg[key], full_key)


def load_config(config_path: str = None) -> dict:
    """加载配置，合并默认值与用户配置"""
    cfg = deepcopy(DEFAULT_CONFIG)

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg:
            _validate_config_keys(user_cfg)
            deep_merge(cfg, user_cfg, copy=False)

    return cfg


def load_preset(preset_name: str) -> dict:
    """加载预设配置"""
    preset_path = Path(__file__).parent.parent / "presets" / f"{preset_name}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"预设不存在: {preset_name}")
    with open(preset_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict, copy: bool = True) -> dict:
    """递归合并字典，override 的值覆盖 base

    Args:
        base: 基础字典
        override: 覆盖字典
        copy: 是否复制 base（True=返回新 dict，False=原地修改）

    Returns:
        合并后的字典
    """
    if copy:
        base = deepcopy(base)
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v, copy=False)
        else:
            if k in base and isinstance(base[k], dict) and not isinstance(v, dict):
                logger.warning("'%s': 非字典值(%s)覆盖了默认字典配置，可能导致阶段崩溃",
                               k, type(v).__name__)
            base[k] = v
    return base

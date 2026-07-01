"""守门: config.yaml 加载零未知配置项警告 (2026-07-02 债2).

之前每次跑 main.py 输出 6 行 '未知配置项' 噪音 (stages.shorts /
intro_outro.intro_music_from_main / output.prefer_gpu / paths / llm / coaches).
修复: 活的 3 个注册到 _ALL_KNOWN_KEYS, 遗留的 3 段 (paths/llm/coaches 主管线
零读取) 列为 _EXTERNAL_CONFIG_SECTIONS 让 validator 跳过."""
import logging
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent


def test_known_keys_registered():
    """主管线读取的 key 必须注册到 _ALL_KNOWN_KEYS (防再漏)"""
    from pipeline.config import _ALL_KNOWN_KEYS
    for k in ("shorts", "intro_music_from_main", "prefer_gpu"):
        assert k in _ALL_KNOWN_KEYS, f"key '{k}' 应注册到 _ALL_KNOWN_KEYS"


def test_external_sections_skipped():
    """paths/llm/coaches 是外部/遗留段 (主管线零读取), validator 应跳过"""
    from pipeline.config import _EXTERNAL_CONFIG_SECTIONS
    for k in ("paths", "llm", "coaches"):
        assert k in _EXTERNAL_CONFIG_SECTIONS, f"'{k}' 应在 _EXTERNAL_CONFIG_SECTIONS"


def test_config_yaml_no_unknown_warnings(caplog):
    """config.yaml 实际加载时不应有任何未知配置项警告"""
    from pipeline.config import _validate_config_keys
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    with caplog.at_level(logging.WARNING):
        _validate_config_keys(cfg)
    unknowns = [r.getMessage() for r in caplog.records if "未知配置项" in r.getMessage()]
    assert not unknowns, f"config.yaml 仍有未知配置项: {unknowns}"

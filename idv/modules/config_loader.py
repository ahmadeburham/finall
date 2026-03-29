"""Configuration loading and validation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(Exception):
    """Configuration loading error."""


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config from path and perform minimal schema checks."""
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError("Config must be a mapping at root level.")
    for key in ["detection", "alignment", "regions", "ocr", "validation", "decision"]:
        if key not in data:
            raise ConfigError(f"Missing required config section: {key}")
    return data

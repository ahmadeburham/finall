from pathlib import Path

from idv.modules.config_loader import load_config


def test_load_config():
    cfg = load_config(Path("config.yaml"))
    assert "regions" in cfg
    assert "ocr" in cfg

"""Health-check style smoke test for project wiring."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from idv.modules.config_loader import load_config
from idv.modules.normalization import normalize_digits, validate_birth_date, validate_id_number


def main() -> int:
    cfg = load_config(Path("config.yaml"))
    assert cfg["detection"]["feature_method"] == "orb"
    assert normalize_digits("١٢٣-45") == "12345"
    ok_bd, _ = validate_birth_date("01/01/2001", 1900, 2100)
    assert ok_bd
    ok_id, _ = validate_id_number("29801010101010", [14])
    assert ok_id
    _dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    print("SMOKE_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

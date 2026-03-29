"""General helpers used by pipeline."""
from __future__ import annotations

import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def generate_run_id() -> str:
    """Generate unique run identifier."""
    return uuid.uuid4().hex


def ensure_dir(path: Path) -> None:
    """Create directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write formatted UTF-8 JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def environment_info() -> Dict[str, str]:
    """Collect runtime environment information."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }


def elapsed_ms(start: float, end: float) -> float:
    """Convert seconds delta into milliseconds."""
    return round((end - start) * 1000.0, 3)

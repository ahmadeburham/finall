"""Image load/save utilities with validation."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class ImageLoadError(Exception):
    """Raised when image fails to load or validate."""


def load_color_image(path: Path) -> np.ndarray:
    """Load BGR image and validate non-empty content."""
    if not path.exists() or not path.is_file():
        raise ImageLoadError(f"Image path does not exist: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ImageLoadError(f"Image is empty or corrupted: {path}")
    return img


def save_image(path: Path, image: np.ndarray) -> str:
    """Save image and return string path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to save image: {path}")
    return str(path)


def image_size(image: np.ndarray) -> Tuple[int, int]:
    """Return width and height for image."""
    h, w = image.shape[:2]
    return w, h

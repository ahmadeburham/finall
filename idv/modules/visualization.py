"""Debug visual artifact export helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .image_io import save_image


def draw_matches(template: np.ndarray, scene: np.ndarray, kp_t, kp_s, matches: List[Any], out_path: Path) -> str:
    """Save keypoint match visualization."""
    vis = cv2.drawMatches(template, kp_t, scene, kp_s, matches[:80], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return save_image(out_path, vis)


def draw_quad_overlay(scene: np.ndarray, quad: np.ndarray, out_path: Path) -> str:
    """Draw detected quadrilateral on scene."""
    vis = scene.copy()
    q = quad.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [q], True, (0, 255, 255), 3)
    return save_image(out_path, vis)


def maybe_redact_regions(image: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Apply privacy-safe region redaction if enabled."""
    if not cfg["output"].get("redact_debug_regions", False):
        return image
    redacted = image.copy()
    h, w = redacted.shape[:2]
    for _, rcfg in cfg["regions"].items():
        x1, y1, x2, y2 = rcfg["box"]
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(redacted, p1, p2, (0, 0, 0), -1)
    return redacted

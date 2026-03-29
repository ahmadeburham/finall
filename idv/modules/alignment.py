"""Homography perspective warp and orientation handling."""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .schemas import StageResult


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order points as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def align_card(scene: np.ndarray, quad: np.ndarray, template_size: Tuple[int, int]) -> Dict[str, Any]:
    """Warp candidate quadrilateral into canonical template orientation."""
    t0 = time.time()
    res = StageResult(stage_name="stage_2_alignment", success=False, next_stage_allowed=False)
    try:
        tw, th = template_size
        src = order_quad(quad)
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        if not np.all(np.isfinite(m)):
            res.error_code = "ALIGN_BAD_MATRIX"
            res.error_message = "Perspective transform matrix invalid."
            return {"stage": res, "aligned": None}
        aligned = cv2.warpPerspective(scene, m, (tw, th))
        if aligned is None or aligned.size == 0:
            res.error_code = "ALIGN_EMPTY"
            res.error_message = "Aligned card is empty."
            return {"stage": res, "aligned": None}

        res.success = True
        res.next_stage_allowed = True
        return {"stage": res, "aligned": aligned, "matrix": m}
    finally:
        res.elapsed_ms = round((time.time() - t0) * 1000.0, 3)

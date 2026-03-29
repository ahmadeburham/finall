"""Template-relative crop extraction."""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .schemas import StageResult


def _to_px(box, w, h):
    x1, y1, x2, y2 = box
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def _apply_padding(x1, y1, x2, y2, pad, w, h):
    pl, pt, pr, pb = pad
    dx1 = int(pl * w)
    dy1 = int(pt * h)
    dx2 = int(pr * w)
    dy2 = int(pb * h)
    return max(0, x1 - dx1), max(0, y1 - dy1), min(w, x2 + dx2), min(h, y2 + dy2)


def extract_regions(aligned: np.ndarray, cfg: Dict[str, Any]) -> Tuple[StageResult, Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """Extract raw and padded field crops from aligned card using config boxes."""
    t0 = time.time()
    res = StageResult(stage_name="stage_3_region_extraction", success=False, next_stage_allowed=False)
    raw, padded = {}, {}
    vis = aligned.copy()
    try:
        h, w = aligned.shape[:2]
        for field, fcfg in cfg["regions"].items():
            x1, y1, x2, y2 = _to_px(fcfg["box"], w, h)
            px1, py1, px2, py2 = _apply_padding(x1, y1, x2, y2, fcfg["padding"], w, h)
            raw[field] = aligned[y1:y2, x1:x2].copy()
            padded[field] = aligned[py1:py2, px1:px2].copy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, field, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            res.metrics[f"{field}_raw_shape"] = list(raw[field].shape)
            res.metrics[f"{field}_padded_shape"] = list(padded[field].shape)

            if raw[field].size == 0 or padded[field].size == 0:
                res.error_code = "REGION_EMPTY_CROP"
                res.error_message = f"Empty crop generated for {field}."
                return res, raw, padded, vis

        res.success = True
        res.next_stage_allowed = True
        return res, raw, padded, vis
    finally:
        res.elapsed_ms = round((time.time() - t0) * 1000.0, 3)

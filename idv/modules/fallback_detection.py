"""Controlled fallback detection path."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .schemas import StageResult


def fallback_detect(template: np.ndarray, scene: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run single fallback path: contour rectangle proposal + histogram similarity."""
    t0 = time.time()
    res = StageResult(stage_name="stage_1b_fallback_detection", success=False, next_stage_allowed=False)
    try:
        if not cfg.get("fallback_detection", {}).get("enabled", True):
            res.error_code = "FALLBACK_DISABLED"
            res.error_message = "Fallback detection disabled by config."
            return {"stage": res, "quad": None}

        gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        c = cfg["fallback_detection"]["contour"]
        edges = cv2.Canny(gray, c["canny_low"], c["canny_high"])
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scene_area = scene.shape[0] * scene.shape[1]
        best_quad: Optional[np.ndarray] = None
        best_score = -1.0

        tpl_hist = cv2.calcHist([cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)], [0], None, [64], [0, 256])
        cv2.normalize(tpl_hist, tpl_hist)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, c["epsilon_factor"] * peri, True)
            if len(approx) != 4:
                continue
            area = cv2.contourArea(approx)
            if area / max(scene_area, 1) < c["min_area_ratio"]:
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            rect = cv2.boundingRect(quad.astype(np.int32))
            x, y, w, h = rect
            crop = gray[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            ch = cv2.calcHist([crop], [0], None, [64], [0, 256])
            cv2.normalize(ch, ch)
            score = float(cv2.compareHist(tpl_hist, ch, cv2.HISTCMP_CORREL))
            if score > best_score:
                best_score = score
                best_quad = quad

        res.metrics["contours_total"] = len(contours)
        res.metrics["best_hist_score"] = best_score
        if best_quad is None:
            res.error_code = "FALLBACK_NO_QUAD"
            res.error_message = "Fallback failed to produce a quadrilateral candidate."
            return {"stage": res, "quad": None}

        res.success = True
        res.next_stage_allowed = True
        return {"stage": res, "quad": best_quad}
    finally:
        res.elapsed_ms = round((time.time() - t0) * 1000.0, 3)

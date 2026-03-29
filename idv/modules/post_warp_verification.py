"""Post-alignment template similarity checks."""
from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np

from .schemas import StageResult


def verify_post_warp(aligned: np.ndarray, template: np.ndarray, cfg: Dict[str, Any]) -> StageResult:
    """Validate aligned card to reduce false positives."""
    t0 = time.time()
    res = StageResult(stage_name="stage_2b_post_warp_verification", success=False, next_stage_allowed=False)
    try:
        vcfg = cfg["alignment"]["post_warp"]
        gray_a = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        hist_a = cv2.calcHist([gray_a], [0], None, [64], [0, 256])
        hist_t = cv2.calcHist([gray_t], [0], None, [64], [0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_t, hist_t)
        corr = float(cv2.compareHist(hist_a, hist_t, cv2.HISTCMP_CORREL))
        brightness = float(gray_a.mean())
        contrast = float(gray_a.std())

        _, bw = cv2.threshold(gray_a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_density = float(np.count_nonzero(bw) / bw.size)

        res.metrics.update(
            {
                "hist_correlation": corr,
                "brightness": brightness,
                "contrast": contrast,
                "ink_density": ink_density,
            }
        )

        if corr < vcfg["min_hist_correlation"]:
            res.error_code = "POSTWARP_LOW_SIMILARITY"
            res.error_message = "Template similarity too low."
            return res
        if not (vcfg["brightness_min"] <= brightness <= vcfg["brightness_max"]):
            res.error_code = "POSTWARP_BAD_BRIGHTNESS"
            res.error_message = "Brightness outside expected range."
            return res
        if contrast < vcfg["contrast_min"]:
            res.error_code = "POSTWARP_LOW_CONTRAST"
            res.error_message = "Contrast too low for reliable OCR."
            return res
        if ink_density < vcfg["min_text_zone_ink_density"]:
            res.error_code = "POSTWARP_LOW_INK"
            res.error_message = "Text/ink density unexpectedly low."
            return res

        res.success = True
        res.next_stage_allowed = True
        return res
    finally:
        res.elapsed_ms = round((time.time() - t0) * 1000.0, 3)

"""Crop quality validation for OCR and face verification readiness."""
from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np

from .schemas import StageResult


def validate_crops(crops: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate each crop quality and return per-field validation metadata."""
    t0 = time.time()
    stage = StageResult(stage_name="stage_4_crop_validation", success=True, next_stage_allowed=True)
    out: Dict[str, Any] = {}
    c = cfg["crop_validation"]

    for field, img in crops.items():
        reasons = []
        h, w = img.shape[:2] if img.size else (0, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.size else np.zeros((1, 1), dtype=np.uint8)
        focus = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if img.size else 0.0
        brightness = float(gray.mean()) if img.size else 0.0
        contrast = float(gray.std()) if img.size else 0.0
        saturation = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) if img.size else 0.0
        dark_ratio = float((gray < 30).mean()) if img.size else 1.0
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_density = float(np.count_nonzero(thr) / thr.size) if thr.size else 0.0

        if img.size == 0:
            reasons.append("empty_crop")
        if w < c["min_width"] or h < c["min_height"]:
            reasons.append("crop_too_small")
        if focus < c["min_focus_var"]:
            reasons.append("blurry")
        if not (c["brightness_min"] <= brightness <= c["brightness_max"]):
            reasons.append("brightness_out_of_range")
        if contrast < c["contrast_min"]:
            reasons.append("low_contrast")
        if saturation > c["max_saturation_mean"]:
            reasons.append("oversaturated")
        if dark_ratio > c["max_dark_ratio"]:
            reasons.append("too_dark")
        if field != "portrait" and ink_density < c["min_ink_density"]:
            reasons.append("low_ink_density")

        valid_ocr = len([r for r in reasons if r != "face_not_found"]) == 0
        out[field] = {
            "valid_for_ocr": valid_ocr if field != "portrait" else False,
            "valid_for_face": field == "portrait" and "empty_crop" not in reasons,
            "reasons": reasons,
            "metrics": {
                "width": w,
                "height": h,
                "focus_var": focus,
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation,
                "dark_ratio": dark_ratio,
                "ink_density": ink_density,
            },
            "artifact_paths": {},
        }

    stage.metrics["fields_validated"] = len(out)
    stage.elapsed_ms = round((time.time() - t0) * 1000.0, 3)
    return {"stage": stage, "validation": out}

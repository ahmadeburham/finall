"""PaddleOCR field-level OCR execution."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2

from .normalization import normalize_text_whitespace
from .schemas import OCRFieldResult, StageResult


def _build_engine(cfg: Dict[str, Any]):
    from paddleocr import PaddleOCR

    use_gpu = bool(cfg["ocr"].get("use_gpu_if_available", False))
    return PaddleOCR(
        use_angle_cls=cfg["ocr"].get("cls", True),
        lang=cfg["ocr"].get("lang", "ar"),
        use_gpu=use_gpu,
        show_log=False,
    )


def _ocr_array(engine, arr) -> List[Dict[str, Any]]:
    raw = engine.ocr(arr, cls=True)
    out: List[Dict[str, Any]] = []
    for line in raw[0] if raw and raw[0] else []:
        text, conf = line[1][0], float(line[1][1])
        out.append({"text": text, "confidence": conf})
    return out


def _rotate(img, angle: int):
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def run_field_ocr(variant_paths: Dict[str, Dict[str, str]], chosen: Dict[str, str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run OCR per field and pick candidate according to field policy."""
    t0 = time.time()
    stage = StageResult(stage_name="stage_6_ocr", success=True, next_stage_allowed=True)
    fields: Dict[str, OCRFieldResult] = {}
    try:
        engine = _build_engine(cfg)
    except Exception as exc:  # explicit failure reporting
        stage.success = False
        stage.next_stage_allowed = False
        stage.error_code = "OCR_ENGINE_INIT_FAIL"
        stage.error_message = str(exc)
        stage.elapsed_ms = round((time.time() - t0) * 1000.0, 3)
        return {"stage": stage, "fields": fields}

    rotations = [int(x) for x in cfg["ocr"].get("try_rotations", [0])]
    for field in ["full_name", "full_address", "id_number", "birth_date"]:
        candidates = []
        for variant_name, path in variant_paths.get(field, {}).items():
            img = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
            if img is None:
                continue
            for angle in rotations:
                test_img = _rotate(img, angle)
                lines = _ocr_array(engine, test_img)
                raw_text = " ".join([x["text"] for x in lines]).strip()
                conf = max([x["confidence"] for x in lines], default=0.0)
                norm = normalize_text_whitespace(raw_text)
                digits = re.sub(r"\D", "", norm)
                candidates.append(
                    {
                        "variant": variant_name,
                        "rotation": angle,
                        "raw_text": raw_text,
                        "normalized_text": norm,
                        "confidence": conf,
                        "digits": digits,
                        "lines": lines,
                    }
                )

        if field in ("id_number", "birth_date"):
            candidates.sort(key=lambda x: (len(x["digits"]), x["confidence"]), reverse=True)
        else:
            candidates.sort(key=lambda x: (len(x["normalized_text"]), x["confidence"]), reverse=True)
        best = candidates[0] if candidates else {
            "variant": chosen.get(field, "unknown"),
            "rotation": 0,
            "raw_text": "",
            "normalized_text": "",
            "confidence": 0.0,
            "lines": [],
        }
        min_conf = cfg["ocr"]["min_confidence"][field]
        valid = bool(best["normalized_text"]) and best["confidence"] >= min_conf

        fields[field] = OCRFieldResult(
            raw_text=best["raw_text"],
            normalized_text=best["normalized_text"],
            confidence=float(best["confidence"]),
            alternate_candidates=[
                {
                    "variant": c["variant"],
                    "rotation": c["rotation"],
                    "raw_text": c["raw_text"],
                    "normalized_text": c["normalized_text"],
                    "confidence": c["confidence"],
                }
                for c in candidates[1:4]
            ],
            validation_passed=valid,
            failure_reason="" if valid else "low_confidence_or_empty",
            preprocessing_variant_used=f'{best["variant"]}@rot{best["rotation"]}',
            detector_metadata={"line_count": len(best.get("lines", []))},
            recognizer_metadata={"min_conf_threshold": min_conf, "tested_rotations": rotations},
        )

    stage.metrics["ocr_fields"] = len(fields)
    stage.metrics["tested_rotations"] = rotations
    stage.elapsed_ms = round((time.time() - t0) * 1000.0, 3)
    return {"stage": stage, "fields": fields}

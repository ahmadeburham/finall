"""Field-specific preprocessing variant generation and selection."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .image_io import save_image
from .schemas import StageResult


def _name_variants(img: np.ndarray) -> Dict[str, np.ndarray]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "name_gray": g,
        "name_clahe": cv2.createCLAHE(2.0, (8, 8)).apply(g),
        "name_denoise": cv2.fastNlMeansDenoising(g, None, 8, 7, 21),
    }


def _address_variants(img: np.ndarray) -> Dict[str, np.ndarray]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    return {
        "address_gray": g,
        "address_clahe": cv2.createCLAHE(2.5, (8, 8)).apply(g),
        "address_denoise": cv2.fastNlMeansDenoising(blur, None, 10, 7, 21),
    }


def _id_variants(img: np.ndarray) -> Dict[str, np.ndarray]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 6)
    otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return {"id_gray": g, "id_adaptive": th, "id_otsu": otsu}


def _birth_variants(img: np.ndarray) -> Dict[str, np.ndarray]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "birth_gray": g,
        "birth_clahe": cv2.createCLAHE(2.0, (8, 8)).apply(g),
        "birth_otsu": cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    }


def preprocess_fields(crops: Dict[str, np.ndarray], out_dir: Path) -> Tuple[StageResult, Dict[str, Dict[str, str]], Dict[str, str]]:
    """Generate controlled preprocessing variants and choose default initial candidate per field."""
    t0 = time.time()
    stage = StageResult(stage_name="stage_5_preprocessing", success=True, next_stage_allowed=True)
    variants_paths: Dict[str, Dict[str, str]] = {}
    chosen: Dict[str, str] = {}

    for field, img in crops.items():
        if field == "full_name":
            variants = _name_variants(img)
        elif field == "full_address":
            variants = _address_variants(img)
        elif field == "id_number":
            variants = _id_variants(img)
        elif field == "birth_date":
            variants = _birth_variants(img)
        elif field == "portrait":
            variants = {"portrait_rgb": img}
        else:
            variants = {f"{field}_raw": img}

        variants_paths[field] = {}
        best = list(variants.keys())[0]
        chosen[field] = best
        for name, var_img in variants.items():
            path = out_dir / field / f"{name}.png"
            save_image(path, var_img)
            variants_paths[field][name] = str(path)

    stage.metrics["fields_preprocessed"] = len(crops)
    stage.elapsed_ms = round((time.time() - t0) * 1000.0, 3)
    return stage, variants_paths, chosen

"""Rectification refinement using OpenCV ECC alignment."""
from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np

from .schemas import StageResult


def refine_alignment_ecc(aligned: np.ndarray, template: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Refine aligned card with ECC maximization against template grayscale.

    Uses cv2.findTransformECC from OpenCV's official video module.
    """
    t0 = time.time()
    stage = StageResult(stage_name="stage_2a_ecc_refinement", success=False, next_stage_allowed=False)
    try:
        ecc_cfg = cfg.get("alignment", {}).get("ecc", {})
        if not ecc_cfg.get("enabled", True):
            stage.success = True
            stage.next_stage_allowed = True
            stage.warnings.append("ecc_disabled")
            return {"stage": stage, "aligned": aligned, "warp_matrix": None}

        g_a = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        g_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        mode_name = ecc_cfg.get("motion_model", "affine").lower()
        mode = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
            "homography": cv2.MOTION_HOMOGRAPHY,
        }.get(mode_name, cv2.MOTION_AFFINE)

        if mode == cv2.MOTION_HOMOGRAPHY:
            warp = np.eye(3, 3, dtype=np.float32)
        else:
            warp = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(ecc_cfg.get("max_iterations", 100)),
            float(ecc_cfg.get("epsilon", 1e-5)),
        )

        cc, warp = cv2.findTransformECC(
            templateImage=g_t,
            inputImage=g_a,
            warpMatrix=warp,
            motionType=mode,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=int(ecc_cfg.get("gaussian_filter_size", 5)),
        )

        if mode == cv2.MOTION_HOMOGRAPHY:
            refined = cv2.warpPerspective(
                aligned,
                warp,
                (template.shape[1], template.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            refined = cv2.warpAffine(
                aligned,
                warp,
                (template.shape[1], template.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )

        stage.success = True
        stage.next_stage_allowed = True
        stage.metrics["ecc_correlation"] = float(cc)
        return {"stage": stage, "aligned": refined, "warp_matrix": warp.tolist()}
    except cv2.error as exc:
        stage.error_code = "ECC_REFINEMENT_FAIL"
        stage.error_message = str(exc)
        return {"stage": stage, "aligned": aligned, "warp_matrix": None}
    finally:
        stage.elapsed_ms = round((time.time() - t0) * 1000.0, 3)

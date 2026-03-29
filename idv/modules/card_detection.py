"""Primary ORB-based template matching card detection."""
from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np

from .schemas import StageResult


def _order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def detect_card_orb(template: np.ndarray, scene: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Detect candidate card quadrilateral using ORB feature matching."""
    t0 = time.time()
    result = StageResult(stage_name="stage_1_card_detection", success=False, next_stage_allowed=False)
    try:
        orb_cfg = cfg["detection"]["orb"]
        mcfg = cfg["detection"]["matching"]
        vcfg = cfg["detection"]["candidate_validation"]

        orb = cv2.ORB_create(
            nfeatures=orb_cfg["nfeatures"],
            scaleFactor=orb_cfg["scaleFactor"],
            nlevels=orb_cfg["nlevels"],
            edgeThreshold=orb_cfg["edgeThreshold"],
        )
        g_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        g_s = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
        kp_t, des_t = orb.detectAndCompute(g_t, None)
        kp_s, des_s = orb.detectAndCompute(g_s, None)

        result.metrics["keypoints_template"] = len(kp_t)
        result.metrics["keypoints_scene"] = len(kp_s)
        if len(kp_t) < mcfg["min_keypoints_template"] or len(kp_s) < mcfg["min_keypoints_scene"]:
            result.error_code = "DET_LOW_KEYPOINTS"
            result.error_message = "Insufficient keypoints for robust matching."
            return {"stage": result, "quad": None, "good_matches": [], "kp_t": kp_t, "kp_s": kp_s}
        if des_t is None or des_s is None:
            result.error_code = "DET_NO_DESCRIPTORS"
            result.error_message = "No descriptors computed."
            return {"stage": result, "quad": None, "good_matches": [], "kp_t": kp_t, "kp_s": kp_s}

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw = bf.knnMatch(des_t, des_s, k=2)
        result.metrics["raw_matches"] = len(raw)
        if len(raw) < mcfg["min_raw_matches"]:
            result.error_code = "DET_LOW_RAW_MATCHES"
            result.error_message = "Raw matches below threshold."
            return {"stage": result, "quad": None, "good_matches": [], "kp_t": kp_t, "kp_s": kp_s}

        good = []
        for pair in raw:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < mcfg["ratio_test"] * n.distance:
                good.append(m)
        result.metrics["good_matches"] = len(good)
        if len(good) < mcfg["min_good_matches"]:
            result.error_code = "DET_LOW_GOOD_MATCHES"
            result.error_message = "Good matches below threshold."
            return {"stage": result, "quad": None, "good_matches": good, "kp_t": kp_t, "kp_s": kp_s}

        src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        method_name = str(mcfg.get("homography_method", "ransac")).lower()
        method = cv2.RANSAC
        if method_name == "usac_magsac" and hasattr(cv2, "USAC_MAGSAC"):
            method = cv2.USAC_MAGSAC
        elif method_name == "usac_default" and hasattr(cv2, "USAC_DEFAULT"):
            method = cv2.USAC_DEFAULT
        h, mask = cv2.findHomography(src_pts, dst_pts, method, mcfg["ransac_reproj_threshold"])
        result.metrics["homography_method_used"] = method_name if method != cv2.RANSAC else "ransac"
        if h is None or not np.all(np.isfinite(h)):
            result.error_code = "DET_HOMOGRAPHY_FAIL"
            result.error_message = "Homography estimation failed."
            return {"stage": result, "quad": None, "good_matches": good, "kp_t": kp_t, "kp_s": kp_s}

        th, tw = template.shape[:2]
        corners = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, h).reshape(-1, 2)
        quad = _order_quad(proj)

        area = cv2.contourArea(quad.astype(np.float32))
        scene_area = scene.shape[0] * scene.shape[1]
        area_ratio = area / max(scene_area, 1)
        result.metrics["candidate_area_ratio"] = float(area_ratio)
        if area_ratio < vcfg["min_area_ratio"] or area_ratio > vcfg["max_area_ratio"]:
            result.error_code = "DET_BAD_AREA"
            result.error_message = "Candidate area ratio outside limits."
            return {"stage": result, "quad": None, "good_matches": good, "kp_t": kp_t, "kp_s": kp_s}

        edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
        if min(edges) < vcfg["min_edge_length_px"]:
            result.error_code = "DET_EDGE_TOO_SHORT"
            result.error_message = "Detected quadrilateral is too small."
            return {"stage": result, "quad": None, "good_matches": good, "kp_t": kp_t, "kp_s": kp_s}

        cand_aspect = (edges[0] + edges[2]) / max((edges[1] + edges[3]), 1e-6)
        tpl_aspect = tw / max(th, 1)
        result.metrics["candidate_aspect"] = float(cand_aspect)
        if abs(cand_aspect - tpl_aspect) > cfg["detection"]["candidate_validation"]["aspect_ratio_tolerance"]:
            result.error_code = "DET_BAD_ASPECT"
            result.error_message = "Aspect ratio mismatch with template."
            return {"stage": result, "quad": None, "good_matches": good, "kp_t": kp_t, "kp_s": kp_s}

        result.success = True
        result.next_stage_allowed = True
        result.metrics["inliers"] = int(mask.sum()) if mask is not None else 0
        return {"stage": result, "quad": quad, "good_matches": good, "kp_t": kp_t, "kp_s": kp_s}
    finally:
        result.elapsed_ms = round((time.time() - t0) * 1000.0, 3)

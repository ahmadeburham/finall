"""InsightFace-based portrait/selfie face verification."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


class FaceVerifier:
    """Wrapper around InsightFace for swap-friendly integration."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.app = None

    def _init_app(self) -> None:
        from insightface.app import FaceAnalysis

        providers = self.cfg["face"]["providers_cpu"]
        if self.cfg["face"].get("use_gpu_if_available", False):
            providers = self.cfg["face"].get("providers_gpu", providers)
        model_pack = self.cfg["face"].get("model_pack", "buffalo_l")
        self.app = FaceAnalysis(name=model_pack, providers=providers)
        self.app.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1, det_size=(640, 640))

    def verify(self, selfie: np.ndarray, portrait: np.ndarray) -> Dict[str, Any]:
        """Compare selfie and portrait embeddings with threshold decision."""
        if self.app is None:
            try:
                self._init_app()
            except Exception as exc:
                return {
                    "attempted": True,
                    "success": False,
                    "verified": False,
                    "score": None,
                    "threshold": self.cfg["face"]["similarity_threshold"],
                    "face_found_in_selfie": False,
                    "face_found_in_portrait": False,
                    "failure_reason": f"face_model_init_failed:{exc}",
                }

        s_faces = self.app.get(selfie)
        p_faces = self.app.get(portrait)

        out = {
            "attempted": True,
            "success": False,
            "verified": False,
            "score": None,
            "threshold": self.cfg["face"]["similarity_threshold"],
            "face_found_in_selfie": len(s_faces) > 0,
            "face_found_in_portrait": len(p_faces) > 0,
            "failure_reason": "",
        }
        if not s_faces or not p_faces:
            out["failure_reason"] = "face_missing"
            return out

        se = s_faces[0].embedding
        pe = p_faces[0].embedding
        cos = float(np.dot(se, pe) / (np.linalg.norm(se) * np.linalg.norm(pe) + 1e-8))
        dist = 1.0 - cos

        out["score"] = dist
        out["success"] = True
        out["verified"] = dist <= out["threshold"]
        if not out["verified"]:
            out["failure_reason"] = "distance_above_threshold"
        return out


def save_face_debug(selfie: np.ndarray, portrait: np.ndarray, out_dir: Path) -> Dict[str, str]:
    """Save face debug crops."""
    out_dir.mkdir(parents=True, exist_ok=True)
    selfie_path = out_dir / "selfie_input.png"
    portrait_path = out_dir / "portrait_input.png"
    cv2.imwrite(str(selfie_path), selfie)
    cv2.imwrite(str(portrait_path), portrait)
    return {"selfie": str(selfie_path), "portrait": str(portrait_path)}

"""Final decision logic for ACCEPT/REVIEW/REJECT."""
from __future__ import annotations

from typing import Any, Dict, Tuple


def decide(
    id_detected: bool,
    alignment_ok: bool,
    ocr_results: Dict[str, Dict[str, Any]],
    field_validation: Dict[str, bool],
    cfg: Dict[str, Any],
) -> Tuple[str, float, list[str]]:
    """Compute final disposition with explicit reasons."""
    reasons: list[str] = []
    if not id_detected:
        return "REJECT", 0.0, ["id_not_detected"]
    if not alignment_ok:
        return "REJECT", 0.0, ["alignment_failed"]

    required = cfg["decision"]["required_fields"]
    pass_count = 0
    confs = []
    for field in required:
        field_ok = bool(field_validation.get(field, False))
        if field_ok:
            pass_count += 1
        confs.append(float(ocr_results.get(field, {}).get("confidence", 0.0)))
        if not field_ok:
            reasons.append(f"{field}_validation_failed")

    overall = sum(confs) / max(len(confs), 1)
    if pass_count == len(required) and overall >= cfg["decision"]["min_overall_confidence_accept"]:
        return "ACCEPT", overall, reasons
    if pass_count >= cfg["decision"]["min_required_pass_for_review"]:
        return "REVIEW", overall, reasons
    return "REJECT", overall, reasons

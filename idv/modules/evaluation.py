"""Batch evaluation helpers for aggregate metrics."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List


def char_similarity(a: str, b: str) -> float:
    """Character-level similarity ratio."""
    return SequenceMatcher(None, a or "", b or "").ratio()


def aggregate(results: List[Dict[str, Any]], labels: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Compute aggregate metrics from per-image pipeline results."""
    total = len(results)
    det_ok = sum(1 for r in results if r.get("id_detected"))
    align_ok = sum(1 for r in results if r.get("stage_results", {}).get("stage_2_alignment", {}).get("success"))
    decisions = Counter(r.get("final_decision", "UNKNOWN") for r in results)

    agg: Dict[str, Any] = {
        "total_images": total,
        "card_detection_success_rate": det_ok / total if total else 0.0,
        "alignment_success_rate": align_ok / total if total else 0.0,
        "decision_counts": dict(decisions),
        "per_field_exact_match_accuracy": {},
        "per_field_char_similarity_mean": {},
        "per_field_digits_exact_match_accuracy": {},
        "failure_breakdown_by_stage": Counter(),
    }

    for r in results:
        for stage_name, stage in r.get("stage_results", {}).items():
            if not stage.get("success", False):
                agg["failure_breakdown_by_stage"][stage_name] += 1

    if labels:
        exact_counts = Counter()
        sim_sums = Counter()
        sim_counts = Counter()
        digit_exact = Counter()
        digit_count = Counter()

        for r in results:
            key = r.get("input_paths", {}).get("scene")
            if not key or key not in labels:
                continue
            gt = labels[key]
            o = r.get("ocr_outputs", {})
            for field in ["full_name", "full_address", "id_number_digits_only", "birth_date_digits_only"]:
                pred = (o.get(field) or "").strip()
                exp = str(gt.get(field, "")).strip()
                if pred == exp:
                    exact_counts[field] += 1
                sim_sums[field] += char_similarity(pred, exp)
                sim_counts[field] += 1
                if "digits_only" in field:
                    digit_count[field] += 1
                    if pred == exp:
                        digit_exact[field] += 1

        for field, c in sim_counts.items():
            agg["per_field_exact_match_accuracy"][field] = exact_counts[field] / c if c else 0.0
            agg["per_field_char_similarity_mean"][field] = sim_sums[field] / c if c else 0.0
        for field, c in digit_count.items():
            agg["per_field_digits_exact_match_accuracy"][field] = digit_exact[field] / c if c else 0.0

    agg["failure_breakdown_by_stage"] = dict(agg["failure_breakdown_by_stage"])
    return agg

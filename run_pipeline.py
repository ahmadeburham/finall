"""CLI entrypoint for Egyptian ID verification pipeline."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import cv2

from idv import __version__
from idv.modules.alignment import align_card
from idv.modules.card_detection import detect_card_orb
from idv.modules.config_loader import load_config
from idv.modules.crop_validation import validate_crops
from idv.modules.decision_engine import decide
from idv.modules.face_verification import FaceVerifier, save_face_debug
from idv.modules.fallback_detection import fallback_detect
from idv.modules.image_io import ImageLoadError, load_color_image, save_image
from idv.modules.logging_utils import setup_logger
from idv.modules.normalization import normalize_digits, validate_birth_date, validate_id_number
from idv.modules.ocr_engine import run_field_ocr
from idv.modules.post_warp_verification import verify_post_warp
from idv.modules.rectification import refine_alignment_ecc
from idv.modules.preprocessing import preprocess_fields
from idv.modules.region_extraction import extract_regions
from idv.modules.schemas import RunContext, StageResult
from idv.modules.utils import elapsed_ms, ensure_dir, environment_info, generate_run_id, utc_now_iso, write_json
from idv.modules.visualization import draw_matches, draw_quad_overlay, maybe_redact_regions


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Egyptian ID verification pipeline")
    p.add_argument("--scene", required=True, type=Path)
    p.add_argument("--template", required=True, type=Path)
    p.add_argument("--selfie", type=Path, default=None)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def stage_to_dict(stage: StageResult) -> Dict[str, Any]:
    return stage.to_dict()


def create_context(args: argparse.Namespace, cfg: Dict[str, Any]) -> RunContext:
    run_id = generate_run_id()
    run_dir = args.output_dir / f"run_{run_id}"
    artifacts = run_dir / "artifacts"
    logs = run_dir / "logs"
    metrics = run_dir / "metrics"
    for p in [run_dir, artifacts, logs, metrics]:
        ensure_dir(p)
    return RunContext(
        run_id=run_id,
        output_dir=args.output_dir,
        run_dir=run_dir,
        artifacts_dir=artifacts,
        logs_dir=logs,
        metrics_dir=metrics,
        config_path=args.config,
        config=cfg,
        scene_path=args.scene,
        template_path=args.template,
        selfie_path=args.selfie,
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    ctx = create_context(args, cfg)
    logger = setup_logger(ctx.logs_dir / "run.log")

    stage_results: Dict[str, Dict[str, Any]] = {}
    warnings: list[str] = []
    failures: list[str] = []
    data: Dict[str, Any] = {
        "aligned_card_path": None,
        "crop_paths": {},
        "crop_validation": {},
        "ocr_outputs": {},
        "face_verification": {"attempted": False, "success": False, "verified": False},
    }

    t0 = time.time()
    s0 = StageResult(stage_name="stage_0_input_validation", success=False, next_stage_allowed=False)
    try:
        scene = load_color_image(ctx.scene_path)
        template = load_color_image(ctx.template_path)
        selfie = load_color_image(ctx.selfie_path) if ctx.selfie_path else None
        s0.success = True
        s0.next_stage_allowed = True
        s0.metrics.update(
            {
                "scene_shape": list(scene.shape),
                "template_shape": list(template.shape),
                "selfie_provided": bool(selfie is not None),
            }
        )
    except (ImageLoadError, Exception) as exc:
        s0.error_code = "INPUT_VALIDATION_FAIL"
        s0.error_message = str(exc)
        failures.append(str(exc))
        scene = template = selfie = None
    finally:
        s0.elapsed_ms = elapsed_ms(t0, time.time())
        stage_results[s0.stage_name] = stage_to_dict(s0)

    if s0.success:
        det = detect_card_orb(template, scene, cfg)
        s1 = det["stage"]
        stage_results[s1.stage_name] = stage_to_dict(s1)
        kp_vis = cv2.drawKeypoints(template, det["kp_t"], None, color=(0, 255, 0))
        save_image(ctx.artifacts_dir / "template_keypoints.png", kp_vis)

        quad = det["quad"]
        if quad is not None:
            stage_results[s1.stage_name]["artifacts"]["match_visualization"] = draw_matches(
                template, scene, det["kp_t"], det["kp_s"], det["good_matches"], ctx.artifacts_dir / "matches.png"
            )
            stage_results[s1.stage_name]["artifacts"]["quad_overlay"] = draw_quad_overlay(scene, quad, ctx.artifacts_dir / "detected_quad.png")
        else:
            fb = fallback_detect(template, scene, cfg)
            s1b = fb["stage"]
            stage_results[s1b.stage_name] = stage_to_dict(s1b)
            quad = fb["quad"]
            if quad is None:
                failures.append(s1.error_message or s1b.error_message)

        if quad is not None:
            a = align_card(scene, quad, (template.shape[1], template.shape[0]))
            s2 = a["stage"]
            stage_results[s2.stage_name] = stage_to_dict(s2)
            if s2.success:
                aligned = a["aligned"]

                ecc = refine_alignment_ecc(aligned, template, cfg)
                s2a = ecc["stage"]
                stage_results[s2a.stage_name] = stage_to_dict(s2a)
                if s2a.success:
                    aligned = ecc["aligned"]
                else:
                    warnings.append("ecc_refinement_failed_using_base_alignment")

                aligned_dbg = maybe_redact_regions(aligned, cfg) if cfg["output"].get("privacy_safe_mode") else aligned
                data["aligned_card_path"] = save_image(ctx.artifacts_dir / "aligned_card.png", aligned_dbg)

                s2b = verify_post_warp(aligned, template, cfg)
                stage_results[s2b.stage_name] = stage_to_dict(s2b)
                if s2b.success:
                    s3, raw_crops, padded_crops, box_vis = extract_regions(aligned, cfg)
                    stage_results[s3.stage_name] = stage_to_dict(s3)
                    if s3.success:
                        data["crop_paths"]["boxes_overlay"] = save_image(ctx.artifacts_dir / "region_boxes.png", box_vis)
                        for k, cimg in raw_crops.items():
                            data["crop_paths"][f"raw_{k}"] = save_image(ctx.artifacts_dir / "crops" / f"raw_{k}.png", cimg)
                        for k, cimg in padded_crops.items():
                            data["crop_paths"][f"padded_{k}"] = save_image(ctx.artifacts_dir / "crops" / f"padded_{k}.png", cimg)

                        cv_out = validate_crops(padded_crops, cfg)
                        s4 = cv_out["stage"]
                        stage_results[s4.stage_name] = stage_to_dict(s4)
                        data["crop_validation"] = cv_out["validation"]

                        s5, variant_paths, chosen = preprocess_fields(padded_crops, ctx.artifacts_dir / "preprocessed")
                        stage_results[s5.stage_name] = stage_to_dict(s5)

                        ocr_out = run_field_ocr(variant_paths, chosen, cfg)
                        s6 = ocr_out["stage"]
                        stage_results[s6.stage_name] = stage_to_dict(s6)
                        ocr_fields = {k: v.to_dict() for k, v in ocr_out["fields"].items()}

                        min_y = cfg["validation"]["birth_date_min_year"]
                        max_y = cfg["validation"]["birth_date_max_year"]
                        id_lengths = cfg["validation"]["id_number_lengths"]

                        birth_raw = ocr_fields.get("birth_date", {}).get("normalized_text", "")
                        id_raw = ocr_fields.get("id_number", {}).get("normalized_text", "")
                        birth_ok, birth_meta = validate_birth_date(birth_raw, min_y, max_y)
                        id_ok, id_meta = validate_id_number(id_raw, id_lengths)

                        field_validity = {
                            "full_name": bool(ocr_fields.get("full_name", {}).get("validation_passed", False)),
                            "full_address": bool(ocr_fields.get("full_address", {}).get("validation_passed", False)),
                            "id_number": id_ok,
                            "birth_date": birth_ok,
                        }

                        decision, overall_conf, decision_reasons = decide(
                            id_detected=True,
                            alignment_ok=True,
                            ocr_results=ocr_fields,
                            field_validation=field_validity,
                            cfg=cfg,
                        )

                        data["ocr_outputs"] = {
                            "full_name": ocr_fields.get("full_name", {}).get("normalized_text", ""),
                            "full_address": ocr_fields.get("full_address", {}).get("normalized_text", ""),
                            "id_number_raw": id_raw,
                            "id_number_digits_only": id_meta["digits_only"],
                            "birth_date_raw": birth_raw,
                            "birth_date_digits_only": birth_meta.get("digits_only", normalize_digits(birth_raw)),
                            "birth_date_formatted_if_reliable": birth_meta.get("formatted", "") if birth_ok else "",
                            "confidences": {k: v.get("confidence", 0.0) for k, v in ocr_fields.items()},
                            "fields": ocr_fields,
                        }

                        if selfie is not None and cfg["face"].get("enabled", True):
                            fv = FaceVerifier(cfg)
                            try:
                                data["face_verification"] = fv.verify(selfie, padded_crops["portrait"])
                                data["face_verification"]["debug_artifacts"] = save_face_debug(
                                    selfie, padded_crops["portrait"], ctx.artifacts_dir / "face_debug"
                                )
                            except Exception as exc:
                                data["face_verification"] = {
                                    "attempted": True,
                                    "success": False,
                                    "verified": False,
                                    "score": None,
                                    "threshold": cfg["face"]["similarity_threshold"],
                                    "face_found_in_selfie": False,
                                    "face_found_in_portrait": False,
                                    "failure_reason": str(exc),
                                }
                                warnings.append("face_verification_failed")
                        else:
                            data["face_verification"] = {
                                "attempted": False,
                                "success": True,
                                "verified": False,
                                "score": None,
                                "threshold": cfg["face"]["similarity_threshold"],
                                "face_found_in_selfie": False,
                                "face_found_in_portrait": False,
                                "failure_reason": "selfie_not_provided_or_face_disabled",
                            }

                        data["final_decision"] = decision
                        data["overall_document_confidence"] = overall_conf
                        data["failure_reasons"] = decision_reasons
                    else:
                        failures.append(s3.error_message)
                else:
                    failures.append(s2b.error_message)
            else:
                failures.append(s2.error_message)

    manifest = {
        "run_id": ctx.run_id,
        "pipeline_version": __version__,
        "config_version": cfg.get("version", "unknown"),
        "timestamp": utc_now_iso(),
        "input_paths": {"scene": str(ctx.scene_path), "template": str(ctx.template_path), "selfie": str(ctx.selfie_path) if ctx.selfie_path else None},
        "environment_info": environment_info(),
    }
    write_json(ctx.run_dir / "run_manifest.json", manifest)

    result = {
        "run_id": ctx.run_id,
        "pipeline_version": __version__,
        "config_version": cfg.get("version", "unknown"),
        "timestamp": utc_now_iso(),
        "overall_status": "SUCCESS" if not failures else "PARTIAL_FAILURE",
        "final_decision": data.get("final_decision", "REJECT"),
        "input_paths": manifest["input_paths"],
        "template_path": str(ctx.template_path),
        "id_detected": bool(stage_results.get("stage_1_card_detection", {}).get("success") or stage_results.get("stage_1b_fallback_detection", {}).get("success")),
        "detection_metrics": stage_results.get("stage_1_card_detection", {}).get("metrics", {}),
        "stage_results": stage_results,
        "aligned_card_path": data.get("aligned_card_path"),
        "crop_paths": data.get("crop_paths", {}),
        "crop_validation": data.get("crop_validation", {}),
        "ocr_outputs": data.get("ocr_outputs", {}),
        "face_verification": data.get("face_verification", {}),
        "warnings": warnings,
        "failure_reasons": failures + data.get("failure_reasons", []),
        "environment_info": manifest["environment_info"],
    }

    if result["final_decision"] in {"REVIEW", "REJECT"}:
        ensure_dir(ctx.run_dir / "review_package")
        write_json(ctx.run_dir / "review_package" / "review_summary.json", result)

    write_json(ctx.run_dir / "result.json", result)
    write_json(ctx.run_dir / "stage_metrics.json", {k: v.get("metrics", {}) for k, v in stage_results.items()})
    logger.info("pipeline_finished", extra={"run_id": ctx.run_id, "final_decision": result["final_decision"]})
    print(str(ctx.run_dir / "result.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

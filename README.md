# Egyptian ID Verification Pipeline (Local, OpenCV + PaddleOCR + InsightFace)

Production-minded local pipeline for Egyptian ID detection, alignment, field extraction, OCR, and optional selfie-to-portrait face verification.

## Features
- **Geometric refinement**: ECC-based post-warp registration refinement before OCR field extraction.
- **Primary localization**: ORB template feature matching + homography + perspective warp.
- **Controlled fallback**: contour quadrilateral proposal verified with template histogram similarity.
- **Template-first extraction**: config-defined normalized regions for portrait, birth date, full name, full address, and ID number.
- **Field-specific processing**: per-field preprocessing variants and selection policy.
- **Field-level OCR**: PaddleOCR run separately per text field.
- **Validation and decisioning**: explicit stage contracts, structured failures, and final `ACCEPT/REVIEW/REJECT` decision.
- **Optional face verification**: InsightFace-based embedding comparison.
- **Batch evaluation**: aggregate detection/alignment/OCR metrics with optional labels.
- **Production readiness**: strict config separation, JSON logs, run manifests, review packages, smoke test.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Commands
Primary inference:
```bash
python run_pipeline.py --scene path/to/scene.jpg --template path/to/template.jpg --output-dir out
```

With selfie:
```bash
python run_pipeline.py --scene path/to/scene.jpg --template path/to/template.jpg --selfie path/to/selfie.jpg --output-dir out
```

Batch evaluation:
```bash
python evaluate_pipeline.py --input-dir path/to/images --template path/to/template.jpg --labels path/to/labels.json --output-dir eval_out
```

Smoke test:
```bash
python smoke_test.py
```

Unit tests:
```bash
pytest -q
```

## Config Overview
All thresholds, region boxes, paddings, fallback behavior, validation limits, OCR limits, and face thresholds are in `config.yaml`.

- `detection.matching.homography_method`: choose `usac_magsac` (preferred when available) or `ransac`.
- `alignment.ecc.*`: OpenCV ECC refinement settings applied after perspective warp.
- `ocr.try_rotations`: explicit angle sweep for field-level OCR candidate selection.

- `regions.*.box`: normalized `[x1, y1, x2, y2]` w.r.t aligned template size.
- Current default region boxes are calibrated to the provided reference card image where the black rectangles mark required extraction zones.
- `regions.*.padding`: per-field normalized padding `[left, top, right, bottom]`.
- `output.privacy_safe_mode` and `output.redact_debug_regions`: privacy-aware debug controls.

## Output Structure
Each run creates `output-dir/run_<run_id>/` containing:
- `result.json`
- `stage_metrics.json`
- `run_manifest.json`
- `logs/run.log`
- `artifacts/` (alignment, overlays, crops, preprocessed variants, optional face debug)
- `review_package/review_summary.json` when decision is `REVIEW` or `REJECT`

## Evaluation
- Works in inference-only mode (no labels).
- If labels are provided, computes exact-match and char-similarity metrics, numeric digits-only exact matches, and failure breakdown by stage.

## Limitations (Honest)
- OCR quality heavily depends on crop quality and template consistency.
- Histogram-based post-warp verification can fail for severe illumination shifts.
- Fallback path is intentionally constrained and may reject hard cases.
- InsightFace model behavior and thresholds should be calibrated on your data.

## Licensing / Compliance Notes
- **InsightFace** and underlying models may have separate licenses and usage constraints. Validate legal/commercial suitability before production.
- PaddleOCR and PaddlePaddle licenses should also be reviewed for your deployment context.

## CPU vs Optional GPU
- Pipeline is CPU-first.
- Optional GPU flags exist in config for OCR and face modules.
- If GPU providers are unavailable, CPU path remains functional.

# Run Report

## Environment
- Date (UTC): 2026-03-29
- OS: containerized Linux environment
- Python: system Python 3.12.x
- Network/package install status: pip install blocked by proxy (403)

## Models used
- OCR: PaddleOCR (configured, runtime import blocked here due missing dependencies).
- Face: InsightFace `buffalo_l` model pack (configured, runtime not executed in this environment).

## External solutions adopted
1. OpenCV USAC/MAGSAC homography estimation path.
2. OpenCV ECC registration refinement stage.
3. PaddleOCR rotation candidate sweep (0/90/180/270) for field OCR robustness.
4. InsightFace initialization hardening with explicit model-pack configuration and explicit failure reasons.

## Test set used
- Searched repository for ID images with extensions (`jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`).
- Result: no ID images available in repo working tree for end-to-end evaluation.

## Commands executed and evidence
1. Repository audit:
   - `rg --files`
2. Build sanity:
   - `python -m compileall -q .` ✅
3. Attempted evaluation run:
   - `mkdir -p eval_empty && python evaluate_pipeline.py --input-dir eval_empty --template config.yaml --output-dir eval_out_empty`
   - Result: failed at import (`ModuleNotFoundError: No module named 'cv2'`) because dependencies are not installed.
4. Attempted dependency install:
   - `pip install numpy==1.26.4 opencv-python==4.10.0.84 PyYAML==6.0.2 python-json-logger==2.0.7 pytest==8.3.3`
   - Result: failed due proxy/network restrictions (403 Forbidden).

## Before vs after results
- **Measured before/after on identical ID-image set**: **not available** (no ID images in repo).
- **Measured code-level changes**:
  - Added Stage `stage_2a_ecc_refinement` into pipeline flow.
  - Added homography method traceability metric (`homography_method_used`).
  - Added OCR rotation candidate evaluation and recorded tested rotations in stage metrics.
  - Added explicit face model init failure reason and configurable InsightFace model pack.

## Per-field extraction results
- Not measurable in this environment due:
  1) no ID images available, and
  2) missing runtime dependencies (`cv2`, `numpy`, `yaml`) caused by blocked package installation.

## Rejection reasons
- Runtime execution blocked by missing dependencies and no test images.
- All failures are explicit; no silent fallback was used.

## Failures that remain
- End-to-end runtime and quantitative comparison remain unexecuted in this environment.
- OCR/face model performance on actual Egyptian IDs remains unmeasured until dependencies and dataset are available.

## What is measured vs unknown
- **Measured**: code compiles (`compileall`), configuration and stage orchestration changes landed.
- **Unknown**: real-world accuracy uplift on detection/OCR/face matching due lack of runnable dependencies + lack of image dataset.

## Exact commands to run (once dependencies and data are available)
```bash
pip install -r requirements.txt
python run_pipeline.py --scene path/to/scene.jpg --template path/to/template.jpg --output-dir out
python run_pipeline.py --scene path/to/scene.jpg --template path/to/template.jpg --selfie path/to/selfie.jpg --output-dir out
python evaluate_pipeline.py --input-dir path/to/images --template path/to/template.jpg --labels path/to/labels.json --output-dir eval_out
pytest -q
python smoke_test.py
```

## Dependency changes in this update
- No new dependency was added in this pass.
- Configuration updated for:
  - homography method selection,
  - ECC refinement settings,
  - OCR rotation sweep,
  - explicit InsightFace model pack.

## Template placement and calibration notes
- Existing template-relative regions remain in `config.yaml` and are still the geometric source of truth.
- ECC refinement was inserted before extraction to reduce residual drift relative to those regions.

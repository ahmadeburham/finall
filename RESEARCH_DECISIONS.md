# Research Decisions for Egyptian ID Pipeline Hardening

This document records issue-by-issue research and implementation choices using primary sources.

## A) Excessive input rejection

- **Observed failure in current pipeline**
  - Card detection relies on strict ORB thresholds and a single homography estimation strategy.
  - This can reject difficult captures where correspondences are noisy.
- **Candidate external solutions found**
  1. OpenCV robust estimators for homography (`RANSAC`, USAC family including MAGSAC).
  2. Deep document detectors (DocTR DBNet, PP-Structure pipelines).
- **Source links**
  - OpenCV homography API: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
  - OpenCV USAC tutorial: https://docs.opencv.org/4.x/de/d3e/tutorial_usac.html
- **Validation / benchmark / evidence summary**
  - OpenCV USAC framework documents robust model fitting with improved outlier handling and modern samplers/scoring.
- **Chosen solution**
  - Add configurable homography method with preference for `USAC_MAGSAC` when available.
- **Why it won**
  - Canonical, officially documented, minimal integration risk with existing ORB matcher.
- **Why other options were rejected**
  - Full detector migration requires new model stacks and evaluation data not present in repo.
- **Implementation scope in this repo**
  - `idv/modules/card_detection.py` + `config.yaml`.

## B) Perspective/rotation/geometric distortion hurting OCR

- **Observed failure in current pipeline**
  - Only one-shot perspective warp after homography; no post-warp registration refinement.
- **Candidate external solutions found**
  1. OpenCV ECC image registration (`findTransformECC`).
  2. Learned document unwarping models from PaddleX / research repos.
- **Source links**
  - OpenCV ECC docs: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
  - ECC sample usage: https://github.com/opencv/opencv/blob/4.x/samples/cpp/image_alignment.cpp
- **Validation / benchmark / evidence summary**
  - ECC is a canonical OpenCV implementation for maximizing alignment correlation.
- **Chosen solution**
  - Add ECC refinement stage (`stage_2a_ecc_refinement`) between homography warp and post-warp verification.
- **Why it won**
  - Official implementation, deterministic, lightweight, directly compatible with current template-based alignment.
- **Why other options were rejected**
  - Learned unwarping requires additional models and reproducible evaluation dataset not available in repo.
- **Implementation scope in this repo**
  - New module `idv/modules/rectification.py` + orchestration updates in `run_pipeline.py` + `config.yaml`.

## C) Arabic OCR quality weakness

- **Observed failure in current pipeline**
  - OCR ran with one orientation assumption per preprocessed crop.
- **Candidate external solutions found**
  1. PaddleOCR Arabic recognition models and angle classification.
  2. Secondary OCR stack replacement (e.g., TrOCR fine-tunes, docTR).
- **Source links**
  - PaddleOCR official repo: https://github.com/PaddlePaddle/PaddleOCR
  - PaddleOCR model list including Arabic rec model: https://paddlepaddle.github.io/PaddleOCR/v3.0.0/en/version2.x/ppocr/model_list.html
- **Validation / benchmark / evidence summary**
  - PaddleOCR officially supports Arabic recognition models and angle classification.
- **Chosen solution**
  - Keep PaddleOCR stack and add controlled 0/90/180/270 rotation sweep per field for candidate selection.
- **Why it won**
  - Faithful to PaddleOCR inference; improves robustness to residual orientation errors.
- **Why other options were rejected**
  - No in-repo benchmark set available to justify full OCR stack replacement.
- **Implementation scope in this repo**
  - `idv/modules/ocr_engine.py` + `config.yaml` (`ocr.try_rotations`).

## D) Arabic-Indic digits / ID number / birth date weakness

- **Observed failure in current pipeline**
  - Numeric outputs can degrade when orientation is off or OCR confidence is weak.
- **Candidate external solutions found**
  1. Continue field-specific OCR + strict normalization/validation.
  2. Train dedicated numeric recognizer for Egyptian IDs.
- **Source links**
  - PaddleOCR recognition pipeline docs/repo: https://github.com/PaddlePaddle/PaddleOCR
- **Validation / benchmark / evidence summary**
  - Existing normalization logic already handles Arabic-Indic digits; improvements should target OCR candidate quality first.
- **Chosen solution**
  - Use rotated OCR candidates for numeric fields and keep conservative structural validators.
- **Why it won**
  - Minimal deviation and no unvalidated model substitution.
- **Why other options were rejected**
  - No trustworthy public Egyptian-ID-specific digit model with direct drop-in integration identified during this pass.
- **Implementation scope in this repo**
  - `idv/modules/ocr_engine.py`, existing `idv/modules/normalization.py` reused.

## E) Face matching fragility

- **Observed failure in current pipeline**
  - Face model init failures were not fully explicit at API boundary for consumers.
- **Candidate external solutions found**
  1. InsightFace `FaceAnalysis` with model packs (e.g., `buffalo_l`).
  2. Migrate to alternate face stack (e.g., ArcFace implementations outside InsightFace wrappers).
- **Source links**
  - InsightFace official repo: https://github.com/deepinsight/insightface
- **Validation / benchmark / evidence summary**
  - InsightFace remains a validated and widely used face analysis stack with packaged models.
- **Chosen solution**
  - Keep InsightFace and harden initialization/failure reasons; make model pack explicit via config.
- **Why it won**
  - Preserves proven stack while making deployment failures explicit and diagnosable.
- **Why other options were rejected**
  - Migration would increase integration risk without in-repo comparative evidence.
- **Implementation scope in this repo**
  - `idv/modules/face_verification.py` + `config.yaml` (`face.model_pack`).

## F) Crop region misalignment after alignment

- **Observed failure in current pipeline**
  - Fixed template boxes can drift when alignment is slightly off.
- **Candidate external solutions found**
  1. ECC post-warp refinement.
  2. Learned keypoint detector for document landmarks.
- **Source links**
  - OpenCV ECC docs: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
- **Validation / benchmark / evidence summary**
  - ECC directly minimizes template/crop registration error.
- **Chosen solution**
  - Apply ECC prior to extraction to reduce region drift.
- **Why it won**
  - No extra model dependencies; canonical algorithm.
- **Why other options were rejected**
  - Landmark models require training data and integration not currently available.
- **Implementation scope in this repo**
  - Same as issue B (rectification module + stage integration).

## G) Weak preprocessing for noisy low-contrast text

- **Observed failure in current pipeline**
  - Preprocessing is basic and not benchmarked per corpus in this repo.
- **Candidate external solutions found**
  1. Keep controlled variants and rely on OCR candidate selection.
  2. Introduce external binarization/deblurring networks.
- **Source links**
  - PaddleOCR repo (practical preprocessing with model robustness): https://github.com/PaddlePaddle/PaddleOCR
- **Validation / benchmark / evidence summary**
  - Without a local benchmark image set, adding unvalidated external restoration networks risks regressions.
- **Chosen solution**
  - No major preprocessing-model substitution in this pass; improved OCR selection strategy instead.
- **Why it won**
  - Avoids untested complexity; remains faithful to validated OCR stack.
- **Why other options were rejected**
  - Lacked trustworthy drop-in evidence for this repo’s exact data.
- **Implementation scope in this repo**
  - `idv/modules/ocr_engine.py` candidate selection enhancement.

## H) Missing production-grade validation steps

- **Observed failure in current pipeline**
  - Missing explicit stage for registration refinement and method traceability.
- **Candidate external solutions found**
  1. Add ECC stage and richer metrics logging.
  2. Full recapture quality frameworks from mobile SDK vendors.
- **Source links**
  - OpenCV ECC docs: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
  - OpenCV USAC tutorial: https://docs.opencv.org/4.x/de/d3e/tutorial_usac.html
- **Validation / benchmark / evidence summary**
  - Canonical methods are available now and provide deterministic evidence in stage metrics.
- **Chosen solution**
  - Added explicit stage for ECC refinement and homography method reporting.
- **Why it won**
  - Immediate robustness gain with transparent metrics/failure reporting.
- **Why other options were rejected**
  - Vendor SDK integrations require external credentials/licensing not present in repo.
- **Implementation scope in this repo**
  - `idv/modules/rectification.py`, `idv/modules/card_detection.py`, `run_pipeline.py`, `config.yaml`.

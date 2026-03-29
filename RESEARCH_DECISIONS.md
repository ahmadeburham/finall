# RESEARCH_DECISIONS

## Audit outcome and scope gate

- **Repository state audited on 2026-03-29 (UTC):** the repository contains only `.gitkeep` and no existing pipeline code, configs, model wiring, test assets, or baseline evaluation scripts.
- **Implication:** full in-place replacement of pipeline components is blocked because there is no implementation surface and no input dataset to run before/after comparisons.
- **Policy followed:** no invented implementation was added as a substitute for missing code/assets.

---

## Issue A — Over-rejection of input photos

### Observed failure in current pipeline
- No pipeline implementation exists in this repository; therefore over-rejection cannot be reproduced from code.

### Candidate external solutions found
1. **Google ML Kit Document Scanner API** (official product docs; automatic capture, edge detection, rotation, in-flow cleanup controls).
   - https://developers.google.com/ml-kit/vision/doc-scanner
2. **Dynamsoft Mobile Web Capture / Capture Vision** (production capture pipeline with auto-capture + quality controls; vendor docs).
   - https://www.dynamsoft.com/blog/insights/mobile-document-scanning-errors-dynamsoft/
   - https://www.dynamsoft.com/Documents/using-cameras-for-document-scanning.pdf

### Validation / benchmark / evidence summary
- ML Kit docs explicitly enumerate production capture controls (auto capture, edge detection, auto-rotation, shadow/stain removal).
- Dynamsoft materials describe quality-oriented capture controls in real-world mobile scanning contexts.

### Chosen solution
- **No implementation yet (blocked by missing codebase and UI/capture entrypoint).**
- Preferred future integration order: (1) ML Kit scanner for Android capture apps when mobile-native path exists; (2) Dynamsoft when enterprise SDK licensing is acceptable.

### Why it won / why others rejected
- Chosen candidates are official product docs with concrete implementation APIs and production capture features.
- Rejected weaker options: ad-hoc single-threshold blur rules without end-to-end capture UX/quality flow.

### Implementation scope in this repo
- **Blocked** pending actual capture module and runnable app code.

---

## Issue B — Perspective/rotation/distortion harming OCR

### Observed failure in current pipeline
- Not reproducible (no rectification code present).

### Candidate external solutions found
1. **DocTr (Document Image Transformer for geometric unwarping + illumination correction, ACM MM 2021 official code).**
   - Paper code repo: https://github.com/fh2019ustc/DocTr
2. **Dewarping by displacement flow FCN** (peer-reviewed dewarping approach).
   - https://arxiv.org/abs/2104.06815

### Validation / benchmark / evidence summary
- DocTr repository is official implementation aligned with published method and doc dewarping benchmark references.
- FCN displacement-flow method reports SOTA-style dewarping results in paper context.

### Chosen solution
- **DocTr official implementation** is preferred for integration fidelity.
- **Not implemented yet** due absent existing pipeline and no sample inputs for verification.

### Why it won / why others rejected
- DocTr provides direct code and stated target task overlap (document geometric unwarping + illumination correction).
- Pure homography-only baseline is insufficient against curved/non-planar distortions.

### Implementation scope in this repo
- **Blocked** until there is a runnable pre-OCR stage and test corpus.

---

## Issue C — Weak Arabic OCR quality

### Observed failure in current pipeline
- No OCR module exists in repo, so failure cannot be measured locally.

### Candidate external solutions found
1. **PaddleOCR official repository** (active canonical OCR toolkit, multilingual support incl. Arabic).
   - https://github.com/PaddlePaddle/PaddleOCR
2. **PP-OCRv5 multilingual recognition docs** (Arabic dataset and multilingual model notes).
   - https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html

### Validation / benchmark / evidence summary
- PaddleOCR is canonical, actively maintained, and includes multilingual model releases with Arabic coverage.
- Official docs provide model family details and dataset notes for multilingual recognition.

### Chosen solution
- **PaddleOCR official stack** (detection + Arabic-capable recognition model) for faithful integration.
- **Not implemented yet** due missing pipeline and absent baseline/eval images.

### Why it won / why others rejected
- Official repository and docs provide strongest reproducibility and production fit.
- Third-party wrappers/forks rejected as primary because canonical implementation is directly available.

### Implementation scope in this repo
- **Blocked** by missing code and assets.

---

## Issue D — Arabic-Indic digits / ID number / birth date extraction weakness

### Observed failure in current pipeline
- Not measurable because no extraction code or field templates exist.

### Candidate external solutions found
1. **PaddleOCR multilingual recognition with field-level cropping** (official stack; enables separate digit-field OCR path).
   - https://github.com/PaddlePaddle/PaddleOCR
2. **Egyptian National ID checksum/date validation rules** (domain constraints to validate OCR outputs).
   - (Requires official government or formally documented source; not yet added due lack of in-repo validator skeleton.)

### Validation / benchmark / evidence summary
- Official OCR model support exists; field-specific OCR design is implementable once ROI mapping is defined.
- Hard validation for ID number/date requires authoritative spec references before coding.

### Chosen solution
- **Deferred implementation** until authoritative Egyptian ID rule source is added and current extractor code exists.

### Why it won / why others rejected
- Prevents invented heuristic parsing and unverified checksum assumptions.

### Implementation scope in this repo
- **Blocked**.

---

## Issue E — Fragile/broken face matching

### Observed failure in current pipeline
- No face module or model files are present.

### Candidate external solutions found
1. **InsightFace (official repo), ArcFace backbone/loss family.**
   - https://github.com/deepinsight/insightface
2. **ArcFace paper (CVPR 2019).**
   - https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf
3. **RetinaFace paper (face detection/alignment synergy with ArcFace).**
   - https://arxiv.org/abs/1905.00641

### Validation / benchmark / evidence summary
- ArcFace and RetinaFace are widely cited canonical methods with official implementations and benchmarked recognition/detection performance.

### Chosen solution
- **InsightFace pipeline (RetinaFace/ArcFace-based stack)** for future integration.
- **Not implemented yet** due missing inference pipeline and no test pair assets.

### Why it won / why others rejected
- Strong canonical evidence and mature ecosystem.
- Weaker or outdated face-match stacks without robust alignment were not selected.

### Implementation scope in this repo
- **Blocked**.

---

## Issue F — Misaligned crop regions after alignment

### Observed failure in current pipeline
- No field-template/crop code present.

### Candidate external solutions found
1. **Keypoint/layout-based ROI mapping after rectification** using official OCR detector geometry from PaddleOCR.
   - https://github.com/PaddlePaddle/PaddleOCR
2. **KIE/document parsing approaches in docTR** for structure-aware extraction.
   - https://github.com/mindee/doctr

### Validation / benchmark / evidence summary
- Both ecosystems provide production-used detection geometry and structured extraction primitives.

### Chosen solution
- **Deferred** until base rectification and template calibration assets are available.

### Why it won / why others rejected
- Avoids inventing fixed coordinates without real card templates and alignment behavior.

### Implementation scope in this repo
- **Blocked**.

---

## Issue G — Weak preprocessing for noisy/low-contrast text

### Observed failure in current pipeline
- No preprocessing module found.

### Candidate external solutions found
1. **DocTr (illumination correction included in design objective).**
   - https://github.com/fh2019ustc/DocTr
2. **DocEnTR (document enhancement transformer, ICPR 2022).**
   - https://github.com/dali92002/DocEnTR

### Validation / benchmark / evidence summary
- Both are paper-backed implementations for degraded document enhancement/rectification scenarios.

### Chosen solution
- **DocTr-first** if geometric and illumination correction are jointly required in card pipeline.
- **Deferred implementation** due missing code and test assets.

### Why it won / why others rejected
- End-to-end published methods preferred over ad-hoc local filtering pipelines.

### Implementation scope in this repo
- **Blocked**.

---

## Issue H — Missing production-grade validation steps

### Observed failure in current pipeline
- Entire validation subsystem absent.

### Candidate external solutions found
1. **Capture-stage quality gates from official scanner SDK docs (ML Kit / enterprise capture SDKs).**
2. **Model-stage structured status outputs used by PaddleOCR/InsightFace ecosystems (detector success, recognizer success, confidence reporting).**

### Validation / benchmark / evidence summary
- Official SDK/docs provide concrete quality-oriented workflow elements.
- Canonical OCR/face stacks expose per-stage outputs that can be logged deterministically.

### Chosen solution
- **Deferred implementation** until baseline app/pipeline exists.

### Why it won / why others rejected
- Cannot bolt production validation on non-existent pipeline without inventing interfaces.

### Implementation scope in this repo
- **Blocked**.

---

## Ranked solution shortlist (for future in-repo implementation)

1. **PaddleOCR official (Arabic-capable OCR stack)** — strongest immediate OCR backbone fit.
2. **DocTr official (document unwarp + illumination correction)** — strongest geometry/preprocess upgrade.
3. **InsightFace (RetinaFace + ArcFace)** — strongest face match replacement.
4. **ML Kit / enterprise capture SDK quality flow** — strongest capture acceptance gate architecture.

## What was intentionally not done

- No fabricated baseline metrics.
- No synthetic “after” numbers.
- No placeholder pipeline implementation in an empty repo.
- No undocumented substitutions.

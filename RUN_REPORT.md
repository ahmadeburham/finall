# RUN_REPORT

## Environment
- Date (UTC): 2026-03-29
- OS/container: provided execution container
- Repository: `/workspace/finall`
- Git head at audit start: `cf7e751` (Initialize repository)

## Models used
- None executed.
- Reason: repository does not contain runnable pipeline code, model integration code, or test assets.

## External solutions adopted
- No runtime adoption yet; only research and decision documentation recorded in `RESEARCH_DECISIONS.md`.

## Test set used
- None available in repository.
- No ID images, selfie images, templates, or benchmark manifests were found.

## Before vs after results
- Not measurable.
- Baseline and updated pipeline runs are impossible because no pipeline implementation exists in this repository.

## Per-field extraction results
- Not available (no runs).

## Rejection reasons
- N/A (no runtime pipeline executed).

## Failures that remain
1. Missing implementation of the entire ID pipeline.
2. Missing dataset/test assets for reproducible evaluation.
3. Missing baseline execution harness and metrics collection scripts.

## What is measured vs unknown
### Measured
- Repository content inspection and git history inspection.

### Unknown
- Real rejection/acceptance rates.
- OCR quality and digit extraction quality.
- Face matching reliability.
- Geometric rectification quality.

## Exact commands run
```bash
pwd && rg --files -g 'AGENTS.md'
find / -name AGENTS.md 2>/dev/null | head -n 50
cd /workspace/finall && rg --files | head -n 200
cd /workspace/finall && ls -la
cd /workspace/finall && git status --short && git log --oneline -n 5
```

## Dependencies added or changed
- None.

## Config notes for template placement and region calibration
- Not applicable yet; no template system exists in repo.

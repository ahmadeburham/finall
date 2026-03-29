"""Batch evaluation CLI for Egyptian ID verification pipeline."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from idv.modules.evaluation import aggregate
from idv.modules.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluation for ID pipeline")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--template", required=True, type=Path)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    per_image_dir = args.output_dir / "per_image"
    ensure_dir(per_image_dir)

    labels = None
    if args.labels and args.labels.exists():
        labels = json.loads(args.labels.read_text(encoding="utf-8"))

    results = []
    for img_path in sorted(args.input_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        out_dir = per_image_dir / img_path.stem
        ensure_dir(out_dir)
        cmd = [
            "python",
            "run_pipeline.py",
            "--scene",
            str(img_path),
            "--template",
            str(args.template),
            "--config",
            str(args.config),
            "--output-dir",
            str(out_dir),
        ]
        subprocess.run(cmd, check=False)
        run_dirs = sorted([p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
        if not run_dirs:
            continue
        result_path = run_dirs[-1] / "result.json"
        if result_path.exists():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))

    write_json(args.output_dir / "per_image_results.json", {"results": results})
    agg = aggregate(results, labels)
    write_json(args.output_dir / "aggregate_metrics.json", agg)
    print(str(args.output_dir / "aggregate_metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

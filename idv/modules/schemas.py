"""Dataclasses and typed contracts for pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    """Strict contract for every stage output."""

    stage_name: str
    success: bool
    error_code: str = ""
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    next_stage_allowed: bool = True
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunContext:
    """Global context for a single execution run."""

    run_id: str
    output_dir: Path
    run_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    metrics_dir: Path
    config_path: Path
    config: Dict[str, Any]
    scene_path: Path
    template_path: Path
    selfie_path: Optional[Path]


@dataclass
class OCRFieldResult:
    """Per-field OCR result schema."""

    raw_text: str
    normalized_text: str
    confidence: float
    alternate_candidates: List[Dict[str, Any]]
    validation_passed: bool
    failure_reason: str
    preprocessing_variant_used: str
    detector_metadata: Dict[str, Any]
    recognizer_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

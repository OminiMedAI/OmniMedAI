"""Machine-readable workflow manifests for study reporting."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass
class WorkflowManifest:
    """Record the parameters and resources used to produce study results."""

    study_name: str
    cohort_name: str
    task: str
    data_version: str = "unspecified"
    code_url: Optional[str] = None
    data_url: Optional[str] = None
    ethics_approval: Optional[str] = None
    created_at: str = field(default_factory=_utc_timestamp)
    acquisition: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    segmentation: Dict[str, Any] = field(default_factory=dict)
    feature_extraction: Dict[str, Any] = field(default_factory=dict)
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    harmonization: Dict[str, Any] = field(default_factory=dict)
    model_development: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    omics: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    REQUIRED_FIELDS = (
        "study_name",
        "cohort_name",
        "task",
        "code_url",
        "data_url",
        "acquisition.sequences",
        "acquisition.scanners",
        "reconstruction.algorithm",
        "reconstruction.parameters",
        "preprocessing.resampling",
        "preprocessing.normalization",
        "segmentation.reader_protocol",
        "feature_extraction.extractor",
        "feature_extraction.parameters",
        "model_development.split_level",
        "model_development.feature_selection_order",
        "model_development.hyperparameters",
        "validation.external_validation",
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save_json(self, path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json() + "\n", encoding="utf-8")
        return output_path

    def missing_required_fields(self) -> List[str]:
        data = self.to_dict()
        missing = []
        for dotted_path in self.REQUIRED_FIELDS:
            value = data
            for part in dotted_path.split("."):
                if not isinstance(value, dict) or part not in value:
                    value = None
                    break
                value = value[part]
            if value in (None, "", [], {}):
                missing.append(dotted_path)
        return missing

    def completeness_report(self) -> Dict[str, Any]:
        missing = self.missing_required_fields()
        return {
            "complete": len(missing) == 0,
            "missing_required_fields": missing,
        }

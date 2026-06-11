"""Configuration settings for multimodal feature fusion."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FusionConfig:
    """Configuration for aligning and merging multimodal feature tables."""

    id_column: str = "patient_id"
    join_strategy: str = "inner"
    modality_prefixes: Dict[str, str] = field(default_factory=dict)
    drop_columns: List[str] = field(default_factory=list)
    fill_missing: Optional[str] = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        valid_joins = {"inner", "left", "right", "outer"}
        if self.join_strategy not in valid_joins:
            raise ValueError(f"join_strategy must be one of {sorted(valid_joins)}")

        valid_fill = {None, "zero", "mean", "median"}
        if self.fill_missing not in valid_fill:
            raise ValueError("fill_missing must be None, 'zero', 'mean', or 'median'")

    def to_dict(self) -> Dict:
        return {
            "id_column": self.id_column,
            "join_strategy": self.join_strategy,
            "modality_prefixes": self.modality_prefixes,
            "drop_columns": self.drop_columns,
            "fill_missing": self.fill_missing,
        }


PRESET_CONFIGS = {
    "strict": FusionConfig(join_strategy="inner", fill_missing=None),
    "research": FusionConfig(join_strategy="outer", fill_missing="median"),
    "clinical": FusionConfig(join_strategy="left", fill_missing="median"),
}

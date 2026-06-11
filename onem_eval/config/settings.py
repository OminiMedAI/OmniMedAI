"""Configuration settings for model evaluation."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""

    task: str = "classification"
    average: str = "weighted"
    positive_label: int = 1
    include_confusion_matrix: bool = True
    include_auc: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if self.average not in {"binary", "micro", "macro", "weighted"}:
            raise ValueError("average must be binary, micro, macro, or weighted")

    def to_dict(self) -> Dict:
        return {
            "task": self.task,
            "average": self.average,
            "positive_label": self.positive_label,
            "include_confusion_matrix": self.include_confusion_matrix,
            "include_auc": self.include_auc,
        }


PRESET_CONFIGS = {
    "binary": EvaluationConfig(average="binary"),
    "multiclass": EvaluationConfig(average="weighted", include_auc=False),
    "research_report": EvaluationConfig(average="weighted", include_auc=True),
}

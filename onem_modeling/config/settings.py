"""Configuration settings for tabular modeling."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelingConfig:
    """Configuration for feature-table modeling."""

    task: str = "classification"
    model_type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    scale_features: bool = True
    feature_selection: str = "none"
    n_features: int = 50
    cv_folds: int = 5
    class_weight: str = "balanced"

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if self.model_type not in {"random_forest", "svm", "logistic_regression", "linear_regression"}:
            raise ValueError("unsupported model_type")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.n_features < 1:
            raise ValueError("n_features must be positive")

    def to_dict(self) -> Dict:
        return {
            "task": self.task,
            "model_type": self.model_type,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "scale_features": self.scale_features,
            "feature_selection": self.feature_selection,
            "n_features": self.n_features,
            "cv_folds": self.cv_folds,
            "class_weight": self.class_weight,
        }


PRESET_CONFIGS = {
    "radiomics_baseline": ModelingConfig(model_type="random_forest", scale_features=True),
    "small_sample": ModelingConfig(model_type="svm", test_size=0.25, scale_features=True),
    "interpretable": ModelingConfig(model_type="logistic_regression", scale_features=True),
}

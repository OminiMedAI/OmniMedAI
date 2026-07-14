"""Composable radiomics feature-selection stages."""

from dataclasses import dataclass, replace
from typing import Iterable, List, Optional


def _require_dependencies():
    try:
        import numpy as np
        import pandas as pd
        from scipy.stats import mannwhitneyu, spearmanr
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        from sklearn.linear_model import LassoCV, LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "Feature selection requires numpy, pandas, scipy, and scikit-learn"
        ) from exc
    return locals()


@dataclass
class FeatureSelectionConfig:
    """Configuration for a sequential radiomics selection procedure."""

    task: str = "classification"
    univariate_p_threshold: Optional[float] = 0.05
    correlation_threshold: Optional[float] = 0.9
    mrmr_features: Optional[int] = 50
    lasso_cv_folds: int = 5
    random_state: int = 42

    def validate(self):
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        if self.univariate_p_threshold is not None and not 0 < self.univariate_p_threshold <= 1:
            raise ValueError("univariate_p_threshold must be in (0, 1]")
        if self.correlation_threshold is not None and not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be in (0, 1]")
        if self.mrmr_features is not None and self.mrmr_features < 1:
            raise ValueError("mrmr_features must be positive")
        if self.lasso_cv_folds < 2:
            raise ValueError("lasso_cv_folds must be at least 2")


class SequentialRadiomicsSelector:
    """Fit a transparent univariate/correlation/mRMR/LASSO sequence."""

    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        self.config = config or FeatureSelectionConfig()
        self.config.validate()
        self.selected_features_: List[str] = []
        self.stage_features_ = {}
        self.coefficients_ = {}
        self.scaler_ = None
        self.model_ = None
        self.feature_names_in_: List[str] = []

    def fit(self, x, y):
        deps = _require_dependencies()
        pd = deps["pd"]
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        numeric = x.select_dtypes(include="number").copy()
        if numeric.empty:
            raise ValueError("No numeric features were provided")
        features = list(numeric.columns)
        self.feature_names_in_ = list(features)

        if self.config.univariate_p_threshold is not None:
            features = self._univariate_filter(numeric[features], y, deps)
        self.stage_features_["univariate"] = list(features)
        if not features:
            raise ValueError("Univariate filtering removed every feature")

        if self.config.correlation_threshold is not None:
            features = self._correlation_filter(
                numeric[features], self.config.correlation_threshold
            )
        self.stage_features_["correlation"] = list(features)

        if self.config.mrmr_features is not None:
            features = self._mrmr_select(
                numeric[features],
                y,
                min(self.config.mrmr_features, len(features)),
                deps,
            )
        self.stage_features_["mrmr"] = list(features)

        self.scaler_ = deps["StandardScaler"]()
        scaled = self.scaler_.fit_transform(numeric[features])
        if self.config.task == "classification":
            self.model_ = deps["LogisticRegressionCV"](
                Cs=20,
                cv=self.config.lasso_cv_folds,
                penalty="l1",
                solver="liblinear",
                scoring="roc_auc",
                class_weight="balanced",
                random_state=self.config.random_state,
                max_iter=5000,
            )
        else:
            self.model_ = deps["LassoCV"](
                cv=self.config.lasso_cv_folds,
                random_state=self.config.random_state,
                max_iter=10000,
            )
        self.model_.fit(scaled, y)
        coefficients = self.model_.coef_
        if getattr(coefficients, "ndim", 1) > 1:
            coefficients = coefficients[0]
        self.coefficients_ = {
            feature: float(coefficient)
            for feature, coefficient in zip(features, coefficients)
        }
        self.selected_features_ = [
            feature for feature in features if self.coefficients_[feature] != 0
        ]
        if not self.selected_features_:
            strongest = max(features, key=lambda name: abs(self.coefficients_[name]))
            self.selected_features_ = [strongest]
        self.stage_features_["lasso"] = list(self.selected_features_)
        return self

    def transform(self, x):
        if not self.selected_features_:
            raise RuntimeError("Selector has not been fitted")
        return x.loc[:, self.selected_features_].copy()

    def fit_transform(self, x, y):
        return self.fit(x, y).transform(x)

    def report(self):
        return {
            "stage_features": self.stage_features_,
            "selected_features": self.selected_features_,
            "coefficients": self.coefficients_,
            "config": self.config.__dict__.copy(),
        }

    def get_support(self):
        if not self.feature_names_in_:
            raise RuntimeError("Selector has not been fitted")
        selected = set(self.selected_features_)
        return [feature in selected for feature in self.feature_names_in_]

    def get_params(self, deep=True):
        return {"config": self.config}

    def set_params(self, **params):
        if "config" in params:
            self.config = params["config"]
            self.config.validate()
        return self

    def _univariate_filter(self, x, y, deps):
        np = deps["np"]
        y_array = np.asarray(y)
        kept = []
        if self.config.task == "classification":
            classes = np.unique(y_array)
            if len(classes) != 2:
                raise ValueError("Mann-Whitney filtering currently requires binary labels")
            for feature in x.columns:
                left = x.loc[y_array == classes[0], feature].dropna()
                right = x.loc[y_array == classes[1], feature].dropna()
                if len(left) and len(right):
                    _, p_value = deps["mannwhitneyu"](left, right, alternative="two-sided")
                    if p_value <= self.config.univariate_p_threshold:
                        kept.append(feature)
        else:
            for feature in x.columns:
                valid = x[feature].notna()
                if valid.sum() > 2:
                    _, p_value = deps["spearmanr"](x.loc[valid, feature], y_array[valid])
                    if p_value <= self.config.univariate_p_threshold:
                        kept.append(feature)
        return kept

    def _correlation_filter(self, x, threshold):
        correlation = x.corr(method="spearman").abs()
        kept = []
        for feature in x.columns:
            if all(correlation.loc[feature, previous] < threshold for previous in kept):
                kept.append(feature)
        return kept

    def _mrmr_select(self, x, y, n_features, deps):
        np = deps["np"]
        values = x.fillna(x.median()).to_numpy()
        if self.config.task == "classification":
            relevance = deps["mutual_info_classif"](
                values, y, random_state=self.config.random_state
            )
        else:
            relevance = deps["mutual_info_regression"](
                values, y, random_state=self.config.random_state
            )
        relevance = dict(zip(x.columns, relevance))
        selected = []
        remaining = list(x.columns)
        while remaining and len(selected) < n_features:
            scores = {}
            for candidate in remaining:
                if selected:
                    redundancy = np.mean(
                        [
                            abs(x[candidate].corr(x[chosen], method="spearman"))
                            for chosen in selected
                        ]
                    )
                else:
                    redundancy = 0.0
                scores[candidate] = relevance[candidate] - redundancy
            best = max(scores, key=scores.get)
            selected.append(best)
            remaining.remove(best)
        return selected


def repeated_seed_feature_selection(
    x,
    y,
    config: Optional[FeatureSelectionConfig] = None,
    n_repeats: int = 10,
    random_states: Optional[Iterable[int]] = None,
):
    """Repeat the complete selection sequence and summarize seed stability.

    When ``random_states`` is omitted, consecutive states beginning at the
    configuration's ``random_state`` are used. The returned report records the
    states for reproducibility even when a manuscript reports only the number
    of repeated runs.
    """
    from .validation import summarize_feature_selection_stability

    base_config = config or FeatureSelectionConfig()
    base_config.validate()
    if random_states is None:
        if n_repeats < 2:
            raise ValueError("n_repeats must be at least 2")
        states = [base_config.random_state + offset for offset in range(n_repeats)]
    else:
        states = [int(state) for state in random_states]
        if len(states) < 2:
            raise ValueError("random_states must contain at least 2 values")
        if len(states) != len(set(states)):
            raise ValueError("random_states must be unique")

    selected_by_seed = {}
    reports_by_seed = {}
    for state in states:
        selector = SequentialRadiomicsSelector(
            replace(base_config, random_state=state)
        ).fit(x, y)
        key = str(state)
        selected_by_seed[key] = list(selector.selected_features_)
        reports_by_seed[key] = selector.report()

    stability = summarize_feature_selection_stability(selected_by_seed)
    stability["comparison_scope"] = "random_seed"
    return {
        "n_runs": len(states),
        "random_states": states,
        "selected_features_by_seed": selected_by_seed,
        "reports_by_seed": reports_by_seed,
        "stability": stability,
    }

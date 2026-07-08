"""Leakage-safe nested cross-validation for patient-level feature tables."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NestedCVConfig:
    """Configuration for nested patient-level model validation."""

    task: str = "classification"
    model_type: str = "logistic_regression"
    feature_selection: str = "k_best"
    n_features: int = 20
    outer_folds: int = 5
    inner_folds: int = 4
    scoring: str = "roc_auc"
    random_state: int = 42
    param_grid: object = field(default_factory=dict)
    selection_parameters: Dict[str, object] = field(default_factory=dict)

    def validate(self):
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        if self.model_type not in {
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "svm",
            "knn",
            "naive_bayes",
            "linear_regression",
            "xgboost",
        }:
            raise ValueError("unsupported model_type")
        if self.task == "regression" and self.model_type in {"logistic_regression", "naive_bayes"}:
            raise ValueError(f"{self.model_type} is classification-only")
        if self.feature_selection not in {"none", "k_best", "l1", "radiomics_sequence"}:
            raise ValueError(
                "feature_selection must be none, k_best, l1, or radiomics_sequence"
            )
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("outer_folds and inner_folds must be at least 2")
        if self.n_features < 1:
            raise ValueError("n_features must be positive")


def _require_dependencies():
    try:
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            mean_absolute_error,
            r2_score,
            roc_auc_score,
        )
        from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC, SVR
    except ImportError as exc:
        raise ImportError(
            "Nested validation requires numpy, pandas, and scikit-learn. "
            "Install with: pip install numpy pandas scikit-learn"
        ) from exc
    return locals()


def _build_estimator(config, deps):
    if config.model_type == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost validation requires xgboost"
            ) from exc
        common = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": config.random_state,
            "n_jobs": 1,
        }
        return XGBClassifier(eval_metric="logloss", **common) if config.task == "classification" else XGBRegressor(**common)
    if config.task == "classification":
        if config.model_type == "logistic_regression":
            return deps["LogisticRegression"](
                max_iter=5000,
                class_weight="balanced",
                random_state=config.random_state,
            )
        if config.model_type == "random_forest":
            return deps["RandomForestClassifier"](
                n_estimators=500,
                class_weight="balanced",
                random_state=config.random_state,
            )
        if config.model_type == "extra_trees":
            return deps["ExtraTreesClassifier"](
                n_estimators=500,
                class_weight="balanced",
                random_state=config.random_state,
                n_jobs=1,
            )
        if config.model_type == "svm":
            return deps["SVC"](
                probability=True,
                class_weight="balanced",
                random_state=config.random_state,
            )
        if config.model_type == "knn":
            return deps["KNeighborsClassifier"]()
        if config.model_type == "naive_bayes":
            return deps["GaussianNB"]()
    else:
        if config.model_type == "linear_regression":
            return deps["LinearRegression"]()
        if config.model_type == "random_forest":
            return deps["RandomForestRegressor"](
                n_estimators=500,
                random_state=config.random_state,
            )
        if config.model_type == "extra_trees":
            return deps["ExtraTreesRegressor"](
                n_estimators=500,
                random_state=config.random_state,
                n_jobs=1,
            )
        if config.model_type == "svm":
            return deps["SVR"]()
        if config.model_type == "knn":
            return deps["KNeighborsRegressor"]()
    raise ValueError(f"{config.model_type} is incompatible with task={config.task}")


def xgboost_param_grid(preset: str = "expanded") -> Dict[str, List[object]]:
    """Return reusable XGBoost GridSearchCV parameter grids.

    ``expanded`` follows the broad reviewer-facing search space. ``compact`` is
    suitable as a default inside nested cross-validation where every candidate
    is refit in each inner fold.
    """
    if preset == "expanded":
        return {
            "model__n_estimators": [50, 80, 100, 150, 200],
            "model__max_depth": [2, 3, 4, 5, 6, 7],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__min_child_weight": [3, 5, 8, 10],
            "model__gamma": [0, 0.05, 0.1, 0.125, 0.2, 0.5],
            "model__subsample": [0.6, 0.7, 0.8],
            "model__colsample_bytree": [0.5, 0.6, 0.7, 0.8],
            "model__reg_alpha": [0, 0.05, 0.1, 0.2, 0.5],
            "model__reg_lambda": [1, 2, 3, 5, 10],
        }
    if preset == "compact":
        return {
            "model__n_estimators": [80, 150],
            "model__max_depth": [2, 3, 5],
            "model__learning_rate": [0.03, 0.08],
            "model__min_child_weight": [3, 8],
            "model__gamma": [0, 0.1],
            "model__subsample": [0.7, 0.8],
            "model__colsample_bytree": [0.6, 0.8],
            "model__reg_alpha": [0, 0.1],
            "model__reg_lambda": [1, 5],
        }
    raise ValueError("preset must be 'compact' or 'expanded'")


def model_param_grid(model_type: str, preset: str = "compact"):
    """Return reusable GridSearchCV parameter grids for supported ML models."""
    if model_type == "xgboost":
        return xgboost_param_grid(preset)

    if model_type == "logistic_regression":
        if preset == "expanded":
            return [
                {
                    "model__penalty": ["l2"],
                    "model__solver": ["lbfgs", "liblinear"],
                    "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                },
                {
                    "model__penalty": ["l1"],
                    "model__solver": ["liblinear"],
                    "model__C": [0.01, 0.1, 1.0, 10.0],
                },
            ]
        if preset == "compact":
            return {"model__C": [0.01, 0.1, 1.0, 10.0]}

    if model_type == "svm":
        if preset == "expanded":
            return [
                {"model__kernel": ["linear"], "model__C": [0.01, 0.1, 1.0, 10.0]},
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.1, 1.0, 10.0, 100.0],
                    "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                },
            ]
        if preset == "compact":
            return {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"]}

    if model_type == "random_forest":
        if preset == "expanded":
            return {
                "model__n_estimators": [100, 200, 500],
                "model__max_depth": [None, 4, 8, 12],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
            }
        if preset == "compact":
            return {
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 3, 5],
            }

    if model_type == "extra_trees":
        if preset == "expanded":
            return {
                "model__n_estimators": [100, 200, 500],
                "model__max_depth": [None, 4, 8, 12],
                "model__min_samples_leaf": [1, 3, 5, 10],
                "model__max_features": ["sqrt", "log2", None],
            }
        if preset == "compact":
            return {
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 3, 5],
            }

    if model_type == "knn":
        if preset == "expanded":
            return {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            }
        if preset == "compact":
            return {
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ["uniform", "distance"],
            }

    if model_type == "naive_bayes":
        if preset == "expanded":
            return {"model__var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]}
        if preset == "compact":
            return {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}

    if model_type == "linear_regression":
        return {}

    raise ValueError(
        "model_type must be logistic_regression, svm, random_forest, "
        "extra_trees, knn, naive_bayes, linear_regression, or xgboost"
    )


def _default_param_grid(config):
    return model_param_grid(config.model_type, "compact")


def _selected_feature_names(fitted_pipeline, feature_columns):
    selector = fitted_pipeline.named_steps.get("selector")
    if selector is None or selector == "passthrough":
        return list(feature_columns)
    support = selector.get_support()
    return [name for name, keep in zip(feature_columns, support) if keep]


def nested_patient_cross_validate(
    data,
    label_column: str,
    patient_column: str = "patient_id",
    feature_columns: Optional[List[str]] = None,
    config: Optional[NestedCVConfig] = None,
):
    """Run nested cross-validation with all learned steps inside each fold.

    ``data`` may be a pandas DataFrame or a CSV path. The returned prediction
    table can be saved and used directly for confidence intervals, calibration,
    and decision-curve analysis.
    """
    config = config or NestedCVConfig()
    config.validate()
    deps = _require_dependencies()
    np = deps["np"]
    pd = deps["pd"]

    df = pd.read_csv(data) if isinstance(data, (str, bytes)) else data.copy()
    required = {label_column, patient_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[patient_column].isna().any() or df[label_column].isna().any():
        raise ValueError("patient and label columns cannot contain missing values")

    patient_label_counts = df.groupby(patient_column)[label_column].nunique()
    if (patient_label_counts > 1).any():
        raise ValueError("Each patient must have one outcome label")

    if feature_columns is None:
        excluded = {patient_column, label_column}
        feature_columns = [
            column
            for column in df.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
        ]
    if not feature_columns:
        raise ValueError("No numeric feature columns were selected")

    x = df[feature_columns]
    y = df[label_column]
    groups = df[patient_column]
    if config.task == "classification":
        splitter = deps["StratifiedGroupKFold"](
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=config.random_state,
        )
        outer_splits = splitter.split(x, y, groups)
    else:
        splitter = deps["GroupKFold"](n_splits=config.outer_folds)
        outer_splits = splitter.split(x, y, groups)

    predictions = []
    fold_results = []
    selected_by_fold = {}
    fitted_models = []

    for fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        train_groups = groups.iloc[train_index]
        test_groups = groups.iloc[test_index]
        overlap = set(train_groups).intersection(set(test_groups))
        if overlap:
            raise RuntimeError(f"Patient leakage in fold {fold}: {sorted(overlap)[:5]}")

        steps = []
        if config.feature_selection == "radiomics_sequence":
            from .feature_selection import (
                FeatureSelectionConfig,
                SequentialRadiomicsSelector,
            )

            selector_options = dict(config.selection_parameters)
            selector_options.setdefault("task", config.task)
            selector_options.setdefault("mrmr_features", config.n_features)
            selector_options.setdefault("random_state", config.random_state)
            selector_config = FeatureSelectionConfig(**selector_options)
            steps.append(
                ("selector", SequentialRadiomicsSelector(selector_config))
            )
        steps.extend(
            [
                ("imputer", deps["SimpleImputer"](strategy="median")),
                ("scaler", deps["StandardScaler"]()),
            ]
        )
        if config.feature_selection == "k_best":
            score_func = deps["f_classif"] if config.task == "classification" else deps["f_regression"]
            k = min(config.n_features, len(feature_columns))
            steps.insert(2, ("selector", deps["SelectKBest"](score_func=score_func, k=k)))
        elif config.feature_selection == "l1":
            if config.task != "classification":
                raise ValueError("l1 feature selection currently supports classification only")
            selector_model = deps["LogisticRegression"](
                penalty="l1",
                solver="liblinear",
                class_weight="balanced",
                random_state=config.random_state,
            )
            steps.insert(2, ("selector", deps["SelectFromModel"](selector_model)))
        steps.append(("model", _build_estimator(config, deps)))
        pipeline = deps["Pipeline"](steps)

        if config.task == "classification":
            inner_splitter = deps["StratifiedGroupKFold"](
                n_splits=config.inner_folds,
                shuffle=True,
                random_state=config.random_state + fold,
            )
        else:
            inner_splitter = deps["GroupKFold"](n_splits=config.inner_folds)

        search = deps["GridSearchCV"](
            pipeline,
            config.param_grid or _default_param_grid(config),
            scoring=config.scoring,
            cv=inner_splitter,
            refit=True,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(x_train, y_train, groups=train_groups)
        fitted_models.append(search.best_estimator_)
        selected_by_fold[str(fold)] = _selected_feature_names(
            search.best_estimator_, feature_columns
        )

        y_pred = search.predict(x_test)
        fold_record = {
            "fold": fold,
            "n_train_patients": int(train_groups.nunique()),
            "n_test_patients": int(test_groups.nunique()),
            "best_params": search.best_params_,
            "inner_best_score": float(search.best_score_),
            "selected_features": selected_by_fold[str(fold)],
        }

        if config.task == "classification":
            if hasattr(search, "predict_proba"):
                y_score = search.predict_proba(x_test)[:, 1]
            else:
                y_score = search.decision_function(x_test)
            fold_record["auc"] = float(deps["roc_auc_score"](y_test, y_score))
            fold_record["accuracy"] = float(deps["accuracy_score"](y_test, y_pred))
        else:
            y_score = y_pred
            fold_record["mae"] = float(deps["mean_absolute_error"](y_test, y_pred))
            fold_record["r2"] = float(deps["r2_score"](y_test, y_pred))
        fold_results.append(fold_record)

        for row_position, row_index in enumerate(test_index):
            predictions.append(
                {
                    "row_index": int(row_index),
                    patient_column: groups.iloc[row_index],
                    "fold": fold,
                    "y_true": y_test.iloc[row_position],
                    "y_pred": y_pred[row_position],
                    "y_score": float(y_score[row_position]),
                }
            )

    prediction_table = pd.DataFrame(predictions).sort_values("row_index").reset_index(drop=True)
    metric_name = "auc" if config.task == "classification" else "mae"
    metric_values = [record[metric_name] for record in fold_results]
    return {
        "predictions": prediction_table,
        "fold_results": fold_results,
        "selected_features_by_fold": selected_by_fold,
        "fitted_models": fitted_models,
        "summary": {
            "task": config.task,
            "split_level": "patient",
            "n_rows": int(len(df)),
            "n_patients": int(df[patient_column].nunique()),
            "n_features": int(len(feature_columns)),
            f"mean_{metric_name}": float(np.mean(metric_values)),
            f"std_{metric_name}": float(np.std(metric_values, ddof=1))
            if len(metric_values) > 1
            else 0.0,
        },
    }

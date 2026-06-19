"""Survival-analysis utilities for real clinical outcome tables."""

from typing import List, Optional, Sequence


def _load_table(data):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for survival analysis") from exc
    return pd.read_csv(data) if isinstance(data, (str, bytes)) else data.copy()


def _require_lifelines():
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.utils import concordance_index
    except ImportError as exc:
        raise ImportError(
            "lifelines is required for Kaplan-Meier and Cox analysis. "
            "Install with: pip install lifelines"
        ) from exc
    return CoxPHFitter, KaplanMeierFitter, concordance_index


def validate_survival_table(
    data,
    duration_column: str,
    event_column: str,
    patient_column: str = "patient_id",
):
    """Validate a patient-level censored survival table."""
    df = _load_table(data)
    required = {duration_column, event_column, patient_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required survival columns: {missing}")
    if df[patient_column].duplicated().any():
        raise ValueError("Survival input must contain one row per patient")
    if df[list(required)].isna().any().any():
        raise ValueError("Patient, duration, and event columns cannot be missing")
    if (df[duration_column] < 0).any():
        raise ValueError("Survival durations cannot be negative")
    invalid_events = set(df[event_column].unique()).difference({0, 1, False, True})
    if invalid_events:
        raise ValueError(f"Event column must be binary; found {sorted(invalid_events)}")
    return df


def kaplan_meier_table(
    data,
    duration_column: str,
    event_column: str,
    group_column: Optional[str] = None,
    patient_column: str = "patient_id",
):
    """Return Kaplan-Meier survival estimates as a tidy DataFrame."""
    _, KaplanMeierFitter, _ = _require_lifelines()
    df = validate_survival_table(data, duration_column, event_column, patient_column)
    groups = [("all", df)] if group_column is None else list(df.groupby(group_column, dropna=False))
    tables = []
    for group, subset in groups:
        model = KaplanMeierFitter(label=str(group))
        model.fit(subset[duration_column], event_observed=subset[event_column])
        table = model.survival_function_.reset_index()
        table.columns = ["timeline", "survival_probability"]
        table["group"] = group
        table["n_patients"] = int(len(subset))
        tables.append(table)
    import pandas as pd

    return pd.concat(tables, ignore_index=True)


def number_at_risk_table(
    data,
    duration_column: str,
    event_column: str,
    time_points: Sequence[float],
    group_column: Optional[str] = None,
    patient_column: str = "patient_id",
):
    """Count patients at risk at requested time points."""
    df = validate_survival_table(data, duration_column, event_column, patient_column)
    groups = [("all", df)] if group_column is None else list(df.groupby(group_column, dropna=False))
    rows = []
    for group, subset in groups:
        for time_point in time_points:
            rows.append(
                {
                    "group": group,
                    "time": float(time_point),
                    "n_at_risk": int((subset[duration_column] >= time_point).sum()),
                }
            )
    import pandas as pd

    return pd.DataFrame(rows)


def fit_cox_model(
    data,
    duration_column: str,
    event_column: str,
    covariates: List[str],
    categorical_covariates: Optional[List[str]] = None,
    patient_column: str = "patient_id",
    penalizer: float = 0.0,
):
    """Fit a multivariable Cox model and return a publication-ready summary."""
    import pandas as pd

    CoxPHFitter, _, concordance_index = _require_lifelines()
    df = validate_survival_table(data, duration_column, event_column, patient_column)
    missing = sorted(set(covariates).difference(df.columns))
    if missing:
        raise ValueError(f"Missing Cox covariates: {missing}")
    model_data = df[[duration_column, event_column] + covariates].dropna().copy()
    categorical_covariates = categorical_covariates or []
    invalid_categorical = sorted(set(categorical_covariates).difference(covariates))
    if invalid_categorical:
        raise ValueError(
            f"Categorical covariates are not included in covariates: {invalid_categorical}"
        )
    if categorical_covariates:
        model_data = pd.get_dummies(
            model_data,
            columns=categorical_covariates,
            drop_first=True,
            dtype=float,
        )
    if len(model_data) < 5:
        raise ValueError("Too few complete cases for Cox modeling")

    model = CoxPHFitter(penalizer=penalizer)
    model.fit(model_data, duration_col=duration_column, event_col=event_column)
    summary = model.summary.reset_index().rename(
        columns={
            "covariate": "term",
            "exp(coef)": "hazard_ratio",
            "exp(coef) lower 95%": "ci_lower",
            "exp(coef) upper 95%": "ci_upper",
            "p": "p_value",
        }
    )
    risk_score = model.predict_partial_hazard(model_data)
    c_index = concordance_index(
        model_data[duration_column],
        -risk_score,
        model_data[event_column],
    )
    return {
        "model": model,
        "summary": summary,
        "concordance_index": float(c_index),
        "n_complete_cases": int(len(model_data)),
        "model_columns": [
            column
            for column in model_data.columns
            if column not in {duration_column, event_column}
        ],
    }


def time_dependent_auc(
    train_data,
    test_data,
    duration_column: str,
    event_column: str,
    risk_score_column: str,
    times: Sequence[float],
):
    """Calculate cumulative/dynamic AUC using scikit-survival."""
    try:
        import numpy as np
        import pandas as pd
        from sksurv.metrics import cumulative_dynamic_auc
        from sksurv.util import Surv
    except ImportError as exc:
        raise ImportError(
            "Time-dependent AUC requires scikit-survival. "
            "Install with: pip install scikit-survival"
        ) from exc

    train = _load_table(train_data)
    test = _load_table(test_data)
    required = {duration_column, event_column, risk_score_column}
    for name, table in (("train", train), ("test", test)):
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{name} data missing columns: {missing}")

    y_train = Surv.from_arrays(
        event=train[event_column].astype(bool),
        time=train[duration_column].astype(float),
    )
    y_test = Surv.from_arrays(
        event=test[event_column].astype(bool),
        time=test[duration_column].astype(float),
    )
    auc_values, mean_auc = cumulative_dynamic_auc(
        y_train,
        y_test,
        test[risk_score_column].to_numpy(),
        np.asarray(times, dtype=float),
    )
    return {
        "table": pd.DataFrame({"time": times, "auc": auc_values}),
        "mean_auc": float(mean_auc),
    }

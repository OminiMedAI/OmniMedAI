"""Treatment-response, interaction, propensity, and waterfall utilities."""

from typing import List, Optional


def _load_table(data):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for treatment analysis") from exc
    return pd.read_csv(data) if isinstance(data, (str, bytes)) else data.copy()


def validate_treatment_table(
    data,
    patient_column: str,
    treatment_column: str,
    outcome_column: str,
):
    """Validate one-row-per-patient treatment data."""
    df = _load_table(data)
    required = {patient_column, treatment_column, outcome_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing treatment columns: {missing}")
    if df[patient_column].duplicated().any():
        raise ValueError("Treatment input must contain one row per patient")
    if df[list(required)].isna().any().any():
        raise ValueError("Patient, treatment, and outcome columns cannot be missing")
    return df


def treatment_interaction_logistic(
    data,
    outcome_column: str,
    treatment_column: str,
    biomarker_column: str,
    covariates: Optional[List[str]] = None,
    patient_column: str = "patient_id",
):
    """Fit a logistic model containing treatment, biomarker, and interaction."""
    try:
        import numpy as np
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError(
            "Interaction analysis requires numpy and statsmodels. "
            "Install with: pip install numpy statsmodels"
        ) from exc
    df = validate_treatment_table(
        data, patient_column, treatment_column, outcome_column
    )
    covariates = covariates or []
    required = {biomarker_column}.union(covariates)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing interaction-model columns: {missing}")
    model_data = df[
        [outcome_column, treatment_column, biomarker_column] + covariates
    ].dropna()
    covariate_terms = " + ".join(covariates)
    formula = (
        f"{outcome_column} ~ {treatment_column} * {biomarker_column}"
        + (f" + {covariate_terms}" if covariate_terms else "")
    )
    model = smf.logit(formula, data=model_data).fit(disp=False)
    interaction_term = f"{treatment_column}:{biomarker_column}"
    if interaction_term not in model.params:
        reverse_term = f"{biomarker_column}:{treatment_column}"
        interaction_term = reverse_term if reverse_term in model.params else interaction_term
    return {
        "model": model,
        "summary": model.summary2().tables[1].reset_index().rename(
            columns={"index": "term"}
        ),
        "interaction_term": interaction_term,
        "interaction_odds_ratio": float(np.exp(model.params[interaction_term])),
        "interaction_p_value": float(model.pvalues[interaction_term]),
        "n_complete_cases": int(len(model_data)),
    }


def estimate_propensity_weights(
    data,
    treatment_column: str,
    covariates: List[str],
    patient_column: str = "patient_id",
    estimand: str = "ate",
    clip: float = 0.01,
):
    """Estimate inverse-probability weights from baseline covariates."""
    try:
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise ImportError(
            "Propensity weighting requires numpy, pandas, and scikit-learn"
        ) from exc
    df = _load_table(data)
    required = {patient_column, treatment_column}.union(covariates)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing propensity columns: {missing}")
    if estimand not in {"ate", "att"}:
        raise ValueError("estimand must be ate or att")
    treatment = df[treatment_column].astype(int)
    if set(treatment.unique()).difference({0, 1}):
        raise ValueError("Treatment must be coded as 0/1")

    numeric = [column for column in covariates if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [column for column in covariates if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    model = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            ("model", LogisticRegression(max_iter=5000)),
        ]
    )
    model.fit(df[covariates], treatment)
    propensity = np.clip(model.predict_proba(df[covariates])[:, 1], clip, 1 - clip)
    if estimand == "ate":
        weights = treatment / propensity + (1 - treatment) / (1 - propensity)
    else:
        weights = treatment + (1 - treatment) * propensity / (1 - propensity)
    result = df[[patient_column, treatment_column]].copy()
    result["propensity_score"] = propensity
    result["weight"] = weights
    return {"model": model, "weights": result}


def waterfall_table(
    data,
    change_column: str,
    patient_column: str = "patient_id",
    group_column: Optional[str] = None,
):
    """Sort patient-level percent changes for a waterfall plot."""
    df = _load_table(data)
    required = {patient_column, change_column}
    if group_column:
        required.add(group_column)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing waterfall columns: {missing}")
    if df[patient_column].duplicated().any():
        raise ValueError("Waterfall data must contain one row per patient")
    columns = [patient_column, change_column] + ([group_column] if group_column else [])
    result = df[columns].dropna(subset=[change_column]).copy()
    result = result.sort_values(change_column).reset_index(drop=True)
    result["plot_order"] = range(1, len(result) + 1)
    return result


def save_waterfall_plot(
    data,
    output_path,
    change_column: str,
    patient_column: str = "patient_id",
    group_column: Optional[str] = None,
    response_threshold: float = -30.0,
    progression_threshold: float = 20.0,
):
    """Save a publication-ready patient-level waterfall plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for waterfall plots") from exc
    table = waterfall_table(data, change_column, patient_column, group_column)
    colors = None
    if group_column:
        categories = list(dict.fromkeys(table[group_column].astype(str)))
        palette = plt.get_cmap("tab10")
        color_map = {name: palette(index) for index, name in enumerate(categories)}
        colors = [color_map[str(value)] for value in table[group_column]]

    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(table["plot_order"], table[change_column], color=colors, edgecolor="black", linewidth=0.4)
    axis.axhline(response_threshold, color="#0072B2", linestyle="--", linewidth=1)
    axis.axhline(progression_threshold, color="#D55E00", linestyle="--", linewidth=1)
    axis.set_xlabel("Patients")
    axis.set_ylabel("Best change from baseline (%)")
    axis.set_xlim(0, len(table) + 1)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return table

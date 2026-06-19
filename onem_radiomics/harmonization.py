"""Multi-center radiomics batch-effect assessment and harmonization."""

from typing import List, Optional


def _load_table(data):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for harmonization") from exc
    return pd.read_csv(data) if isinstance(data, (str, bytes)) else data.copy()


def batch_effect_summary(
    data,
    batch_column: str,
    feature_columns: Optional[List[str]] = None,
):
    """Summarize feature means, standard deviations, and missingness by batch."""
    import pandas as pd

    df = _load_table(data)
    if batch_column not in df.columns:
        raise ValueError(f"Missing batch column: {batch_column}")
    if feature_columns is None:
        feature_columns = [
            column
            for column in df.columns
            if column != batch_column and pd.api.types.is_numeric_dtype(df[column])
        ]
    rows = []
    for batch, subset in df.groupby(batch_column, dropna=False):
        for feature in feature_columns:
            values = pd.to_numeric(subset[feature], errors="coerce")
            rows.append(
                {
                    "batch": batch,
                    "feature": feature,
                    "n": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "missing_rate": float(values.isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def combat_harmonize(
    data,
    batch_column: str,
    feature_columns: Optional[List[str]] = None,
    categorical_covariates: Optional[List[str]] = None,
    continuous_covariates: Optional[List[str]] = None,
):
    """Apply neuroCombat to a training or analysis cohort.

    Do not combine held-out external validation data with training data when
    estimating harmonization parameters. For strict deployment-style external
    validation, use a method that supports learned train-to-test transforms.
    """
    try:
        import pandas as pd
        from neuroCombat import neuroCombat
    except ImportError as exc:
        raise ImportError(
            "ComBat harmonization requires pandas and neuroCombat. "
            "Install with: pip install pandas neuroCombat"
        ) from exc

    df = _load_table(data)
    categorical_covariates = categorical_covariates or []
    continuous_covariates = continuous_covariates or []
    required = {batch_column}.union(categorical_covariates, continuous_covariates)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing harmonization columns: {missing}")
    if feature_columns is None:
        excluded = required
        feature_columns = [
            column
            for column in df.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
        ]
    if df[feature_columns].isna().any().any():
        raise ValueError("Impute missing feature values before ComBat")

    covariate_columns = [batch_column] + categorical_covariates + continuous_covariates
    covariates = df[covariate_columns].copy()
    result = neuroCombat(
        dat=df[feature_columns].T,
        covars=covariates,
        batch_col=batch_column,
        categorical_cols=categorical_covariates or None,
        continuous_cols=continuous_covariates or None,
    )
    harmonized = df.copy()
    harmonized.loc[:, feature_columns] = result["data"].T
    return {
        "data": harmonized,
        "estimates": result.get("estimates"),
        "info": result.get("info"),
        "feature_columns": feature_columns,
    }

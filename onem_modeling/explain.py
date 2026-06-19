"""Model explanation helpers with optional SHAP support."""


def shap_feature_summary(model, x, max_samples: int = 200):
    """Return mean absolute SHAP importance and signed mean effects."""
    try:
        import numpy as np
        import pandas as pd
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP explanations require numpy, pandas, and shap"
        ) from exc

    sample = x.iloc[:max_samples] if hasattr(x, "iloc") else x[:max_samples]
    explainer = shap.Explainer(model, sample)
    explanation = explainer(sample)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[..., -1]
    feature_names = list(getattr(sample, "columns", explanation.feature_names))
    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.mean(np.abs(values), axis=0),
            "mean_signed_shap": np.mean(values, axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)

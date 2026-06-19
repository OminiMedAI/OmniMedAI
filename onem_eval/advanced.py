"""Confidence intervals, calibration, and decision-curve analysis."""

from typing import Dict, Sequence


def _require_array_frame_dependencies():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "numpy and pandas are required for advanced evaluation"
        ) from exc
    return np, pd


def _require_auc_dependencies():
    np, pd = _require_array_frame_dependencies()
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for bootstrap AUC confidence intervals"
        ) from exc
    return np, pd, roc_auc_score


def bootstrap_auc_ci(
    y_true,
    y_score,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """Estimate ROC AUC with a nonparametric bootstrap confidence interval."""
    np, _, roc_auc_score = _require_auc_dependencies()
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive")

    rng = np.random.default_rng(random_state)
    auc_values = []
    for _ in range(n_bootstraps):
        sample_index = rng.integers(0, len(y_true), len(y_true))
        sample_true = y_true[sample_index]
        if len(np.unique(sample_true)) < 2:
            continue
        auc_values.append(roc_auc_score(sample_true, y_score[sample_index]))
    if not auc_values:
        raise ValueError("Could not compute bootstrap AUC because all samples had one class")

    alpha = 1.0 - confidence_level
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "ci_lower": float(np.percentile(auc_values, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(auc_values, 100 * (1 - alpha / 2))),
        "n_bootstraps_used": int(len(auc_values)),
    }


def calibration_table(y_true, y_score, n_bins: int = 10):
    """Create observed event rates grouped by predicted-probability bin."""
    _, pd = _require_array_frame_dependencies()
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()
    df["bin"] = pd.cut(df["y_score"], bins=n_bins, include_lowest=True, duplicates="drop")
    table = df.groupby("bin", observed=True).agg(
        n=("y_true", "size"),
        mean_predicted=("y_score", "mean"),
        observed_rate=("y_true", "mean"),
    ).reset_index()
    table["absolute_error"] = (table["mean_predicted"] - table["observed_rate"]).abs()
    return table


def decision_curve(y_true, y_score, thresholds: Sequence[float] = None):
    """Compute net benefit for model, treat-all, and treat-none strategies."""
    np, pd = _require_array_frame_dependencies()
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    prevalence = y_true.mean()
    n = len(y_true)
    rows = []
    for threshold in thresholds:
        if threshold <= 0 or threshold >= 1:
            continue
        predicted_positive = y_score >= threshold
        tp = int(((predicted_positive == 1) & (y_true == 1)).sum())
        fp = int(((predicted_positive == 1) & (y_true == 0)).sum())
        weight = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "model_net_benefit": float((tp / n) - (fp / n) * weight),
                "treat_all_net_benefit": float(prevalence - (1 - prevalence) * weight),
                "treat_none_net_benefit": 0.0,
            }
        )
    return pd.DataFrame(rows)


def compare_binary_models(
    y_true,
    model_scores: Dict[str, object],
    n_bootstraps: int = 2000,
    random_state: int = 42,
):
    """Compare paired binary model scores using bootstrap AUC differences."""
    np, pd, roc_auc_score = _require_auc_dependencies()
    truth = np.asarray(y_true)
    scores = {name: np.asarray(values) for name, values in model_scores.items()}
    if len(scores) < 2:
        raise ValueError("At least two models are required")
    if any(len(values) != len(truth) for values in scores.values()):
        raise ValueError("All model scores must align with y_true")
    observed = {name: roc_auc_score(truth, values) for name, values in scores.items()}
    rng = np.random.default_rng(random_state)
    pairs = []
    names = list(scores)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            differences = []
            for _ in range(n_bootstraps):
                index = rng.integers(0, len(truth), len(truth))
                sample_truth = truth[index]
                if len(np.unique(sample_truth)) < 2:
                    continue
                differences.append(
                    roc_auc_score(sample_truth, scores[left_name][index])
                    - roc_auc_score(sample_truth, scores[right_name][index])
                )
            differences = np.asarray(differences)
            pairs.append(
                {
                    "model_1": left_name,
                    "model_2": right_name,
                    "auc_1": float(observed[left_name]),
                    "auc_2": float(observed[right_name]),
                    "auc_difference": float(observed[left_name] - observed[right_name]),
                    "ci_lower": float(np.percentile(differences, 2.5)),
                    "ci_upper": float(np.percentile(differences, 97.5)),
                    "two_sided_p_value": float(
                        min(1.0, 2 * min((differences <= 0).mean(), (differences >= 0).mean()))
                    ),
                }
            )
    return pd.DataFrame(pairs)

"""Cohort splitting, leakage checks, and feature-selection stability."""

from itertools import combinations
from typing import Dict, Mapping, Tuple


def _require_sklearn():
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for patient-level splitting. "
            "Install with: pip install scikit-learn"
        ) from exc
    return train_test_split


def assert_no_patient_overlap(*cohorts, patient_col: str = "patient_id") -> bool:
    """Raise ValueError if any patient appears in more than one cohort."""
    seen = set()
    for cohort in cohorts:
        if patient_col not in cohort.columns:
            raise ValueError(f"Missing patient column: {patient_col}")
        patients = set(cohort[patient_col].dropna().tolist())
        overlap = patients.intersection(seen)
        if overlap:
            examples = sorted(str(item) for item in list(overlap)[:5])
            raise ValueError(
                "Patient-level leakage detected between cohorts. "
                f"Examples: {', '.join(examples)}"
            )
        seen.update(patients)
    return True


def patient_level_train_test_split(
    df,
    patient_col: str = "patient_id",
    label_col: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[object, object, Dict[str, object]]:
    """Split rows by unique patient ID instead of image or ROI row."""
    if patient_col not in df.columns:
        raise ValueError(f"Missing patient column: {patient_col}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    train_test_split = _require_sklearn()
    patients = df[[patient_col]].drop_duplicates().copy()

    stratify = None
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"Missing label column: {label_col}")
        patient_labels = df.groupby(patient_col)[label_col].nunique()
        if (patient_labels > 1).any():
            raise ValueError("Each patient must have a single label for stratified splitting")
        patients[label_col] = df.groupby(patient_col)[label_col].first().reindex(
            patients[patient_col]
        ).values
        if patients[label_col].nunique() > 1:
            stratify = patients[label_col]

    train_patients, test_patients = train_test_split(
        patients[patient_col],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_df = df[df[patient_col].isin(set(train_patients))].copy()
    test_df = df[df[patient_col].isin(set(test_patients))].copy()
    assert_no_patient_overlap(train_df, test_df, patient_col=patient_col)

    summary = {
        "split_level": "patient",
        "patient_col": patient_col,
        "label_col": label_col,
        "random_state": random_state,
        "test_size": test_size,
        "n_train_patients": int(train_df[patient_col].nunique()),
        "n_test_patients": int(test_df[patient_col].nunique()),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
    }
    return train_df, test_df, summary


def summarize_feature_selection_stability(selected_features_by_seed) -> Dict[str, object]:
    """Summarize selected-feature robustness across random seeds."""
    if isinstance(selected_features_by_seed, Mapping):
        selections = {
            str(seed): set(features)
            for seed, features in selected_features_by_seed.items()
        }
    else:
        selections = {
            str(index): set(features)
            for index, features in enumerate(selected_features_by_seed)
        }

    keys = list(selections.keys())
    pairwise = []
    for left, right in combinations(keys, 2):
        union = selections[left] | selections[right]
        score = 1.0 if not union else len(selections[left] & selections[right]) / len(union)
        pairwise.append(score)

    all_features = sorted(set().union(*selections.values())) if selections else []
    frequency = {
        feature: sum(feature in selected for selected in selections.values())
        for feature in all_features
    }
    n_runs = max(len(selections), 1)
    consensus = [
        feature
        for feature, count in frequency.items()
        if count == n_runs
    ]

    if pairwise:
        mean_jaccard = sum(pairwise) / len(pairwise)
        min_jaccard = min(pairwise)
        max_jaccard = max(pairwise)
    else:
        mean_jaccard = min_jaccard = max_jaccard = 1.0

    return {
        "n_runs": len(selections),
        "mean_pairwise_jaccard": float(mean_jaccard),
        "min_pairwise_jaccard": float(min_jaccard),
        "max_pairwise_jaccard": float(max_jaccard),
        "consensus_features": consensus,
        "feature_frequency": frequency,
    }

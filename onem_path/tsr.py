"""Pathological tumor-stroma ratio ground-truth utilities."""

from typing import Optional


def _load_table(data):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for TSR validation") from exc
    return pd.read_csv(data) if isinstance(data, (str, bytes)) else data.copy()


def validate_tsr_table(
    data,
    patient_column: str = "patient_id",
    reader_column: str = "reader_id",
    slide_column: str = "slide_id",
    tsr_column: str = "tsr_percent",
    specimen_column: Optional[str] = "specimen_type",
):
    """Validate reader/slide-level pathological TSR measurements."""
    table = _load_table(data)
    required = {patient_column, reader_column, slide_column, tsr_column}
    if specimen_column:
        required.add(specimen_column)
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing pathological TSR columns: {missing}")
    if table[list(required)].isna().any().any():
        raise ValueError("Required TSR fields cannot be missing")
    if table.duplicated([patient_column, reader_column, slide_column]).any():
        raise ValueError("Duplicate patient-reader-slide TSR records detected")
    values = table[tsr_column].astype(float)
    if ((values < 0) | (values > 100)).any():
        raise ValueError("TSR percentages must be between 0 and 100")
    return table


def aggregate_tsr_ground_truth(
    data,
    cutoff: float,
    patient_column: str = "patient_id",
    reader_column: str = "reader_id",
    slide_column: str = "slide_id",
    tsr_column: str = "tsr_percent",
    specimen_column: Optional[str] = "specimen_type",
    high_label: str = "H-TSR",
    low_label: str = "L-TSR",
):
    """Aggregate multi-reader, multi-slide TSR with heterogeneity summaries."""
    table = validate_tsr_table(
        data,
        patient_column=patient_column,
        reader_column=reader_column,
        slide_column=slide_column,
        tsr_column=tsr_column,
        specimen_column=specimen_column,
    )
    grouped = table.groupby(patient_column)[tsr_column]
    result = grouped.agg(
        pathological_tsr="median",
        tsr_mean="mean",
        tsr_std="std",
        tsr_min="min",
        tsr_max="max",
        n_measurements="size",
    ).reset_index()
    result["tsr_range"] = result["tsr_max"] - result["tsr_min"]
    result["tsr_group"] = result["pathological_tsr"].apply(
        lambda value: high_label if value >= cutoff else low_label
    )
    result["cutoff"] = float(cutoff)
    return result


def tsr_interobserver_icc(
    data,
    patient_column: str = "patient_id",
    reader_column: str = "reader_id",
    tsr_column: str = "tsr_percent",
):
    """Calculate ICC(2,1) across readers after averaging slides per reader."""
    import numpy as np

    table = _load_table(data)
    required = {patient_column, reader_column, tsr_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing agreement columns: {missing}")
    pivot = table.pivot_table(
        index=patient_column,
        columns=reader_column,
        values=tsr_column,
        aggfunc="mean",
    ).dropna()
    n_subjects, n_readers = pivot.shape
    if n_subjects < 2 or n_readers < 2:
        raise ValueError("At least two complete patients and readers are required")
    values = pivot.to_numpy(dtype=float)
    grand_mean = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ms_rows = n_readers * ((row_means - grand_mean) ** 2).sum() / (n_subjects - 1)
    ms_columns = (
        n_subjects * ((column_means - grand_mean) ** 2).sum() / (n_readers - 1)
    )
    residuals = values - row_means[:, None] - column_means[None, :] + grand_mean
    ms_error = (residuals ** 2).sum() / ((n_subjects - 1) * (n_readers - 1))
    denominator = (
        ms_rows
        + (n_readers - 1) * ms_error
        + n_readers * (ms_columns - ms_error) / n_subjects
    )
    icc = float("nan") if denominator == 0 else float((ms_rows - ms_error) / denominator)
    return {
        "icc2_1": icc,
        "n_patients": int(n_subjects),
        "n_readers": int(n_readers),
    }


def tsr_group_agreement(
    data,
    cutoff: float,
    patient_column: str = "patient_id",
    reader_column: str = "reader_id",
    tsr_column: str = "tsr_percent",
):
    """Calculate pairwise Cohen's kappa for reader TSR groups."""
    try:
        import pandas as pd
        from sklearn.metrics import cohen_kappa_score
    except ImportError as exc:
        raise ImportError(
            "TSR group agreement requires pandas and scikit-learn"
        ) from exc
    table = _load_table(data)
    reader_values = table.pivot_table(
        index=patient_column,
        columns=reader_column,
        values=tsr_column,
        aggfunc="mean",
    )
    readers = list(reader_values.columns)
    rows = []
    for left_index, left in enumerate(readers):
        for right in readers[left_index + 1 :]:
            paired = reader_values[[left, right]].dropna()
            rows.append(
                {
                    "reader_1": left,
                    "reader_2": right,
                    "n_patients": int(len(paired)),
                    "cohen_kappa": float(
                        cohen_kappa_score(
                            paired[left] >= cutoff,
                            paired[right] >= cutoff,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def compare_specimen_tsr(
    data,
    specimen_column: str = "specimen_type",
    patient_column: str = "patient_id",
    tsr_column: str = "tsr_percent",
):
    """Compare paired biopsy and resection TSR measurements."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Specimen comparison requires numpy and pandas") from exc
    table = _load_table(data)
    required = {specimen_column, patient_column, tsr_column}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Missing specimen comparison columns: {missing}")
    pivot = table.pivot_table(
        index=patient_column,
        columns=specimen_column,
        values=tsr_column,
        aggfunc="mean",
    )
    if pivot.shape[1] != 2:
        raise ValueError("Exactly two specimen types are required")
    paired = pivot.dropna()
    left, right = paired.columns
    difference = paired[right] - paired[left]
    return {
        "specimen_1": left,
        "specimen_2": right,
        "n_paired": int(len(paired)),
        "mean_difference": float(difference.mean()),
        "median_absolute_difference": float(np.median(np.abs(difference))),
        "spearman_correlation": float(paired[left].corr(paired[right], method="spearman")),
        "paired_table": paired.reset_index(),
    }

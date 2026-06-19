"""Clinical table consistency checks."""


def check_stage_size_consistency(
    clinical_table,
    stage_column: str = "t_stage",
    diameter_column: str = "tumor_diameter_cm",
):
    """Flag basic T-stage and tumor-size contradictions.

    The default thresholds reflect the manuscript review comments. Confirm the
    thresholds against the AJCC edition used by each study before reporting.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for clinical consistency checks") from exc

    if stage_column not in clinical_table.columns:
        raise ValueError(f"Missing stage column: {stage_column}")
    if diameter_column not in clinical_table.columns:
        raise ValueError(f"Missing diameter column: {diameter_column}")

    df = clinical_table.copy()
    stage = df[stage_column].astype(str).str.upper().str.replace(" ", "", regex=False)
    diameter = pd.to_numeric(df[diameter_column], errors="coerce")

    reasons = []
    for current_stage, current_diameter in zip(stage, diameter):
        row_reasons = []
        if current_stage.startswith("T1") and current_diameter > 2:
            row_reasons.append("T1 recorded with diameter > 2 cm")
        if current_stage.startswith("T3") and current_diameter <= 4:
            row_reasons.append("T3 recorded with diameter <= 4 cm")
        reasons.append("; ".join(row_reasons))

    flagged = df[[stage_column, diameter_column]].copy()
    flagged["consistency_issue"] = reasons
    return flagged[flagged["consistency_issue"] != ""].reset_index(drop=False)

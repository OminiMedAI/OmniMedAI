"""Multimodal feature alignment and fusion helpers."""

from typing import Dict, Iterable, Optional

from .config.settings import FusionConfig


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for feature fusion. Install with: pip install pandas") from exc
    return pd


def _load_table(table):
    pd = _require_pandas()
    if isinstance(table, (str, bytes)):
        return pd.read_csv(table)
    return table.copy()


def _prefix_features(df, modality_name: str, config: FusionConfig):
    prefix = config.modality_prefixes.get(modality_name, modality_name)
    rename_map = {}
    for column in df.columns:
        if column == config.id_column or column in config.drop_columns:
            continue
        if column.startswith(f"{prefix}_"):
            continue
        rename_map[column] = f"{prefix}_{column}"
    return df.rename(columns=rename_map)


def align_feature_tables(feature_tables: Dict[str, object], config: Optional[FusionConfig] = None):
    """
    Align feature tables by patient/sample identifier.

    Args:
        feature_tables: Mapping from modality name to DataFrame or CSV path.
        config: Fusion configuration.

    Returns:
        A single pandas DataFrame containing aligned features.
    """
    config = config or FusionConfig()
    if not feature_tables:
        raise ValueError("feature_tables cannot be empty")

    merged = None
    for modality_name, table in feature_tables.items():
        df = _load_table(table)
        if config.id_column not in df.columns:
            raise ValueError(f"{modality_name} table is missing id column: {config.id_column}")

        if config.drop_columns:
            df = df.drop(columns=[c for c in config.drop_columns if c in df.columns])

        df = _prefix_features(df, modality_name, config)
        merged = df if merged is None else merged.merge(df, on=config.id_column, how=config.join_strategy)

    if config.fill_missing:
        merged = _fill_missing_values(merged, config)

    return merged


def concatenate_features(feature_tables: Iterable[object], id_column: str = "patient_id"):
    """Concatenate feature tables with default modality names."""
    tables = {f"modality{i + 1}": table for i, table in enumerate(feature_tables)}
    return align_feature_tables(tables, FusionConfig(id_column=id_column))


def _fill_missing_values(df, config: FusionConfig):
    numeric_columns = [c for c in df.columns if c != config.id_column]
    if config.fill_missing == "zero":
        df[numeric_columns] = df[numeric_columns].fillna(0)
    elif config.fill_missing in {"mean", "median"}:
        for column in numeric_columns:
            if not hasattr(df[column], config.fill_missing):
                continue
            value = getattr(df[column], config.fill_missing)()
            df[column] = df[column].fillna(value)
    return df


class FeatureFusion:
    """Convenience wrapper for repeated multimodal table fusion."""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def fit_transform(self, feature_tables: Dict[str, object]):
        return align_feature_tables(feature_tables, self.config)

    def save(self, fused_table, output_csv_path: str):
        fused_table.to_csv(output_csv_path, index=False)
        return output_csv_path

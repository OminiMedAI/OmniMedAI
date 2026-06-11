"""Lightweight tabular modeling workflows for medical AI features."""

from typing import List, Optional

from .config.settings import ModelingConfig


def _require_dependencies():
    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC, SVR
    except ImportError as exc:
        raise ImportError(
            "pandas and scikit-learn are required for modeling. "
            "Install with: pip install pandas scikit-learn"
        ) from exc

    return {
        "pd": pd,
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer,
        "LinearRegression": LinearRegression,
        "LogisticRegression": LogisticRegression,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
        "SVR": SVR,
    }


class TabularModeler:
    """Train baseline models from fused medical feature tables."""

    def __init__(self, config: Optional[ModelingConfig] = None):
        self.config = config or ModelingConfig()
        self.pipeline = None
        self.feature_columns = None

    def fit_from_csv(
        self,
        csv_path: str,
        label_column: str,
        id_columns: Optional[List[str]] = None,
    ):
        deps = _require_dependencies()
        pd = deps["pd"]
        df = pd.read_csv(csv_path)
        return self.fit(df, label_column=label_column, id_columns=id_columns)

    def fit(self, df, label_column: str, id_columns: Optional[List[str]] = None):
        deps = _require_dependencies()
        train_test_split = deps["train_test_split"]

        id_columns = id_columns or ["patient_id", "image_path", "mask_path"]
        drop_columns = [label_column] + [c for c in id_columns if c in df.columns]
        self.feature_columns = [c for c in df.columns if c not in drop_columns]

        x = df[self.feature_columns]
        y = df[label_column]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if self.config.task == "classification" else None,
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(x_train, y_train)

        predictions = self.pipeline.predict(x_test)
        probabilities = None
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(x_test)

        return {
            "model": self.pipeline,
            "feature_columns": self.feature_columns,
            "x_test": x_test,
            "y_test": y_test,
            "predictions": predictions,
            "probabilities": probabilities,
        }

    def predict(self, df):
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted")
        return self.pipeline.predict(df[self.feature_columns])

    def _build_pipeline(self):
        deps = _require_dependencies()
        Pipeline = deps["Pipeline"]
        SimpleImputer = deps["SimpleImputer"]
        StandardScaler = deps["StandardScaler"]

        steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.config.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", self._create_model()))
        return Pipeline(steps)

    def _create_model(self):
        deps = _require_dependencies()
        cfg = self.config

        if cfg.task == "classification":
            if cfg.model_type == "random_forest":
                return deps["RandomForestClassifier"](
                    n_estimators=200,
                    random_state=cfg.random_state,
                    class_weight=cfg.class_weight,
                )
            if cfg.model_type == "svm":
                return deps["SVC"](probability=True, class_weight=cfg.class_weight)
            if cfg.model_type == "logistic_regression":
                return deps["LogisticRegression"](
                    max_iter=1000,
                    class_weight=cfg.class_weight,
                    random_state=cfg.random_state,
                )

        if cfg.model_type == "random_forest":
            return deps["RandomForestRegressor"](
                n_estimators=200,
                random_state=cfg.random_state,
            )
        if cfg.model_type == "svm":
            return deps["SVR"]()
        if cfg.model_type == "linear_regression":
            return deps["LinearRegression"]()

        raise ValueError(f"Unsupported model type for task: {cfg.model_type}")


def train_tabular_model(csv_path: str, label_column: str, config: Optional[ModelingConfig] = None):
    """Train a baseline tabular model from a CSV feature table."""
    modeler = TabularModeler(config)
    return modeler.fit_from_csv(csv_path, label_column=label_column)

"""Training-data preprocessing helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import add_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing transformer from the training feature schema."""
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
    )


def prepare_features(df: pd.DataFrame, target: str | None = None):
    """Apply feature engineering and optionally separate the target."""
    out = add_features(df)
    if target is None:
        return out
    if target not in out.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    return out.drop(columns=[target]), out[target]

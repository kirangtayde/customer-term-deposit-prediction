"""Reusable training pipeline for bank term-deposit propensity modeling."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_columns: list[str]


def build_pipeline(X: pd.DataFrame) -> ModelBundle:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    return ModelBundle(Pipeline([("preprocess", preprocessor), ("model", model)]), X.columns.tolist())

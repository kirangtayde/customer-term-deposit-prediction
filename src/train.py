"""Train and save a baseline propensity model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import build_preprocessor, prepare_features


def train(input_path: str, target: str, model_path: str) -> dict[str, float]:
    df = pd.read_csv(input_path)
    X, y = prepare_features(df, target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probability)
    print(classification_report(y_test, prediction, zero_division=0))
    print(f"ROC-AUC: {auc:.4f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return {"roc_auc": float(auc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", default="y")
    parser.add_argument("--model", default="models/model.joblib")
    args = parser.parse_args()
    train(args.input, args.target, args.model)

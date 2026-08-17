"""Score customer records with a trained propensity model."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from data_preprocessing import prepare_features


def score(input_path: str, model_path: str, output_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    raw = pd.read_csv(input_path)
    X = prepare_features(raw)
    probability = model.predict_proba(X)[:, 1]
    result = raw.copy()
    result["subscription_probability"] = probability
    result["prediction"] = (probability >= 0.5).astype(int)
    result["segment"] = pd.cut(
        probability,
        bins=[-0.01, 0.20, 0.40, 0.60, 0.80, 1.01],
        labels=["Very Low", "Low", "Medium", "Strong", "High"],
    )
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    score(args.input, args.model, args.output)

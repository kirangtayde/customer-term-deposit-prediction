"""Evaluation metrics for binary propensity models."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


def evaluate(y_true, probability, threshold: float = 0.5) -> dict[str, float]:
    """Return ROC-AUC, PR-AUC and thresholded classification metrics."""
    prediction = (probability >= threshold).astype(int)
    report = classification_report(y_true, prediction, output_dict=True, zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
    }

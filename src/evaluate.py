"""Model evaluation helpers for imbalanced propensity classification."""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

def classification_report_summary(y_true, probability, threshold: float = 0.50) -> dict[str, float]:
    pred = (probability >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def threshold_table(y_true, probability, thresholds=None) -> pd.DataFrame:
    thresholds = thresholds or [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    rows = []
    for t in thresholds:
        m = classification_report_summary(y_true, probability, t)
        rows.append({"threshold": t, **m})
    return pd.DataFrame(rows)

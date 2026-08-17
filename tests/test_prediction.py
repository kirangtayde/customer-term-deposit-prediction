import numpy as np

from src.evaluation import evaluate


def test_evaluate_returns_expected_metrics():
    y_true = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate(y_true, probability)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0

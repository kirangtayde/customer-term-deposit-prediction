"""Business-oriented scoring, segmentation and lift utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

def segment_propensity(probability: float) -> str:
    if probability >= 0.80: return "High Potential"
    if probability >= 0.60: return "Strong Potential"
    if probability >= 0.40: return "Medium"
    if probability >= 0.20: return "Low"
    return "Very Low"


def add_segments(frame: pd.DataFrame, probability_col: str = "propensity") -> pd.DataFrame:
    out = frame.copy()
    out["segment"] = out[probability_col].map(segment_propensity)
    return out.sort_values(probability_col, ascending=False)


def decile_lift(y_true, probability) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(probability)})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), 10, labels=False) + 1
    overall = df["y"].mean()
    result = df.groupby("decile", as_index=False).agg(records=("y", "size"), conversions=("y", "sum"), conversion_rate=("y", "mean"))
    result["lift"] = result["conversion_rate"] / overall if overall else np.nan
    return result

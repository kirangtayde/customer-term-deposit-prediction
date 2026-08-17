"""Feature engineering utilities for banking campaign data."""

from __future__ import annotations

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe, lightweight derived features.

    The function only uses columns that are present. This makes it reusable
    across datasets with slightly different schemas.
    """
    out = df.copy()

    if "age" in out.columns:
        out["age_group"] = pd.cut(
            out["age"], bins=[0, 25, 35, 50, 65, float("inf")],
            labels=["young", "adult", "mid_age", "senior", "65_plus"],
        ).astype("string")

    if "balance" in out.columns:
        out["balance_group"] = pd.cut(
            out["balance"], bins=[-float("inf"), 0, 1000, 5000, float("inf")],
            labels=["negative", "low", "medium", "high"],
        ).astype("string")

    if "previous" in out.columns:
        out["previous_contact_flag"] = (out["previous"].fillna(0) > 0).astype(int)

    if "pdays" in out.columns:
        out["previously_contacted"] = (out["pdays"].fillna(-1) >= 0).astype(int)

    if "campaign" in out.columns:
        out["campaign_intensity"] = out["campaign"].clip(lower=0)

    return out

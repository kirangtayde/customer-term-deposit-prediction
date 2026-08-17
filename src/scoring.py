from __future__ import annotations

import pandas as pd


def rank_leads(probabilities: pd.Series, capacity: int | None = None, threshold: float = 0.5) -> pd.DataFrame:
    """Turn model probabilities into ranked campaign leads."""
    if probabilities.empty:
        raise ValueError("probabilities cannot be empty")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if capacity is not None and capacity < 1:
        raise ValueError("capacity must be >= 1")

    result = pd.DataFrame({"propensity": probabilities.astype(float)})
    result["priority"] = (result["propensity"] >= threshold).astype(int)
    result = result.sort_values(["priority", "propensity"], ascending=[False, False])
    if capacity is not None:
        result = result.head(capacity)
    result["lead_rank"] = range(1, len(result) + 1)
    return result

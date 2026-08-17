import pandas as pd

from src.scoring import rank_leads


def test_rank_leads_prioritizes_high_propensity():
    result = rank_leads(pd.Series([0.2, 0.9, 0.7], index=["a", "b", "c"]), capacity=2)
    assert list(result.index) == ["b", "c"]
    assert list(result["lead_rank"]) == [1, 2]

import pandas as pd
from src.business import add_segments, segment_propensity

def test_segment_boundaries():
    assert segment_propensity(0.81) == "High Potential"
    assert segment_propensity(0.60) == "Strong Potential"
    assert segment_propensity(0.10) == "Very Low"

def test_add_segments():
    out = add_segments(pd.DataFrame({"propensity": [0.9, 0.1]}))
    assert out.iloc[0]["segment"] == "High Potential"

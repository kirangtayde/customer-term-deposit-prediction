import pandas as pd

from src.feature_engineering import add_features


def test_add_features_creates_previous_contact_flag():
    df = pd.DataFrame({"age": [30], "balance": [1200], "previous": [2]})
    result = add_features(df)
    assert result.loc[0, "previous_contact_flag"] == 1
    assert "age_group" in result.columns
    assert "balance_group" in result.columns

"""Minimal Streamlit interface for trained customer propensity models."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_preprocessing import prepare_features

st.set_page_config(page_title="Term Deposit Propensity", page_icon="📈", layout="wide")
st.title("Customer Term Deposit Propensity")
st.caption("Score approved customer records with a trained ML model.")

model_path = Path("models/model.joblib")
if not model_path.exists():
    st.warning("No trained model found. Train a model first and save it to models/model.joblib.")
    st.stop()

uploaded = st.file_uploader("Upload a CSV for batch scoring", type=["csv"])
if uploaded:
    model = joblib.load(model_path)
    raw = pd.read_csv(uploaded)
    X = prepare_features(raw)
    probability = model.predict_proba(X)[:, 1]
    result = raw.copy()
    result["subscription_probability"] = probability
    result["segment"] = pd.cut(
        probability,
        bins=[-0.01, 0.20, 0.40, 0.60, 0.80, 1.01],
        labels=["Very Low", "Low", "Medium", "Strong", "High"],
    )
    st.dataframe(result, use_container_width=True)
    st.download_button("Download scored CSV", result.to_csv(index=False), "predictions.csv", "text/csv")

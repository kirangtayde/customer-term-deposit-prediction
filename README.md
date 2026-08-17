# Customer Term Deposit Prediction

An end-to-end machine learning project that predicts whether a bank customer is likely to subscribe to a term deposit and converts model probabilities into actionable marketing lead segments.

## Business Problem

Banks may contact thousands of customers during deposit campaigns while only a fraction convert. The goal is to identify high-potential customers before outreach so limited sales capacity can be prioritized.

### Objective

Build a supervised binary classification pipeline that estimates:

`P(customer subscribes | customer and campaign characteristics)`

The output is a probability score rather than only a yes/no prediction.

## Business Value

- Prioritize high-propensity leads
- Improve campaign conversion efficiency
- Reduce unnecessary outreach
- Support data-driven marketing allocation
- Enable threshold selection based on campaign economics

## ML Workflow

```text
Raw Data -> Validation -> EDA -> Cleaning -> Feature Engineering
        -> Encoding -> Stratified Split -> Imbalance Handling
        -> Baseline -> Tree/Boosting Models -> Tuning
        -> Evaluation -> Explainability -> Threshold Optimization
        -> Lead Segmentation -> Deployment
```

## Target

`term_deposit_subscription`

| Value | Meaning |
|---|---|
| 0 | Customer does not subscribe |
| 1 | Customer subscribes |

The repository intentionally does not include customer PII or a dataset by default. Place an approved dataset in `data/raw/` locally and document its source/license before sharing it.

## Features

Typical banking-marketing features can include age, job, marital status, education, balance, credit/default status, housing and personal loans, contact channel, campaign timing, number of contacts, previous campaign history and previous outcome.

Feature engineering can create signals such as age group, balance group, previous-contact flag, campaign intensity and previous-success flag. All transformations should be fit on training data only to prevent leakage.

## Models

The project is designed to compare:

1. Logistic Regression — interpretable baseline
2. Random Forest — nonlinear interactions
3. Gradient Boosting — strong tabular baseline
4. XGBoost — optional advanced boosting model

Model selection should consider business objectives rather than accuracy alone.

## Evaluation

Recommended metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Calibration / probability quality
- Lift and conversion by propensity decile

For marketing use cases, precision and recall must be interpreted together with the cost of contacting a customer and the value of a successful subscription.

## Lead Segmentation

Example probability policy:

| Probability | Segment | Action |
|---:|---|---|
| >= 0.80 | High Potential | Immediate sales priority |
| 0.60-0.79 | Strong Potential | Priority follow-up |
| 0.40-0.59 | Medium | Secondary campaign |
| 0.20-0.39 | Low | Low-cost digital campaign |
| < 0.20 | Very Low | Do not prioritize |

These thresholds are illustrative. They should be optimized against validation data and campaign economics.

## Explainability

The project supports feature importance and can be extended with SHAP. Explainability should answer questions such as which customer attributes drive propensity and whether the model behaves consistently across customer segments.

## Project Structure

```text
customer-term-deposit-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── models/
├── reports/
│   └── figures/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── evaluation.py
├── app/
│   └── app.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── .github/workflows/ci.yml
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Quick Start

```bash
git clone https://github.com/kirangtayde/customer-term-deposit-prediction.git
cd customer-term-deposit-prediction
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Train after placing a compatible dataset in `data/raw/`:

```bash
python src/train.py --input data/raw/bank_marketing.csv --target y
```

Generate a prediction for a CSV containing feature columns:

```bash
python src/predict.py --input data/raw/scoring.csv --model models/model.joblib --output predictions.csv
```

## API Demo

The Streamlit app provides an interactive scoring interface when a trained model exists:

```bash
streamlit run app/app.py
```

## Testing and CI

GitHub Actions runs dependency installation and the unit test suite on pushes and pull requests. The project uses small deterministic tests so the CI pipeline remains useful even when no private/customer dataset is committed.

## Data Privacy

Do not commit personally identifiable information, confidential bank records, credentials, `.env` files, raw production data or customer-level sensitive data. Use `.gitignore` and approved data-governance procedures.

## Limitations

Historical campaign data can contain sampling bias, changing customer behavior and campaign-specific effects. A model should be monitored for data drift, calibration, subgroup performance and changing business economics before production use.

## Future Improvements

- Hyperparameter optimization with Optuna
- SHAP explainability dashboard
- Probability calibration
- Cost-sensitive learning
- Uplift modeling
- MLflow experiment tracking
- Data/model drift monitoring
- FastAPI production service
- Dockerized deployment
- Model registry and automated retraining
- Campaign A/B testing

## Resume Value

**Customer Term Deposit Prediction | Python, Pandas, Scikit-learn, XGBoost, SHAP**

Built an end-to-end propensity modeling workflow for banking campaign targeting, including EDA, feature engineering, class-imbalance strategy, model comparison, probability scoring, threshold-based lead segmentation and reproducible testing/CI.

## Author

**Kiran Tayde**  
Data Science | Machine Learning | Python | Analytics

GitHub: https://github.com/kirangtayde

## License

MIT License. See `LICENSE`.
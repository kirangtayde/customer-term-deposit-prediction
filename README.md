# Customer Term Deposit Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![ML](https://img.shields.io/badge/ML-Classification-orange)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-purple)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green)

An end-to-end **customer propensity modeling** project that predicts the likelihood of a bank customer subscribing to a term deposit and converts model probabilities into actionable marketing lead segments.

## About the Author

**Kiran Tayde** — Senior Data Scientist | Machine Learning | NLP | Predictive Analytics | Business Intelligence

I build practical, production-oriented data science solutions that connect **business problems, machine learning, analytics and measurable outcomes**. My portfolio focuses on predictive modeling, NLP, recommendation systems, marketing analytics, data quality and BI.

**GitHub:** https://github.com/kirangtayde

## Business problem

Banking campaigns can contact thousands of customers while only a small proportion convert. The objective is to estimate **who is most likely to subscribe**, rank customers by propensity and help sales teams prioritize limited outreach capacity.

The model output is a probability:

`P(term_deposit_subscription | customer + campaign features)`

rather than a simple yes/no label.

## Business value

- Prioritize high-potential leads
- Improve campaign conversion efficiency
- Reduce low-value outreach
- Support data-driven marketing allocation
- Enable threshold selection based on business economics
- Provide interpretable drivers for model decisions

## ML workflow

```text
Raw Data → Validation → EDA → Feature Engineering → Preprocessing
→ Model Training → Evaluation → Threshold Optimization
→ Lead Segmentation → Explainability → Deployment / Monitoring
```

## Models

1. Logistic Regression — interpretable baseline
2. Random Forest — nonlinear interactions
3. Gradient Boosting — strong tabular baseline
4. XGBoost — advanced gradient boosting option

## Evaluation

Precision, Recall, F1-score, ROC-AUC, PR-AUC, confusion matrix, probability calibration, lift/gain and conversion rate by lead segment.

## Lead segmentation

| Probability | Segment | Suggested action |
|---:|---|---|
| >= 0.80 | High Potential | Immediate sales priority |
| 0.60–0.79 | Strong Potential | Priority follow-up |
| 0.40–0.59 | Medium | Secondary campaign |
| 0.20–0.39 | Low | Low-cost digital campaign |
| < 0.20 | Very Low | Do not prioritize |

These values are examples only. Production thresholds should be optimized against contact cost, conversion value and sales capacity.

## Explainability & responsible AI

The project supports feature-importance analysis and can be extended with SHAP. Banking propensity models should be evaluated for leakage, sampling bias, calibration, privacy, subgroup performance and model drift before production use.

## Project structure

```text
customer-term-deposit-prediction/
├── data/ ├── notebooks/ ├── models/ ├── reports/ ├── src/ ├── app/ ├── tests/
├── .github/workflows/ci.yml
├── requirements.txt
├── Dockerfile
└── README.md
```

## Quick start

```bash
git clone https://github.com/kirangtayde/customer-term-deposit-prediction.git
cd customer-term-deposit-prediction
python -m venv .venv
pip install -r requirements.txt
pytest -q
```

## Resume-ready summary

**Customer Term Deposit Prediction | Python, Pandas, Scikit-learn, XGBoost, SHAP** — Built an end-to-end propensity modeling workflow for banking campaign targeting, including preprocessing, feature engineering, model comparison, probability scoring, threshold-based lead segmentation, explainability, testing and CI.

## Connect

**Kiran Tayde** · Senior Data Scientist · Machine Learning · NLP · Analytics

[GitHub](https://github.com/kirangtayde)

## License

MIT License. See `LICENSE`.
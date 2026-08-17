# 🏦 Customer Term Deposit Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-purple)
![Testing](https://img.shields.io/badge/Testing-PyTest-green)

An end-to-end **customer propensity modeling** project that predicts the likelihood of a bank customer subscribing to a term deposit, ranks customers by probability and converts predictions into actionable lead segments.

## 👨‍💻 Author

**Kiran Tayde — Senior Data Scientist | Machine Learning | NLP | Predictive Analytics | Business Intelligence**

GitHub: https://github.com/kirangtayde

## 🎯 Business Problem

Banking campaigns can contact thousands of customers while only a small proportion convert. The objective is to estimate customer propensity, prioritize outreach and improve campaign efficiency.

The model estimates:

`P(subscription | customer + campaign features)`

rather than relying only on a binary decision.

## 💼 Business Value

- Rank high-potential customers for targeted outreach
- Improve campaign conversion efficiency
- Reduce low-value contacts
- Support capacity-aware campaign prioritization
- Translate model probabilities into actionable segments
- Explain the main drivers behind predictions

## 🔬 End-to-End ML Workflow

```text
Raw Data
   ↓
Validation & Data Quality
   ↓
EDA
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Baseline + ML Models
   ↓
Evaluation & Calibration
   ↓
Threshold Optimization
   ↓
Lead Segmentation
   ↓
Explainability
   ↓
Deployment / Monitoring
```

## 🤖 Models

- Logistic Regression — interpretable baseline
- Random Forest — nonlinear relationships
- Gradient Boosting — strong tabular baseline
- XGBoost — advanced gradient boosting

## 📊 Evaluation

Precision • Recall • F1 • ROC-AUC • PR-AUC • Confusion Matrix • Calibration • Lift/Gain • Segment Conversion

Production thresholds should be selected using campaign cost, customer value, contact capacity and business objectives.

## 🧠 Explainability & Responsible AI

The project supports feature-importance analysis and SHAP-based explainability. Before real banking deployment, evaluate leakage, calibration, privacy, subgroup performance, fairness, drift and regulatory requirements.

## 📁 Project Structure

```text
customer-term-deposit-prediction/
├── data/
├── notebooks/
├── models/
├── reports/
├── src/
├── app/
├── tests/
├── .github/workflows/
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quick Start

```bash
git clone https://github.com/kirangtayde/customer-term-deposit-prediction.git
cd customer-term-deposit-prediction
python -m venv .venv
pip install -r requirements.txt
pytest -q
```

## 📌 Resume Summary

**Customer Term Deposit Prediction | Python, Pandas, Scikit-learn, XGBoost, SHAP** — Built an end-to-end banking propensity modeling workflow covering preprocessing, feature engineering, model comparison, probability scoring, threshold-based lead segmentation, explainability, testing and CI.

## 🔗 Connect

**Kiran Tayde** · Senior Data Scientist · Machine Learning · NLP · Analytics

https://github.com/kirangtayde

## License

MIT License. See `LICENSE`.
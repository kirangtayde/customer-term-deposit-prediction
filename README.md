# Customer Term Deposit Prediction

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![ML](https://img.shields.io/badge/ML-Classification-orange)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-purple)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-green)

An end-to-end **customer propensity modeling** project that predicts the likelihood of a bank customer subscribing to a term deposit and converts model probabilities into actionable marketing lead segments.

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
Raw Data
   ↓
Validation & Data Quality
   ↓
EDA + Feature Engineering
   ↓
Preprocessing / Encoding
   ↓
Stratified Train-Test Split
   ↓
Baseline + Tree / Boosting Models
   ↓
Probability Evaluation
   ↓
Threshold Optimization
   ↓
Lead Segmentation
   ↓
Explainability
   ↓
Deployment / Monitoring
```

## Target

The default target is `y` and can represent term-deposit subscription using either binary values or common `yes`/`no` labels.

The repository does **not** include customer PII or production banking data. Put an approved dataset in `data/raw/` locally and document its source and license before publication.

## Features

Typical bank-marketing variables can include:

- Demographics: age, job, marital status, education
- Financial profile: account balance, default, housing loan, personal loan
- Campaign information: contact channel, campaign count, previous contacts
- Historical outcome: previous campaign result
- Engineered signals: balance groups, age groups, previous-contact flags and campaign intensity

All transformations should be fitted only on training data to avoid data leakage.

## Models

The project is structured for model comparison:

1. **Logistic Regression** — interpretable baseline
2. **Random Forest** — nonlinear interactions
3. **Gradient Boosting** — strong tabular baseline
4. **XGBoost** — advanced gradient boosting option

Model selection should prioritize the business objective, not accuracy alone.

## Evaluation

Recommended metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Probability calibration
- Lift / gain by propensity decile
- Conversion rate by lead segment

For imbalanced marketing data, PR-AUC, recall at a practical contact budget and expected campaign value can be more informative than raw accuracy.

## Lead segmentation

Illustrative policy:

| Probability | Segment | Suggested action |
|---:|---|---|
| >= 0.80 | High Potential | Immediate sales priority |
| 0.60–0.79 | Strong Potential | Priority follow-up |
| 0.40–0.59 | Medium | Secondary campaign |
| 0.20–0.39 | Low | Low-cost digital campaign |
| < 0.20 | Very Low | Do not prioritize |

These values are examples only. Production thresholds should be optimized on validation data against contact cost, conversion value and available sales capacity.

## Explainability

The project supports feature-importance analysis and can be extended with SHAP to explain individual predictions and global model behavior. Explainability should be used to identify stable business signals and detect unexpected model behavior across customer groups.

## Project structure

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
├── .github/workflows/ci.yml
├── requirements.txt
├── Dockerfile
├── .gitignore
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

Train on an approved CSV:

```bash
python src/train.py --input data/raw/bank_marketing.csv --target y --model models/model.joblib
```

Generate scores:

```bash
python src/predict.py --input data/raw/scoring.csv --model models/model.joblib --output predictions.csv
```

Run the interactive application when a compatible trained model is available:

```bash
streamlit run app/app.py
```

## Engineering practices

- Reusable Python modules instead of notebook-only code
- Unit tests for preprocessing and prediction behavior
- GitHub Actions CI on pushes and pull requests
- Docker support for reproducible execution
- No credentials, PII or production records committed to Git
- Explicit separation of training, scoring and evaluation

## Responsible AI and data privacy

Banking propensity models can affect how customers are targeted. The project should be evaluated for leakage, sampling bias, calibration, subgroup performance, privacy and changing campaign behavior before production use. Do not commit customer-level sensitive information or confidential banking records.

## Limitations

Historical campaign data can contain selection bias, campaign-specific effects and changing customer behavior. A strong offline metric does not guarantee business impact. Production deployment should include drift monitoring, recalibration, threshold review and controlled campaign experiments.

## Future improvements

- Hyperparameter optimization with Optuna
- SHAP explainability dashboard
- Probability calibration
- Cost-sensitive learning
- Uplift modeling
- MLflow experiment tracking
- Data/model drift monitoring
- FastAPI inference service
- Dockerized deployment
- Automated retraining and model registry
- Campaign A/B testing

## Resume-ready summary

**Customer Term Deposit Prediction | Python, Pandas, Scikit-learn, XGBoost, SHAP** — Built an end-to-end propensity modeling workflow for banking campaign targeting, including preprocessing, feature engineering, class-imbalance strategy, model comparison, probability scoring, threshold-based lead segmentation, explainability, testing and CI.

## Author

**Kiran Tayde**  
Senior Data Scientist | Machine Learning | Data Science | NLP

GitHub: https://github.com/kirangtayde

## License

MIT License. See `LICENSE`.
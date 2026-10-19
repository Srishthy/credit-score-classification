# Credit Score Classification — Leak-Free Evaluation

Predicting a customer's credit score (**Poor / Standard / Good**) from monthly banking and credit records,
with an evaluation design that reflects how the model would actually be used.

**Data:** [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) (Kaggle):
100,000 labelled rows (12,500 customers × January–August) and 50,000 unlabelled rows (the same customers, September–December).

## Key finding: the usual evaluation leaks

Each customer has 8 nearly identical monthly rows. A random train/validation split puts some of a customer's months in
training and the rest in validation, so the model partly *recognizes customers* instead of generalizing.
This inflates scores by **10+ points**; most public notebooks on this dataset report numbers of this kind.

This project evaluates two honest, deployment-matched framings instead:

| Setup | Validation | Accuracy | Macro F1 |
|---|---|---|---|
| Random split (leaky, common practice) | random 5-fold | 80.2% | 0.798 |
| **A — New customers** | customer-grouped 5-fold CV | **69.9%** | **0.690** |
| B — Baseline: repeat last known score | time split Jan–Jun → Jul–Aug | 72.9% | 0.722 |
| B — Features only | time split Jan–Jun → Jul–Aug | 71.6% | 0.706 |
| **B — Existing customers + score history** | time split Jan–Jun → Jul–Aug | **75.7%** | **0.756** |

- **Framing A** answers *"how well can we score a customer the bank has never seen?"* All of a customer's rows
  fall in the same fold (`StratifiedGroupKFold` on `Customer_ID`).
- **Framing B** answers *"given a customer's history so far, what will their score be in the coming months?"*,
  which is exactly what the test set is. Validation is a time split, history features use only scores known before the
  cut-off, and the model is trained on 1–4-month horizons to match the 1–4-month test horizon.
  It beats the naive "repeat last score" baseline by **+2.8 points accuracy** and **+3.4 points macro F1**,
  and the gain holds at both 1 and 2 months ahead.

## Pipeline

1. **Per-customer data repair.** The raw data is heavily corrupted: numbers stored as text (`'45_'`), placeholders
   (`'_'`, `'_______'`, `'__10000__'`, `'!@9#%8'`) and injected outliers (ages of 8,698, incomes ×300, 1,400 bank accounts).
   Because each customer has 12 monthly rows, an invalid value is detected relative to that customer's own median
   and replaced with their mode or median; credit-history age is reconstructed from its +1-month-per-month trend.
2. **Feature engineering.** Credit-history months, per-loan-type counts parsed from free text, affordability ratios
   (debt-to-income, EMI-to-salary), and 72 temporal features: lags, month-to-month changes, rolling means,
   and customer-level level/volatility/trend.
3. **Model comparison (framing A).** Logistic Regression, Random Forest, XGBoost and LightGBM on identical folds.
   The tree models tie at about 0.69 macro F1; temporal features add +0.3 to +0.8 points.
4. **Forecasting model (framing B).** LightGBM with score-history features (last and previous score, mean score,
   share of Poor/Good months, number of changes, horizon), compared against a naive baseline and a features-only model.
5. **Outputs.** `submission.csv` (framing A) and `submission_forecast.csv` (framing B) with September–December predictions,
   plus saved scikit-learn pipelines (`.joblib`).

Framing A model comparison (customer-grouped CV, with temporal features):

| Model | Accuracy | Macro F1 |
|---|---|---|
| XGBoost | 69.9% | 0.690 |
| LightGBM | 70.0% | 0.689 |
| Random Forest | 70.7% | 0.688 |
| Logistic Regression | 66.0% | 0.654 |

## Run it

Open `credit_score_pipeline.ipynb` in Google Colab and select **Runtime → Run all** (about 40 minutes on a CPU runtime).
The dataset downloads automatically through `kagglehub`; if prompted, run `kagglehub.login()` with your Kaggle API token.

```
pip install kagglehub lightgbm xgboost scikit-learn pandas matplotlib seaborn
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · matplotlib · seaborn

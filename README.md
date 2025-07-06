# ────────────────────────────────  SET-UP  ────────────────────────────────
$ git clone https://github.com/<your-handle>/rainfall-prediction-classifier.git
$ cd rainfall-prediction-classifier
$ pip install -r requirements.txt          # installs pandas, scikit-learn, etc.
$ python src/train.py                      # trains RF + LR, saves best_model.joblib


# ───────────────────────────────  QUICK TEST  ──────────────────────────────
$ python - <<'PY'
from joblib import load
import pandas as pd

model = load("artifacts/best_model.joblib")   # load the saved pipeline
sample = pd.read_csv("data/sample_day.csv")   # one-day weather record
prob_rain = model.predict_proba(sample)[0, 1]
print(f"Chance of rain tomorrow: {prob_rain:.2%}")
PY


# ───────────────────────────  REPO OVERVIEW (cat)  ─────────────────────────
$ cat README.md
Rainfall Prediction Classifier
──────────────────────────────
A scikit-learn pipeline that predicts next-day rainfall for Melbourne, Australia,
with full EDA, feature engineering, preprocessing and model tuning.

OBJECTIVE
  Predict rain vs no-rain while maintaining recall on the minority (rain) class.

DATASET
  • Australian Bureau of Meteorology – Kaggle “Rain in Australia”
  • ~33 k rows (2007-2017, Melbourne Airport)
  • Target: RainTomorrow (Yes / No)

APPROACH
  1. Exploratory data analysis (79 % dry / 21 % rain imbalance)
  2. Feature engineering: season, rainfall lag, one-hot categoricals
  3. Preprocessing: ColumnTransformer → impute + scale / one-hot
  4. Models: RandomForestClassifier (main) and LogisticRegression (baseline)
     tuned with 5-fold stratified GridSearchCV
  5. Metrics: accuracy, precision/recall/F1, ROC-AUC, calibration, confusion matrix
  6. Interpretability: feature importances + SHAP

KEY RESULTS
  • Random Forest: 0.84 test accuracy, F1(rain) ≈ 0.60, ROC-AUC ≈ 0.88
  • Beats 0.79 always-dry baseline
  • Top predictors: location, rainfall lag, humidity 3 pm, season

ROAD-MAP
  • Add radar imagery features
  • Try LightGBM / XGBoost with class-weighted loss
  • Deploy REST API (FastAPI + Docker)
  • Streamlit dashboard for live forecasts

LICENSE: MIT


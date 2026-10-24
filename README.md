# 🧠 Why Am I Procrastinating?

> A college-level explainable AI/ML project.  
> Predicts the most likely contributing factor behind your procrastination
> using a supervised multi-class classification pipeline.

---

## ⚠️ Data Disclaimer

All models in this project are trained on **programmatically generated
synthetic data**.  This data is created for the sole purpose of testing and
demonstrating the ML pipeline.  It is **not** real survey or clinical data and
**must not** be presented as scientific or research findings.

---

## 📁 Project Structure

```
why-am-i-procrastinating/
│
├── data/
│   ├── raw/                          # Input CSV (synthetic demo or real data)
│   └── processed/
│       ├── figures/                  # Generated charts (confusion matrices, SHAP, …)
│       └── evaluation_report.txt     # Full text evaluation report
│
├── models/
│   ├── random_forest.pkl             # Primary model (Random Forest)
│   ├── logistic_regression.pkl       # Baseline model
│   ├── xgboost.pkl                   # Advanced model
│   ├── label_encoder.pkl             # Fitted LabelEncoder
│   └── scaler.pkl                    # Fitted StandardScaler
│
├── notebooks/                        # Jupyter notebooks for EDA (optional)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Data generation, loading, encoding, splitting
│   ├── train.py                      # Model training (LR, RF, XGBoost)
│   ├── evaluate.py                   # Metrics, confusion matrices, CV, charts
│   ├── predict.py                    # Single-sample inference + SHAP
│   ├── explain.py                    # Global SHAP + feature importance figures
│   └── recommendations.py            # Per-class personalised advice
│
├── app.py                            # Streamlit web application (entry point)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Create & activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the synthetic demo dataset

```bash
python -m src.preprocessing
```

### 4. Train all three models

```bash
python -m src.train
```

### 5. Evaluate models + generate charts

```bash
python -m src.evaluate
```

### 6. Generate SHAP explainability figures

```bash
python -m src.explain
```

### 7. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 🤖 ML Problem

| Property | Value |
|---|---|
| Problem type | Supervised multi-class classification |
| Target column | `procrastination_reason` |
| Number of classes | 8 |
| Number of features | 12 |
| Train / test split | 80 % / 20 % (stratified) |
| Cross-validation | 5-fold on Random Forest |

### Target Classes

| # | Class |
|---|-------|
| 1 | Digital Distraction |
| 2 | Low Motivation |
| 3 | Task Difficulty |
| 4 | Fear of Failure |
| 5 | Perfectionism |
| 6 | Poor Sleep / Low Energy |
| 7 | Lack of Clarity |
| 8 | Poor Time Management |

### Input Features

| Feature | Type | Range |
|---------|------|-------|
| `sleep_hours` | float | 2.0 – 10.0 |
| `task_difficulty` | int | 1 – 10 |
| `task_interest` | int | 1 – 10 |
| `stress_level` | int | 1 – 10 |
| `phone_distraction` | int | 1 – 10 |
| `social_media_usage` | float | 0.0 – 8.0 |
| `energy_level` | int | 1 – 10 |
| `task_clarity` | int | 1 – 10 |
| `confidence` | int | 1 – 10 |
| `deadline_pressure` | int | 1 – 10 |
| `perfectionism` | int | 1 – 10 |
| `previous_procrastination` | int | 1 – 10 |

---

## 🔍 Explainability

- **RF feature importances** — mean decrease in impurity (built-in)
- **SHAP TreeExplainer** — exact SHAP values for Random Forest
  - Global bar chart (mean |SHAP| across all classes)
  - Multi-class beeswarm summary plot
  - Per-prediction waterfall chart (rendered live in the Streamlit app)

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt).

Key libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`,
`joblib`, `matplotlib`, `seaborn`, `plotly`, `streamlit`

---

*Built as an educational AI/ML portfolio project using synthetic data.*

# EDA Report — Why Am I Procrastinating?

> **Generated**: 2026-10-01 11:14  
> **Dataset**: `data/raw/demo_dataset.csv`  
> **⚠️ Synthetic demo data only — not real collected data**

---

## ⚠️ Important Caveats

All findings in this report are derived from a **synthetically generated**
dataset and reflect the statistical patterns of the generator, NOT real human
behaviour.  Association patterns (correlations, class separations) observed
here **do not imply causation** and must not be interpreted as psychological
or scientific findings.

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Rows | 1,000 |
| Columns (total) | 14 |
| Feature columns | 12 |
| Target column | `procrastination_reason` |
| Memory usage | 240.1 KB |

## 2. Data Quality

| Check | Result |
|---|---|
| Missing values | None detected ✓ |
| Duplicate rows | 0 (0.0%) |

## 3. Target Class Distribution

Balance ratio (min / max class count): **0.931**  
✓ Classes are approximately balanced.

| Class | Count | % |
|---|---|---|
| Digital Distraction | 123 | 12.3% |
| Fear of Failure | 130 | 13.0% |
| Lack of Clarity | 128 | 12.8% |
| Low Motivation | 121 | 12.1% |
| Perfectionism | 125 | 12.5% |
| Poor Sleep / Low Energy | 127 | 12.7% |
| Poor Time Management | 125 | 12.5% |
| Task Difficulty | 121 | 12.1% |

## 4. Feature Distributions

Histograms (Figure 02) show the marginal distribution of each feature across
all classes combined.  Most features span their full valid range. Skewness
values near 0 indicate approximately symmetric distributions; large positive
skew means a long right tail; large negative skew means a long left tail.

| Feature | Mean | Std | Min | Median | Max | Skew |
|---|---|---|---|---|---|---|
| Sleep Hours | 6.10 | 1.16 | 2.6 | 6.1 | 9.5 | -0.162 |
| Task Difficulty | 5.97 | 1.66 | 1.0 | 6.0 | 10.0 | -0.146 |
| Task Interest | 4.95 | 1.57 | 1.0 | 5.0 | 10.0 | 0.155 |
| Stress Level | 6.43 | 1.54 | 2.0 | 6.0 | 10.0 | -0.136 |
| Phone Distraction | 5.59 | 1.67 | 1.0 | 6.0 | 10.0 | 0.048 |
| Social Media (hrs/day) | 4.10 | 1.47 | 0.0 | 4.1 | 8.0 | 0.028 |
| Energy Level | 4.68 | 1.68 | 1.0 | 5.0 | 10.0 | 0.144 |
| Task Clarity | 4.75 | 1.62 | 1.0 | 5.0 | 10.0 | -0.071 |
| Confidence | 4.12 | 1.50 | 1.0 | 4.0 | 8.0 | 0.188 |
| Deadline Pressure | 6.03 | 1.61 | 1.0 | 6.0 | 10.0 | -0.150 |
| Perfectionism | 5.44 | 1.80 | 1.0 | 5.0 | 10.0 | 0.106 |
| Prev. Procrastination | 6.59 | 1.20 | 3.0 | 7.0 | 10.0 | -0.231 |

## 5. Correlation Analysis

> ⚠️ **Correlation does not imply causation.**  All associations below are observational and may reflect the synthetic data generator's structure.

Strongest correlated feature pair: **`phone_distraction` ↔ `social_media_usage`  (r = +0.463)**

The correlation heatmap (Figure 04) shows Pearson r values between all feature
pairs.  Features with inter-correlation near 0 provide largely independent
information to the classifier.  Highly correlated pairs may contribute
redundant information.

## 6. Outlier Analysis

Outliers were detected using the IQR fence method (Q1 − 1.5×IQR, Q3 +
1.5×IQR).  For bounded Likert-scale features (1–10), outliers within the valid
range represent the statistical tails of the distribution rather than data
errors.  No rows were removed.

| Feature | IQR Outliers |
|---|---|
| Sleep Hours | 4 |
| Task Difficulty | 4 |
| Task Interest | 2 |
| Stress Level | 0 |
| Phone Distraction | 0 |
| Social Media (hrs/day) | 0 |
| Energy Level | 3 |
| Task Clarity | 1 |
| Confidence | 0 |
| Deadline Pressure | 1 |
| Perfectionism | 0 |
| Prev. Procrastination | 80 |

## 7. Feature vs Target

Violin plots (Figure 05) for the top most discriminative features (ranked by
ANOVA F-statistic on the actual data) show that certain features have clearly
different distributions across classes.  This gives the ML models a meaningful
signal to learn from, even in synthetic data.  Features with heavily
overlapping violins across all classes provide weaker class discrimination on
their own.

> ⚠️ Feature–class associations observed here do not prove that a feature *causes* a procrastination type.

## 8. Generated Figures

- `reports/figures/01_class_distribution.png`
- `reports/figures/02_feature_histograms.png`
- `reports/figures/04_correlation_heatmap.png`
- `reports/figures/03_feature_boxplots.png`
- `reports/figures/07_outlier_summary.png`
- `reports/figures/05_top_features_vs_target.png`
- `reports/figures/06_all_features_vs_target.png`
- `reports/figures/08_pairplot_top_features.png`

---

## 9. Next Steps

1. Replace synthetic demo data with a **real, properly collected dataset** (e.g. Procrastination Assessment Scale for Students, PASS).
2. Re-run EDA to validate that real data patterns match model assumptions.
3. Re-train and re-evaluate all models on the real dataset.
4. Do not report any model metrics obtained from synthetic data as real findings.

"""
src/eda.py
===========
Exploratory Data Analysis (EDA) for the "Why Am I Procrastinating?" dataset.

All statistics and visualisations are computed directly from the supplied
dataset — nothing is hard-coded.  Results therefore reflect the actual
data passed in (synthetic demo data or a real replacement dataset).

⚠️  Correlation ≠ Causation
    All associations reported here are observational.  A strong correlation
    between two features does not imply that one causes the other.

Analysis steps
--------------
 1. Dataset overview       — shape, dtypes, memory usage
 2. Data types             — feature type breakdown
 3. Missing values         — null counts and percentages
 4. Duplicate rows         — exact-row duplicate detection
 5. Target distribution    — class counts, balance ratio
 6. Feature distributions  — descriptive statistics + histograms
 7. Correlation analysis   — Pearson correlation heatmap
 8. Outlier analysis       — IQR-based detection + box plots
 9. Feature vs target      — per-class violin / box plots

Outputs saved to ``reports/figures/``
--------------------------------------
  01_class_distribution.png
  02_feature_histograms.png
  03_feature_boxplots.png
  04_correlation_heatmap.png
  05_top_features_vs_target.png
  06_all_features_vs_target.png
  07_outlier_summary.png
  08_pairplot_top_features.png

A Markdown summary is saved to ``reports/eda_summary.md``.

Public API
----------
run_eda(path)           — run the complete EDA and return a stats dict
"""

from __future__ import annotations

import os
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Import project constants ───────────────────────────────────────────────────
from src.preprocessing import FEATURE_COLS, TARGET_COL, FEATURE_RANGES, CLASS_LABELS

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

REPORTS_DIR: str = "reports"
FIGURES_DIR: str = os.path.join(REPORTS_DIR, "figures")
SUMMARY_PATH: str = os.path.join(REPORTS_DIR, "eda_summary.md")

#: Default dataset path (synthetic demo data).
_DEFAULT_DATA = os.path.join("data", "raw", "demo_dataset.csv")

#: Seaborn palette — one colour per class, consistent across all plots.
_CLASS_PALETTE = "tab10"

# Matplotlib global style
plt.rcParams.update({
    "figure.dpi":        120,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.titlesize":  13,
    "figure.titleweight": "bold",
})

# Human-readable feature labels (for axis labels)
_FEAT_LABEL: dict[str, str] = {
    "sleep_hours":               "Sleep Hours",
    "task_difficulty":           "Task Difficulty",
    "task_interest":             "Task Interest",
    "stress_level":              "Stress Level",
    "phone_distraction":         "Phone Distraction",
    "social_media_usage":        "Social Media (hrs/day)",
    "energy_level":              "Energy Level",
    "task_clarity":              "Task Clarity",
    "confidence":                "Confidence",
    "deadline_pressure":         "Deadline Pressure",
    "perfectionism":             "Perfectionism",
    "previous_procrastination":  "Prev. Procrastination",
}


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, filename: str) -> str:
    """Save a figure and close it.  Returns the full save path."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [eda] Saved → {path}")
    return path


def _section(title: str) -> None:
    """Print a section header to stdout."""
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _wrap(text: str, width: int = 78) -> str:
    """Word-wrap a string for the Markdown report."""
    return textwrap.fill(text, width=width)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Dataset overview
# ══════════════════════════════════════════════════════════════════════════════

def analyze_overview(df: pd.DataFrame) -> dict:
    """
    Compute basic dataset statistics: shape, memory, dtypes, sentinel column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys: ``n_rows``, ``n_cols``, ``memory_kb``,
                    ``dtype_counts``, ``has_sentinel``
    """
    _section("1 · DATASET OVERVIEW")

    n_rows, n_cols = df.shape
    mem_kb = df.memory_usage(deep=True).sum() / 1024

    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_str    = {str(k): v for k, v in dtype_counts.items()}

    has_sentinel = "data_source" in df.columns
    if has_sentinel:
        sentinel_val = df["data_source"].iloc[0]
        print(f"  data_source sentinel : '{sentinel_val}'")

    print(f"  Rows         : {n_rows:,}")
    print(f"  Columns      : {n_cols}")
    print(f"  Memory usage : {mem_kb:.1f} KB")
    print(f"  dtypes       : {dtype_str}")

    return {
        "n_rows":       n_rows,
        "n_cols":       n_cols,
        "memory_kb":    round(mem_kb, 1),
        "dtype_counts": dtype_str,
        "has_sentinel": has_sentinel,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Missing values
# ══════════════════════════════════════════════════════════════════════════════

def analyze_missing(df: pd.DataFrame) -> dict:
    """
    Count and report null values for every required column.

    Returns
    -------
    dict mapping column names with >0 nulls to their null count.
    Empty dict = no missing values.
    """
    _section("3 · MISSING VALUES")

    cols   = FEATURE_COLS + [TARGET_COL]
    nulls  = df[cols].isnull().sum()
    missing = nulls[nulls > 0].to_dict()

    if missing:
        for col, n in missing.items():
            print(f"  {col:<30} {n:>4} nulls  ({100*n/len(df):.1f}%)")
    else:
        print("  ✓ No missing values in required columns.")

    return missing


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Duplicate rows
# ══════════════════════════════════════════════════════════════════════════════

def analyze_duplicates(df: pd.DataFrame) -> dict:
    """
    Detect exact duplicate rows across all feature + target columns.

    Returns
    -------
    dict with keys ``n_duplicates``, ``pct_duplicates``.
    """
    _section("4 · DUPLICATE ROWS")

    feature_target_cols = FEATURE_COLS + [TARGET_COL]
    n_dup = int(df[feature_target_cols].duplicated().sum())
    pct   = 100 * n_dup / len(df)

    print(f"  Duplicate rows : {n_dup} ({pct:.1f}%)")
    if n_dup > 0:
        warnings.warn(
            f"[eda] {n_dup} duplicate rows found.  "
            "Consider deduplicating before training.",
            UserWarning, stacklevel=2,
        )

    return {"n_duplicates": n_dup, "pct_duplicates": round(pct, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Target class distribution
# ══════════════════════════════════════════════════════════════════════════════

def analyze_target_distribution(df: pd.DataFrame) -> dict:
    """
    Compute class counts, percentages, and balance ratio.

    Balance ratio = min_count / max_count.  A value of 1.0 is perfectly
    balanced; values below 0.5 indicate notable imbalance.

    Returns
    -------
    dict with keys ``counts``, ``percentages``, ``balance_ratio``.
    """
    _section("5 · TARGET CLASS DISTRIBUTION")

    counts = df[TARGET_COL].value_counts().sort_index()
    pcts   = (100 * counts / counts.sum()).round(2)
    ratio  = round(counts.min() / counts.max(), 3)

    for cls in counts.index:
        bar = "█" * int(pcts[cls] / 2)
        print(f"  {cls:<30} {counts[cls]:>4}  {pcts[cls]:>5.1f}%  {bar}")

    print(f"\n  Balance ratio (min/max) : {ratio:.3f}"
          f"  {'✓ Balanced' if ratio >= 0.8 else '⚠ Imbalanced'}")

    return {
        "counts":        counts.to_dict(),
        "percentages":   pcts.to_dict(),
        "balance_ratio": ratio,
    }


def plot_class_distribution(df: pd.DataFrame) -> str:
    """
    Bar chart of target class frequencies.

    Summary: Shows how many records belong to each procrastination class.
    Balanced classes are important for unbiased classifier training.
    """
    counts  = df[TARGET_COL].value_counts().sort_values(ascending=False)
    palette = sns.color_palette(_CLASS_PALETTE, len(counts))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(counts.index, counts.values, color=palette,
                  edgecolor="white", linewidth=0.7)

    # Annotate with count + percentage
    total = counts.sum()
    for bar, (cls, cnt) in zip(bars, counts.items()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.003,
                f"{cnt}\n({100*cnt/total:.1f}%)",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Reference line at perfectly balanced count
    ax.axhline(total / len(counts), color="crimson", linestyle="--",
               linewidth=1.2, label=f"Perfect balance ({total//len(counts)})")

    ax.set_title("Target Class Distribution\n"
                 "⚠ Synthetic demo data — not real survey results")
    ax.set_xlabel("Procrastination Reason")
    ax.set_ylabel("Number of Records")
    ax.legend(fontsize=9)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    return _save(fig, "01_class_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Feature distributions
# ══════════════════════════════════════════════════════════════════════════════

def analyze_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for every numeric feature.

    Returns
    -------
    pd.DataFrame  (features × stats: mean, std, min, 25%, 50%, 75%, max, skew)
    """
    _section("6 · FEATURE DISTRIBUTIONS")

    stats = df[FEATURE_COLS].describe().T
    stats["skew"] = df[FEATURE_COLS].skew().round(3)
    stats = stats.round(3)

    print(stats[["mean", "std", "min", "50%", "max", "skew"]].to_string())
    return stats


def plot_feature_histograms(df: pd.DataFrame) -> str:
    """
    4 × 3 grid of histograms with KDE overlay for all 12 features.

    Summary: Histograms show the marginal distribution of each feature
    across ALL classes combined.  A roughly bell-shaped distribution
    suggests the feature spans its range naturally.  Very skewed or
    bimodal histograms indicate the feature may cluster differently
    across classes — useful diagnostic before modelling.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes_flat = axes.flatten()

    for ax, feat in zip(axes_flat, FEATURE_COLS):
        lo, hi = FEATURE_RANGES[feat]
        data   = df[feat].dropna()

        ax.hist(data, bins=20, color="#4C72B0", edgecolor="white",
                linewidth=0.5, density=True, alpha=0.75, label="Histogram")

        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde   = gaussian_kde(data)
            xs    = np.linspace(lo, hi, 200)
            ax.plot(xs, kde(xs), color="crimson", linewidth=1.5, label="KDE")
        except ImportError:
            pass   # scipy optional — skip KDE if unavailable

        ax.axvline(data.mean(), color="darkorange", linestyle="--",
                   linewidth=1.2, label=f"Mean={data.mean():.1f}")
        ax.axvline(data.median(), color="green", linestyle=":",
                   linewidth=1.2, label=f"Median={data.median():.1f}")

        ax.set_title(_FEAT_LABEL[feat])
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("Feature Distributions (All Classes Combined)\n"
                 "⚠ Synthetic demo data", y=1.01)
    plt.tight_layout()
    return _save(fig, "02_feature_histograms.png")


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Outlier analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_outliers(df: pd.DataFrame) -> dict:
    """
    Detect outliers using the IQR (interquartile range) method.

    A value is flagged as an outlier if it falls below
    Q1 − 1.5 × IQR  or above  Q3 + 1.5 × IQR.

    ⚠️  Note: the IQR method flags statistical extremes within the dataset,
    not necessarily erroneous values.  For bounded Likert-scale features
    (1–10) all valid values are in range, so the IQR outliers represent
    the tails of within-range distributions.

    Returns
    -------
    dict mapping feature name → outlier count.
    """
    _section("8 · OUTLIER ANALYSIS (IQR METHOD)")

    result = {}
    for feat in FEATURE_COLS:
        q1, q3  = df[feat].quantile(0.25), df[feat].quantile(0.75)
        iqr     = q3 - q1
        lo_iqr  = q1 - 1.5 * iqr
        hi_iqr  = q3 + 1.5 * iqr
        n_out   = int(((df[feat] < lo_iqr) | (df[feat] > hi_iqr)).sum())
        pct_out = 100 * n_out / len(df)
        result[feat] = n_out
        flag = "⚠" if n_out > 0 else "✓"
        print(f"  {flag} {feat:<30} {n_out:>4} outliers ({pct_out:.1f}%)"
              f"  fence [{lo_iqr:.1f}, {hi_iqr:.1f}]")

    return result


def plot_feature_boxplots(df: pd.DataFrame, outlier_counts: dict) -> str:
    """
    4 × 3 box-plot grid — one box per feature, coloured by outlier count.

    Summary: Box plots show the median, IQR (box), whiskers (1.5 × IQR),
    and individual outlier points.  Features with many outlier dots often
    have heavy tails or a bimodal distribution caused by mixing data from
    very different class profiles.  Clipping at the domain boundary
    (e.g. 1–10) concentrates values at the edge and can produce apparent
    outliers near the limits.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes_flat = axes.flatten()
    max_out   = max(outlier_counts.values()) if outlier_counts else 1

    for ax, feat in zip(axes_flat, FEATURE_COLS):
        n_out   = outlier_counts.get(feat, 0)
        # Colour: white (0 outliers) → red (max outliers)
        colour  = plt.cm.RdYlGn_r(n_out / max(max_out, 1))
        bp      = ax.boxplot(df[feat].dropna(), vert=True, patch_artist=True,
                             medianprops={"color": "black", "linewidth": 2},
                             flierprops={"marker": "o", "markerfacecolor": "crimson",
                                         "markersize": 3, "alpha": 0.5})
        bp["boxes"][0].set_facecolor(colour)
        ax.set_title(f"{_FEAT_LABEL[feat]}\n({n_out} IQR outliers)", fontsize=9)
        ax.set_ylabel("Value")

    fig.suptitle("Feature Box Plots — IQR Outlier Detection\n"
                 "⚠ Synthetic demo data", y=1.01)
    plt.tight_layout()
    return _save(fig, "03_feature_boxplots.png")


def plot_outlier_summary(outlier_counts: dict) -> str:
    """
    Horizontal bar chart summarising outlier counts per feature.

    Summary: Quickly identifies which features have the most statistical
    outliers under the IQR rule.  High outlier counts in bounded features
    (1–10 scale) often reflect strongly skewed within-class distributions
    or deliberate signal in the synthetic generator profiles.
    """
    feats  = list(outlier_counts.keys())
    counts = list(outlier_counts.values())
    order  = np.argsort(counts)
    palette = ["#d62728" if counts[i] > 0 else "#2ca02c" for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh([_FEAT_LABEL[feats[i]] for i in order],
                   [counts[i] for i in order],
                   color=palette, edgecolor="white", linewidth=0.5)
    for bar, idx in zip(bars, order):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(counts[idx]), va="center", fontsize=9)

    ax.set_xlabel("Number of IQR Outlier Records")
    ax.set_title("Outlier Count per Feature (IQR Method)\n"
                 "⚠ Synthetic demo data")
    plt.tight_layout()
    return _save(fig, "07_outlier_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Correlation analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix for all numeric features.

    ⚠️  Pearson correlation measures *linear* association only.
        A high correlation does not imply causation.

    Returns
    -------
    pd.DataFrame — symmetric correlation matrix, shape (12, 12).
    """
    _section("7 · CORRELATION ANALYSIS")

    corr = df[FEATURE_COLS].corr()

    # Report the strongest pairs (|r| > 0.3)
    pairs = []
    for i, f1 in enumerate(FEATURE_COLS):
        for f2 in FEATURE_COLS[i+1:]:
            r = corr.loc[f1, f2]
            if abs(r) > 0.3:
                pairs.append((f1, f2, r))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if pairs:
        print("  Pairs with |r| > 0.30:")
        for f1, f2, r in pairs:
            direction = "positive" if r > 0 else "negative"
            print(f"    {f1:<30} ↔ {f2:<30}  r = {r:+.3f}  ({direction})")
        print("\n  ⚠️  Correlation does not imply causation.")
    else:
        print("  No feature pairs with |r| > 0.30 found.")

    return corr


def plot_correlation_heatmap(corr: pd.DataFrame) -> str:
    """
    Annotated Pearson correlation heatmap for all 12 features.

    Summary: Each cell shows the linear correlation coefficient r between
    two features.  Values near +1 indicate strong positive linear
    association; near −1 indicate strong negative.  Near 0 means no
    linear relationship (there may still be non-linear relationships).

    ⚠️  Correlation does NOT imply causation.  These associations are
    observational and may reflect the structure of the synthetic generator
    rather than real behavioural relationships.
    """
    labels = [_FEAT_LABEL[f] for f in FEATURE_COLS]
    mask   = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle masked

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title(
        "Pearson Correlation Matrix — Numeric Features\n"
        "⚠ Correlation ≠ Causation  |  Synthetic demo data",
        pad=14,
    )
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return _save(fig, "04_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Feature vs target visualisations
# ══════════════════════════════════════════════════════════════════════════════

def _rank_features_by_class_separation(df: pd.DataFrame) -> list[str]:
    """
    Rank features by their ability to separate classes, measured as the
    ratio of between-class variance to within-class variance (a simple
    one-way ANOVA F-statistic proxy).

    Returns
    -------
    list[str] — feature names sorted by F-statistic descending.
    """
    from scipy.stats import f_oneway

    scores = {}
    groups_by_class = {
        cls: df.loc[df[TARGET_COL] == cls, FEATURE_COLS]
        for cls in df[TARGET_COL].unique()
    }
    for feat in FEATURE_COLS:
        groups = [g[feat].dropna().values for g in groups_by_class.values()]
        try:
            f_stat, _ = f_oneway(*groups)
            scores[feat] = float(f_stat) if np.isfinite(f_stat) else 0.0
        except Exception:
            scores[feat] = 0.0

    return sorted(scores, key=lambda f: scores[f], reverse=True)


def plot_top_features_vs_target(df: pd.DataFrame, n_top: int = 6) -> str:
    """
    Violin + box plots for the top N most discriminative features vs target class.

    The most discriminative features are selected automatically using a
    one-way ANOVA F-statistic on the actual data — not pre-specified.

    Summary: Each violin shows the full distribution of a feature broken
    down by procrastination class.  Wider sections = more records at
    that value.  Features where the violin shapes differ most between
    classes are the most useful discriminators for the classifier.

    ⚠️  Association between a feature and a class does not mean that
    feature *causes* that type of procrastination.
    """
    try:
        ranked = _rank_features_by_class_separation(df)[:n_top]
    except ImportError:
        # scipy unavailable — fall back to first n features
        ranked = FEATURE_COLS[:n_top]

    class_order = sorted(df[TARGET_COL].unique())
    palette     = sns.color_palette(_CLASS_PALETTE, len(class_order))

    n_cols = 2
    n_rows = (n_top + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes_flat = axes.flatten()

    for ax, feat in zip(axes_flat, ranked):
        sns.violinplot(
            data=df, x=TARGET_COL, y=feat,
            order=class_order, palette=palette,
            inner="box", ax=ax,
            cut=0, linewidth=0.8,
        )
        ax.set_title(f"{_FEAT_LABEL[feat]}  vs  Procrastination Class")
        ax.set_xlabel("")
        ax.set_ylabel(_FEAT_LABEL[feat])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right",
                           fontsize=7)
        lo, hi = FEATURE_RANGES[feat]
        ax.set_ylim(lo - 0.3, hi + 0.3)

    # Hide unused subplots
    for ax in axes_flat[n_top:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Top {n_top} Most Discriminative Features vs Procrastination Class\n"
        "(Ranked by ANOVA F-statistic)  |  ⚠ Correlation ≠ Causation",
        y=1.01,
    )
    plt.tight_layout()
    return _save(fig, "05_top_features_vs_target.png")


def plot_all_features_vs_target(df: pd.DataFrame) -> str:
    """
    Compact box-plot grid: all 12 features × all classes.

    Summary: Provides a single reference view of how every feature's
    distribution varies across all eight procrastination classes.
    Classes with distinctly shifted medians (the horizontal line inside
    each box) contribute the most discriminative signal to the ML models.
    """
    class_order = sorted(df[TARGET_COL].unique())
    palette     = sns.color_palette(_CLASS_PALETTE, len(class_order))

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    axes_flat = axes.flatten()

    for ax, feat in zip(axes_flat, FEATURE_COLS):
        sns.boxplot(
            data=df, x=TARGET_COL, y=feat,
            order=class_order, palette=palette,
            ax=ax, linewidth=0.8,
            flierprops={"markersize": 2, "alpha": 0.4},
        )
        ax.set_title(_FEAT_LABEL[feat], fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Value", fontsize=8)
        ax.set_xticklabels(
            [c.replace(" / ", "\n").replace(" ", "\n") for c in class_order],
            rotation=0, ha="center", fontsize=5.5,
        )
        lo, hi = FEATURE_RANGES[feat]
        ax.set_ylim(lo - 0.3, hi + 0.3)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    fig.suptitle(
        "All Features vs Procrastination Class (Box Plots)\n"
        "⚠ Synthetic demo data  |  Correlation ≠ Causation",
        y=1.01, fontsize=13,
    )
    plt.tight_layout()
    return _save(fig, "06_all_features_vs_target.png")


def plot_pairplot_top_features(df: pd.DataFrame, n_top: int = 4) -> str:
    """
    Seaborn pairplot of the top N most discriminative features, coloured by class.

    Summary: Each off-diagonal scatter plot shows whether two features
    together can separate the classes into distinct clusters.  Tight,
    well-separated clusters indicate that the feature *pair* is a strong
    combined predictor.  Heavily overlapping clouds indicate the classifier
    will need many features together to discriminate well.

    ⚠️  Pairwise separability shown here does not mean causation.
    """
    try:
        ranked = _rank_features_by_class_separation(df)[:n_top]
    except ImportError:
        ranked = FEATURE_COLS[:n_top]

    cols  = ranked + [TARGET_COL]
    sub   = df[cols].copy()
    # shorten class labels for readability inside the pairplot
    sub[TARGET_COL] = sub[TARGET_COL].str.replace(" / ", "/").str.replace(
        "Poor Sleep/Low Energy", "Sleep/Energy"
    )

    g = sns.pairplot(
        sub,
        hue=TARGET_COL,
        vars=ranked,
        plot_kws={"alpha": 0.35, "s": 12},
        diag_kind="kde",
        palette=_CLASS_PALETTE,
        height=2.2,
    )
    g.figure.suptitle(
        f"Pairplot — Top {n_top} Discriminative Features\n"
        "⚠ Synthetic demo data  |  Correlation ≠ Causation",
        y=1.02, fontsize=11,
    )
    path = os.path.join(FIGURES_DIR, "08_pairplot_top_features.png")
    g.figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  [eda] Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Markdown summary report
# ══════════════════════════════════════════════════════════════════════════════

def generate_eda_report(
    path: str,
    stats: dict,
    saved_figures: list[str],
) -> None:
    """
    Write a Markdown EDA summary report to ``reports/eda_summary.md``.

    All statistics are drawn from the ``stats`` dict produced during analysis
    — nothing is hard-coded.

    Parameters
    ----------
    path : str
        Original dataset path (for reference in the report).
    stats : dict
        Aggregated statistics from all analysis steps.
    saved_figures : list[str]
        List of figure file paths that were saved.
    """
    overview  = stats.get("overview", {})
    missing   = stats.get("missing", {})
    dups      = stats.get("duplicates", {})
    target    = stats.get("target_dist", {})
    feat_stats = stats.get("feature_stats")
    outliers  = stats.get("outliers", {})
    corr      = stats.get("correlation")

    n_rows         = overview.get("n_rows", "?")
    n_cols         = overview.get("n_cols", "?")
    balance_ratio  = target.get("balance_ratio", "?")
    n_dups         = dups.get("n_duplicates", "?")
    n_missing_cols = len(missing)

    # Most-correlated pair (if any)
    top_corr_str = "None above 0.30"
    if corr is not None:
        pairs = []
        for i, f1 in enumerate(FEATURE_COLS):
            for f2 in FEATURE_COLS[i+1:]:
                r = corr.loc[f1, f2]
                if abs(r) > 0.3:
                    pairs.append((f1, f2, r))
        if pairs:
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            f1, f2, r = pairs[0]
            top_corr_str = f"`{f1}` ↔ `{f2}`  (r = {r:+.3f})"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# EDA Report — Why Am I Procrastinating?",
        "",
        f"> **Generated**: {timestamp}  ",
        f"> **Dataset**: `{path}`  ",
        f"> **⚠️ Synthetic demo data only — not real collected data**",
        "",
        "---",
        "",
        "## ⚠️ Important Caveats",
        "",
        _wrap(
            "All findings in this report are derived from a **synthetically generated** "
            "dataset and reflect the statistical patterns of the generator, NOT real "
            "human behaviour.  Association patterns (correlations, class separations) "
            "observed here **do not imply causation** and must not be interpreted as "
            "psychological or scientific findings."
        ),
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        f"| Property | Value |",
        f"|---|---|",
        f"| Rows | {n_rows:,} |",
        f"| Columns (total) | {n_cols} |",
        f"| Feature columns | {len(FEATURE_COLS)} |",
        f"| Target column | `{TARGET_COL}` |",
        f"| Memory usage | {overview.get('memory_kb', '?'):.1f} KB |",
        "",
        "## 2. Data Quality",
        "",
        f"| Check | Result |",
        f"|---|---|",
        f"| Missing values | {'None detected ✓' if n_missing_cols == 0 else f'{n_missing_cols} column(s) affected'} |",
        f"| Duplicate rows | {n_dups} ({dups.get('pct_duplicates', 0):.1f}%) |",
        "",
    ]

    if missing:
        lines += ["**Missing value details:**", ""]
        for col, cnt in missing.items():
            lines.append(f"- `{col}`: {cnt} nulls ({100*cnt/n_rows:.1f}%)")
        lines.append("")

    lines += [
        "## 3. Target Class Distribution",
        "",
        f"Balance ratio (min / max class count): **{balance_ratio}**  ",
        ("✓ Classes are approximately balanced." if float(balance_ratio) >= 0.8
         else "⚠️ Notable class imbalance detected — consider `class_weight='balanced'`."),
        "",
        "| Class | Count | % |",
        "|---|---|---|",
    ]
    for cls, cnt in target.get("counts", {}).items():
        pct = target.get("percentages", {}).get(cls, 0)
        lines.append(f"| {cls} | {cnt} | {pct}% |")

    lines += [
        "",
        "## 4. Feature Distributions",
        "",
        _wrap(
            "Histograms (Figure 02) show the marginal distribution of each feature "
            "across all classes combined.  Most features span their full valid range. "
            "Skewness values near 0 indicate approximately symmetric distributions; "
            "large positive skew means a long right tail; large negative skew means a "
            "long left tail."
        ),
        "",
    ]

    if feat_stats is not None:
        lines += [
            "| Feature | Mean | Std | Min | Median | Max | Skew |",
            "|---|---|---|---|---|---|---|",
        ]
        for feat in FEATURE_COLS:
            row = feat_stats.loc[feat]
            lines.append(
                f"| {_FEAT_LABEL[feat]} "
                f"| {row['mean']:.2f} | {row['std']:.2f} "
                f"| {row['min']:.1f} | {row['50%']:.1f} "
                f"| {row['max']:.1f} | {row['skew']:.3f} |"
            )
        lines.append("")

    lines += [
        "## 5. Correlation Analysis",
        "",
        "> ⚠️ **Correlation does not imply causation.**  All associations below are "
        "observational and may reflect the synthetic data generator's structure.",
        "",
        f"Strongest correlated feature pair: **{top_corr_str}**",
        "",
        _wrap(
            "The correlation heatmap (Figure 04) shows Pearson r values between all "
            "feature pairs.  Features with inter-correlation near 0 provide largely "
            "independent information to the classifier.  Highly correlated pairs may "
            "contribute redundant information."
        ),
        "",
        "## 6. Outlier Analysis",
        "",
        _wrap(
            "Outliers were detected using the IQR fence method "
            "(Q1 − 1.5×IQR, Q3 + 1.5×IQR).  For bounded Likert-scale features (1–10), "
            "outliers within the valid range represent the statistical tails of the "
            "distribution rather than data errors.  No rows were removed."
        ),
        "",
        "| Feature | IQR Outliers |",
        "|---|---|",
    ]

    for feat, cnt in outliers.items():
        lines.append(f"| {_FEAT_LABEL.get(feat, feat)} | {cnt} |")

    lines += [
        "",
        "## 7. Feature vs Target",
        "",
        _wrap(
            "Violin plots (Figure 05) for the top most discriminative features "
            "(ranked by ANOVA F-statistic on the actual data) show that certain "
            "features have clearly different distributions across classes.  "
            "This gives the ML models a meaningful signal to learn from, even in "
            "synthetic data.  Features with heavily overlapping violins across all "
            "classes provide weaker class discrimination on their own."
        ),
        "",
        "> ⚠️ Feature–class associations observed here do not prove that a feature "
        "*causes* a procrastination type.",
        "",
        "## 8. Generated Figures",
        "",
    ]

    for fp in saved_figures:
        lines.append(f"- `{fp}`")

    lines += [
        "",
        "---",
        "",
        "## 9. Next Steps",
        "",
        "1. Replace synthetic demo data with a **real, properly collected dataset** "
        "(e.g. Procrastination Assessment Scale for Students, PASS).",
        "2. Re-run EDA to validate that real data patterns match model assumptions.",
        "3. Re-train and re-evaluate all models on the real dataset.",
        "4. Do not report any model metrics obtained from synthetic data as real findings.",
        "",
    ]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"\n  [eda] Markdown summary → {SUMMARY_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# Public API — full EDA runner
# ══════════════════════════════════════════════════════════════════════════════

def run_eda(path: str = _DEFAULT_DATA) -> dict:
    """
    Run the complete EDA pipeline on the dataset at ``path``.

    Steps executed
    --------------
    1  Dataset overview
    2  Data types (part of overview)
    3  Missing values
    4  Duplicate rows
    5  Target class distribution + bar chart
    6  Feature distributions + histograms
    7  Correlation analysis + heatmap
    8  Outlier analysis + box plots + summary chart
    9  Feature vs target (violin plots, pairplot)
       Markdown summary report

    Parameters
    ----------
    path : str
        Path to the input CSV.  Defaults to the synthetic demo dataset.

    Returns
    -------
    dict
        All computed statistics, including:
        ``overview``, ``missing``, ``duplicates``, ``target_dist``,
        ``feature_stats``, ``correlation``, ``outliers``.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at ``path``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[eda] Dataset not found at '{path}'.\n"
            "  Generate the demo dataset first:\n"
            "    python -m src.data_generator"
        )

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print(f"  Dataset : {path}")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(path)

    saved_figures: list[str] = []
    stats: dict = {}

    # ── Steps 1–2: Overview + dtypes ──────────────────────────────────────────
    stats["overview"] = analyze_overview(df)

    # Drop the sentinel column for numeric analysis (it is non-numeric)
    df_clean = df.drop(columns=["data_source"], errors="ignore")

    # ── Step 3: Missing values ─────────────────────────────────────────────────
    stats["missing"] = analyze_missing(df_clean)

    # ── Step 4: Duplicates ─────────────────────────────────────────────────────
    stats["duplicates"] = analyze_duplicates(df_clean)

    # ── Step 5: Target distribution ───────────────────────────────────────────
    stats["target_dist"] = analyze_target_distribution(df_clean)
    saved_figures.append(plot_class_distribution(df_clean))

    # ── Step 6: Feature distributions ─────────────────────────────────────────
    stats["feature_stats"] = analyze_feature_stats(df_clean)
    saved_figures.append(plot_feature_histograms(df_clean))

    # ── Step 7: Correlation ────────────────────────────────────────────────────
    corr = analyze_correlations(df_clean)
    stats["correlation"] = corr
    saved_figures.append(plot_correlation_heatmap(corr))

    # ── Step 8: Outliers ──────────────────────────────────────────────────────
    outliers = analyze_outliers(df_clean)
    stats["outliers"] = outliers
    saved_figures.append(plot_feature_boxplots(df_clean, outliers))
    saved_figures.append(plot_outlier_summary(outliers))

    # ── Step 9: Feature vs target ─────────────────────────────────────────────
    _section("9 · FEATURE vs TARGET VISUALISATIONS")
    saved_figures.append(plot_top_features_vs_target(df_clean, n_top=6))
    saved_figures.append(plot_all_features_vs_target(df_clean))
    saved_figures.append(plot_pairplot_top_features(df_clean, n_top=4))

    # ── Markdown summary ──────────────────────────────────────────────────────
    generate_eda_report(path, stats, saved_figures)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ EDA COMPLETE")
    print(f"  Figures saved : {len(saved_figures)}")
    print(f"  Report saved  : {SUMMARY_PATH}")
    print("=" * 60 + "\n")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run EDA for the 'Why Am I Procrastinating?' dataset."
    )
    parser.add_argument(
        "--data", type=str, default=_DEFAULT_DATA,
        help=f"Path to the CSV dataset (default: {_DEFAULT_DATA})",
    )
    args = parser.parse_args()

    # Auto-generate demo data if missing
    if not os.path.exists(args.data):
        print(f"[eda] '{args.data}' not found — generating synthetic demo data …")
        from src.data_generator import generate_demo_dataset
        generate_demo_dataset(output_path=args.data)

    run_eda(args.data)

"""
src/evaluate.py
================
Evaluation pipeline for "Why Am I Procrastinating?"

Computes ALL metrics from the **actual trained models** applied to the
**held-out test set**.  Nothing is hard-coded or estimated.

Metrics computed (per model)
----------------------------
* Accuracy (overall)
* Per-class and averaged Precision, Recall, F1-score
  — both ``macro`` (unweighted mean across classes) and
    ``weighted`` (weighted by class support)
* Confusion matrix
* 5-fold stratified cross-validation accuracy (RF only)

Artefacts produced
------------------
reports/figures/cm_logistic_regression.png   — confusion matrix heatmap
reports/figures/cm_random_forest.png
reports/figures/cm_xgboost.png               — (if XGBoost trained)
reports/figures/model_comparison.png         — side-by-side accuracy bar chart
reports/figures/cv_rf_boxplot.png            — CV fold distribution
reports/evaluation_report.txt               — human-readable text report
reports/metrics.json                        — machine-readable metrics

⚠️  All metrics shown are computed on SYNTHETIC DEMO data unless you have
    replaced the dataset.  Do not present them as real-world results.

Public API
----------
evaluate_all(data)  -> dict
    Accepts the dict from ``src.train.train_all`` and returns a metrics dict.

CLI
---
    python -m src.evaluate [--data PATH] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe everywhere
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ══════════════════════════════════════════════════════════════════════════════
# Output paths
# ══════════════════════════════════════════════════════════════════════════════

REPORTS_DIR  = "reports"
FIGURES_DIR  = os.path.join(REPORTS_DIR, "figures")
REPORT_PATH  = os.path.join(REPORTS_DIR, "evaluation_report.txt")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")

# matplotlib global style
plt.rcParams.update({
    "figure.dpi":        130,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [evaluate] Saved → {path}")
    return path


def _section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-model evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_single(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    report_lines: list[str],
) -> dict:
    """
    Evaluate one model on the held-out test set.

    All metrics are computed directly from ``model.predict(X_test)``
    — nothing is invented.

    Parameters
    ----------
    name : str
        Human-readable model name (e.g. ``"Random Forest"``).
    model :
        Fitted sklearn-compatible classifier.
    X_test : np.ndarray
        Feature matrix for the test set (already in the correct space for
        this model — scaled for LR, raw for RF / XGBoost).
    y_test : np.ndarray
        True integer-encoded labels for the test set.
    class_names : list[str]
        Human-readable class names in label-encoder order.
    report_lines : list[str]
        Mutable list; text lines are appended in-place.

    Returns
    -------
    dict
        ``accuracy``, ``precision_macro``, ``recall_macro``, ``f1_macro``,
        ``precision_weighted``, ``recall_weighted``, ``f1_weighted``,
        ``per_class`` (dict of per-class metrics).
    """
    _section(name)
    t0 = time.perf_counter()

    y_pred = model.predict(X_test)

    # ── Core metrics ──────────────────────────────────────────────────────────
    accuracy = float(accuracy_score(y_test, y_pred))

    # classification_report gives a clean multi-class breakdown
    clf_report_str = classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    # Numeric per-class and averaged metrics
    prec_mc, rec_mc, f1_mc, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_test, y_pred, average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    per_class = {
        cls: {
            "precision": float(prec_pc[i]),
            "recall":    float(rec_pc[i]),
            "f1":        float(f1_pc[i]),
            "support":   int(sup_pc[i]),
        }
        for i, cls in enumerate(class_names)
    }

    elapsed = time.perf_counter() - t0

    # ── Console output ────────────────────────────────────────────────────────
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  F1 macro  : {float(f1_mc):.4f}")
    print(f"  F1 weighted: {float(f1_wt):.4f}")
    print(f"  Eval time : {elapsed:.2f}s")
    print(f"\n{clf_report_str}")

    # ── Append to text report ─────────────────────────────────────────────────
    sep = "=" * 60
    report_lines += [
        f"\n{sep}",
        name,
        sep,
        f"Accuracy            : {accuracy:.4f}  ({accuracy * 100:.2f}%)",
        f"Precision (macro)   : {float(prec_mc):.4f}",
        f"Recall    (macro)   : {float(rec_mc):.4f}",
        f"F1-score  (macro)   : {float(f1_mc):.4f}",
        f"Precision (weighted): {float(prec_wt):.4f}",
        f"Recall    (weighted): {float(rec_wt):.4f}",
        f"F1-score  (weighted): {float(f1_wt):.4f}",
        "",
        "Classification Report:",
        clf_report_str,
    ]

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    _plot_confusion_matrix(cm, class_names, name)

    return {
        "accuracy":            accuracy,
        "precision_macro":     float(prec_mc),
        "recall_macro":        float(rec_mc),
        "f1_macro":            float(f1_mc),
        "precision_weighted":  float(prec_wt),
        "recall_weighted":     float(rec_wt),
        "f1_weighted":         float(f1_wt),
        "per_class":           per_class,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    model_name: str,
) -> None:
    """
    Plot a normalised + absolute confusion matrix heatmap.

    Left panel  : raw counts (integer).
    Right panel : row-normalised proportions (recall per class).
    """
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    short   = [c.replace(" / ", "/").replace(" ", "\n") for c in class_names]
    slug    = model_name.lower().replace(" ", "_")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, fmt, cmap, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Blues", "YlOrRd"],
        ["Counts", "Normalised (recall per class)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap=cmap,
            xticklabels=short, yticklabels=short,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(f"{model_name}\n{subtitle}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        f"Confusion Matrix — {model_name}\n"
        "⚠️ Synthetic demo data  |  Do not interpret as real-world metrics",
        y=1.01, fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, f"cm_{slug}.png")


def _plot_model_comparison(metrics: dict) -> None:
    """
    Side-by-side grouped bar chart comparing accuracy, macro-F1, weighted-F1
    across all trained models.

    The random-baseline is drawn as a horizontal dashed reference.
    """
    model_names = list(metrics.keys())
    n_models    = len(model_names)
    if n_models == 0:
        return

    metric_keys    = ["accuracy", "f1_macro", "f1_weighted"]
    metric_labels  = ["Accuracy", "F1 (macro)", "F1 (weighted)"]
    colours        = ["#4C72B0", "#55A868", "#C44E52"]

    x      = np.arange(n_models)
    width  = 0.22
    n_classes = None  # derive from per_class size

    fig, ax = plt.subplots(figsize=(11, 6))

    for offset, (mk, label, colour) in enumerate(
        zip(metric_keys, metric_labels, colours)
    ):
        vals = [metrics[m][mk] * 100 for m in model_names]
        bars = ax.bar(
            x + (offset - 1) * width,
            vals,
            width=width,
            label=label,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}%",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )
        # infer n_classes from first model's per_class
        if n_classes is None and "per_class" in metrics[model_names[0]]:
            n_classes = len(metrics[model_names[0]]["per_class"])

    # Random-chance baseline
    if n_classes:
        baseline = 100.0 / n_classes
        ax.axhline(
            baseline, color="dimgrey", linestyle="--", linewidth=1.2,
            label=f"Random baseline ({baseline:.1f}%)",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        "Model Comparison — Accuracy, F1 Macro, F1 Weighted\n"
        "⚠️ Synthetic demo data — metrics are NOT real-world results",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    _save(fig, "model_comparison.png")


def _plot_cv_boxplot(cv_scores: np.ndarray, model_name: str) -> None:
    """
    Box plot of k-fold CV accuracy scores to visualise variance.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        cv_scores * 100,
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "#55A868", "alpha": 0.7},
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": "o", "markerfacecolor": "crimson", "markersize": 5},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )
    # Overlay individual fold points
    jitter = np.random.default_rng(0).uniform(-0.05, 0.05, size=len(cv_scores))
    ax.scatter(
        np.ones(len(cv_scores)) + jitter,
        cv_scores * 100,
        color="#4C72B0", zorder=5, s=40, label="Fold score",
    )
    ax.set_xticks([1])
    ax.set_xticklabels([model_name])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        f"{len(cv_scores)}-Fold CV — {model_name}\n"
        f"Mean: {cv_scores.mean() * 100:.2f}%  |  Std: {cv_scores.std() * 100:.2f}%\n"
        "⚠️ Synthetic demo data",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    _save(fig, "cv_rf_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════════
# Cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def _run_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    model_name: str = "Random Forest",
) -> np.ndarray:
    """
    Run stratified k-fold cross-validation and return per-fold scores.

    Stratified CV preserves the class distribution in every fold, which
    is important for multi-class problems.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : np.ndarray — feature matrix (full training set)
    y : np.ndarray — integer labels (full training set)
    n_folds : int
    model_name : str — used in console output only.

    Returns
    -------
    np.ndarray, shape (n_folds,) — accuracy per fold.
    """
    _section(f"5-Fold Stratified CV — {model_name}")
    print(f"  Running {n_folds}-fold CV … ", end="", flush=True)
    t0 = time.perf_counter()

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.1f}s)")
    print(f"  Fold scores : {np.round(scores * 100, 2).tolist()}")
    print(f"  Mean        : {scores.mean() * 100:.2f}%")
    print(f"  Std         : {scores.std() * 100:.2f}%")
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Text + JSON report writers
# ══════════════════════════════════════════════════════════════════════════════

def _write_text_report(
    report_lines: list[str],
    cv_scores: np.ndarray | None,
    metrics: dict,
) -> None:
    """Write a human-readable evaluation report to ``reports/evaluation_report.txt``."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    header = [
        "WHY AM I PROCRASTINATING? — MODEL EVALUATION REPORT",
        "=" * 60,
        "⚠️  All metrics computed on SYNTHETIC DEMO data.",
        "   Do NOT interpret these as real-world accuracy figures.",
        "   Replace demo_dataset.csv with real data before publishing.",
        "=" * 60,
    ]

    # Model comparison table
    table = [
        "\nMODEL COMPARISON TABLE",
        "-" * 60,
        f"{'Model':<22} {'Accuracy':>9} {'F1 Macro':>9} {'F1 Wt.':>9}",
        "-" * 60,
    ]
    for mname, m in metrics.items():
        table.append(
            f"{mname:<22} {m['accuracy'] * 100:>8.2f}%"
            f" {m['f1_macro'] * 100:>8.2f}%"
            f" {m['f1_weighted'] * 100:>8.2f}%"
        )
    table.append("-" * 60)

    cv_lines: list[str] = []
    if cv_scores is not None:
        cv_lines = [
            "\nRANDOM FOREST — 5-FOLD CROSS-VALIDATION",
            "-" * 60,
            f"Fold scores : {np.round(cv_scores * 100, 2).tolist()}",
            f"Mean        : {cv_scores.mean() * 100:.2f}%",
            f"Std         : {cv_scores.std() * 100:.2f}%",
        ]

    all_lines = header + table + cv_lines + report_lines
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(all_lines))
    print(f"\n  [evaluate] Text report → {REPORT_PATH}")


def _write_json_metrics(
    metrics: dict,
    cv_scores: np.ndarray | None,
) -> None:
    """Write machine-readable metrics to ``reports/metrics.json``."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    payload = {
        "disclaimer": (
            "All metrics computed on SYNTHETIC DEMO data. "
            "Replace dataset before publishing."
        ),
        "models": metrics,
    }
    if cv_scores is not None:
        payload["cross_validation_rf"] = {
            "n_folds":     len(cv_scores),
            "fold_scores": [round(float(s), 6) for s in cv_scores],
            "mean":        round(float(cv_scores.mean()), 6),
            "std":         round(float(cv_scores.std()), 6),
        }

    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  [evaluate] JSON metrics → {METRICS_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(data: dict) -> dict:
    """
    Evaluate all three models and write report / visualisation artefacts.

    No metric is hard-coded.  Every number is derived from
    ``model.predict(X_test)`` applied to the held-out test set.

    Parameters
    ----------
    data : dict
        Output of ``src.train.train_all()``.  Expected keys:
        ``lr``, ``rf``, ``xgb`` (model objects, xgb may be None),
        ``X_test``, ``X_test_sc``, ``y_test``,
        ``X_train``, ``y_train``,
        ``label_encoder``, ``class_names``.

    Returns
    -------
    dict
        Machine-readable metrics for every model, plus CV results:

        .. code-block:: python

            {
                "Logistic Regression": {
                    "accuracy": ..., "f1_macro": ..., "f1_weighted": ...,
                    "precision_macro": ..., "recall_macro": ...,
                    "precision_weighted": ..., "recall_weighted": ...,
                    "per_class": { <class>: {"precision", "recall", "f1", "support"} }
                },
                "Random Forest": { ... },
                "XGBoost": { ... },          # omitted if xgb is None
                "cv_rf": np.ndarray          # 5-fold CV scores for RF
            }

    Side-effects
    ------------
    Writes to ``reports/figures/``, ``reports/evaluation_report.txt``,
    and ``reports/metrics.json``.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    le          = data["label_encoder"]
    class_names = list(le.classes_)
    X_test      = data["X_test"]       # raw     — RF / XGBoost
    X_test_sc   = data["X_test_sc"]    # scaled  — Logistic Regression
    y_test      = data["y_test"]

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print(f"  Test samples : {len(y_test):,}")
    print(f"  Classes      : {len(class_names)}")
    print("  ⚠️  Metrics from SYNTHETIC demo data")
    print("=" * 60)

    report_lines: list[str] = []
    metrics: dict = {}

    # ── Logistic Regression (scaled features) ─────────────────────────────────
    metrics["Logistic Regression"] = _evaluate_single(
        "Logistic Regression",
        data["lr"],
        X_test_sc,          # ← scaled
        y_test,
        class_names,
        report_lines,
    )

    # ── Random Forest (raw features) ──────────────────────────────────────────
    metrics["Random Forest"] = _evaluate_single(
        "Random Forest",
        data["rf"],
        X_test,             # ← raw
        y_test,
        class_names,
        report_lines,
    )

    # ── XGBoost (raw features, optional) ──────────────────────────────────────
    cv_scores = None
    if data.get("xgb") is not None:
        metrics["XGBoost"] = _evaluate_single(
            "XGBoost",
            data["xgb"],
            X_test,         # ← raw
            y_test,
            class_names,
            report_lines,
        )
    else:
        print("\n  [evaluate] XGBoost skipped (model not available).")

    # ── 5-fold stratified CV on Random Forest ─────────────────────────────────
    # RF is the primary model, so we measure its generalisation robustly.
    cv_scores = _run_cross_validation(
        data["rf"],
        data["X_train"],
        data["y_train"],
        n_folds=5,
        model_name="Random Forest",
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    _section("Visualisations")
    _plot_model_comparison(metrics)
    _plot_cv_boxplot(cv_scores, "Random Forest")

    # ── Reports ───────────────────────────────────────────────────────────────
    _write_text_report(report_lines, cv_scores, metrics)
    # JSON: convert numpy arrays inside per_class before serialising
    _write_json_metrics(metrics, cv_scores)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅  EVALUATION COMPLETE")
    for mname, m in metrics.items():
        print(
            f"  {mname:<22}  acc={m['accuracy'] * 100:.1f}%"
            f"  f1_mac={m['f1_macro'] * 100:.1f}%"
        )
    if cv_scores is not None:
        print(
            f"  Random Forest CV     "
            f"  mean={cv_scores.mean() * 100:.1f}%"
            f"  std={cv_scores.std() * 100:.1f}%"
        )
    print("=" * 60 + "\n")

    return {**metrics, "cv_rf": cv_scores}


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained models for 'Why Am I Procrastinating?'"
    )
    parser.add_argument(
        "--data", type=str,
        default=os.path.join("data", "raw", "demo_dataset.csv"),
        help="Path to the input CSV dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible preprocessing split",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[evaluate] Dataset not found at '{args.data}' — generating …")
        from src.data_generator import generate_demo_dataset
        generate_demo_dataset(output_path=args.data)

    from src.preprocessing import build_pipeline
    from src.train import train_all

    data = build_pipeline(path=args.data, random_seed=args.seed)
    data = train_all(data)
    evaluate_all(data)

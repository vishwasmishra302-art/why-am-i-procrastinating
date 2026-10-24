"""
src/explain.py
===============
Generates global explainability artefacts for the trained Random Forest model:

1. **Feature importance bar chart** — built-in RF mean decrease in impurity.
2. **SHAP global importance bar chart** — mean |SHAP| across all test samples
   and all classes, giving a model-agnostic view of feature impact.
3. **SHAP multi-class beeswarm** — ``shap.summary_plot`` across all classes.

All charts are written to ``data/processed/figures/``.

Note: Per-prediction (local) SHAP explanations are handled separately in
``src.predict.shap_waterfall_figure`` so the Streamlit app can compute them
at runtime without re-running this module.

Public API
----------
explain_all(data) -> None
"""

from __future__ import annotations  # PEP 563 — defers annotation evaluation (Python 3.9 compat)

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap


# ── Paths ──────────────────────────────────────────────────────────────────────
_FIGURES_DIR = os.path.join("data", "processed", "figures")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _plot_feature_importance(
    rf_model,
    feature_names: list[str],
    save_path: str,
) -> None:
    """
    Bar chart of built-in Random Forest feature importances (mean decrease
    in impurity, also called the Gini importance).

    Parameters
    ----------
    rf_model :
        Fitted RandomForestClassifier.
    feature_names : list[str]
    save_path : str
    """
    importances = rf_model.feature_importances_
    order = np.argsort(importances)  # ascending so horizontal bar reads top-to-bottom

    palette = sns.color_palette("viridis", len(feature_names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in order],
        importances[order],
        color=[palette[j] for j in range(len(feature_names))],
        edgecolor="black", linewidth=0.4,
    )
    ax.set_xlabel("Mean Decrease in Impurity (feature importance)", fontsize=11)
    ax.set_title(
        "Random Forest — Built-in Feature Importance\n⚠️ Synthetic demo data",
        fontsize=13, fontweight="bold",
    )
    # Annotate each bar with its value
    for i, idx in enumerate(order):
        ax.text(
            importances[idx] + 0.001, i,
            f"{importances[idx]:.3f}",
            va="center", fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Feature importance chart → {save_path}")


def _plot_shap_global_bar(
    shap_values: list[np.ndarray],
    feature_names: list[str],
    save_path: str,
) -> None:
    """
    Bar chart of mean absolute SHAP values averaged across all classes and
    all test samples — a global view of which features matter most.

    Parameters
    ----------
    shap_values : list of np.ndarray
        One array per class, each shape (n_test, n_features).
    feature_names : list[str]
    save_path : str
    """
    # Average |SHAP| over all classes and all samples → shape (n_features,)
    global_importance = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
    )
    order = np.argsort(global_importance)

    palette = sns.color_palette("rocket_r", len(feature_names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in order],
        global_importance[order],
        color=[palette[j] for j in range(len(feature_names))],
        edgecolor="black", linewidth=0.4,
    )
    ax.set_xlabel("Mean |SHAP value| (global feature impact across all classes)", fontsize=10)
    ax.set_title(
        "SHAP — Global Feature Importance (Random Forest)\n⚠️ Synthetic demo data",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] SHAP global bar chart → {save_path}")


def _plot_shap_beeswarm(
    shap_values: list[np.ndarray],
    X_test: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    save_path: str,
) -> None:
    """
    Multi-class SHAP beeswarm summary using the standard ``shap.summary_plot``.

    Parameters
    ----------
    shap_values : list of np.ndarray
    X_test : np.ndarray
    feature_names : list[str]
    class_names : list[str]
    save_path : str
    """
    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        class_names=class_names,
        show=False,
        max_display=12,
    )
    plt.title(
        "SHAP Multi-Class Summary (beeswarm)\n⚠️ Synthetic demo data",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] SHAP beeswarm chart → {save_path}")


# ── Public API ─────────────────────────────────────────────────────────────────

def explain_all(data: dict) -> None:
    """
    Generate and save all global explainability figures.

    Parameters
    ----------
    data : dict
        Output of ``src.train.train_all()``.
    """
    os.makedirs(_FIGURES_DIR, exist_ok=True)

    rf            = data["rf"]
    X_test        = data["X_test"]
    feature_names = data["feature_names"]
    class_names   = list(data["label_encoder"].classes_)

    # 1. Built-in feature importance
    _plot_feature_importance(
        rf, feature_names,
        os.path.join(_FIGURES_DIR, "feature_importance_rf.png"),
    )

    # 2. SHAP values (TreeExplainer is exact for tree models)
    print("[explain] Computing SHAP values … (may take a few seconds)")
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)   # list of (n_test, n_features)

    # 3. Global SHAP bar chart
    _plot_shap_global_bar(
        shap_values, feature_names,
        os.path.join(_FIGURES_DIR, "shap_global_importance.png"),
    )

    # 4. Multi-class beeswarm
    _plot_shap_beeswarm(
        shap_values, X_test, feature_names, class_names,
        os.path.join(_FIGURES_DIR, "shap_beeswarm.png"),
    )

    print("[explain] ✅ All explainability figures saved.")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.preprocessing import generate_synthetic_data, build_pipeline
    from src.train import train_all

    generate_synthetic_data()
    data = build_pipeline()
    data = train_all(data)
    explain_all(data)

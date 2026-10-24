"""
src/train.py
=============
ML training pipeline for "Why Am I Procrastinating?"

Trains three classifiers on the data produced by ``src.preprocessing.build_pipeline``
and saves every fitted model to ``models/`` with joblib.

Models
------
1. **Logistic Regression** (LR) — linear baseline, uses *scaled* features.
2. **Random Forest** (RF)       — primary model, uses *raw* features,
                                  used for SHAP explanations in the app.
3. **XGBoost** (XGB)            — gradient-boosted comparison, raw features.

Key design decisions
---------------------
* ``class_weight="balanced"`` / ``scale_pos_weight`` — handles mild class imbalance.
* All ``random_state`` values are pinned to ``RANDOM_STATE`` (42) for
  full reproducibility.
* XGBoost receives integer-encoded labels directly; no extra label encoding step.
* ``multi:softmax`` / ``multi:softprob`` are handled automatically when
  ``num_class`` is set.
* ``use_label_encoder`` kwarg is intentionally omitted (removed in XGBoost ≥ 2.0).

Public API
----------
train_all(data)  -> dict
    Accepts the dict returned by ``src.preprocessing.build_pipeline`` and
    returns it augmented with ``"lr"``, ``"rf"``, ``"xgb"`` keys.

CLI
---
    python -m src.train [--data PATH] [--seed N]
"""

from __future__ import annotations

import argparse
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# XGBoost may need libomp on macOS; give a clear error if missing.
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except Exception as _xgb_err:  # noqa: BLE001
    _XGB_AVAILABLE = False
    _XGB_ERR_MSG = str(_xgb_err)


# ══════════════════════════════════════════════════════════════════════════════
# Paths & hyper-parameters
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR  = "models"
LR_PATH    = os.path.join(MODEL_DIR, "logistic_regression.pkl")
RF_PATH    = os.path.join(MODEL_DIR, "random_forest.pkl")
XGB_PATH   = os.path.join(MODEL_DIR, "xgboost.pkl")

#: Single random-state used by every model for reproducibility.
RANDOM_STATE: int = 42


# ══════════════════════════════════════════════════════════════════════════════
# Individual trainers
# ══════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(
    X_train_sc: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """
    Fit a multi-class Logistic Regression classifier.

    Uses **scaled** features (``X_train_sc``) because LR is sensitive to
    feature magnitude.  ``class_weight="balanced"`` re-weights samples so
    that rare classes contribute proportionally during training.

    Hyper-parameters
    ----------------
    solver      : lbfgs — supports multinomial loss, good for medium datasets.
    max_iter    : 1000  — enough for convergence on 12 features / 8 classes.
    C           : 1.0   — default L2 regularisation.

    Parameters
    ----------
    X_train_sc : np.ndarray, shape (n_train, 12)
        StandardScaler-transformed training features.
    y_train : np.ndarray, shape (n_train,)
        Integer-encoded target labels.

    Returns
    -------
    LogisticRegression
        Fitted model, also saved to ``models/logistic_regression.pkl``.
    """
    print("[train] Logistic Regression … ", end="", flush=True)
    t0 = time.perf_counter()

    model = LogisticRegression(
        solver="lbfgs",
        multi_class="auto",   # auto → multinomial for > 2 classes
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_sc, y_train)

    elapsed = time.perf_counter() - t0
    joblib.dump(model, LR_PATH)
    print(f"done ({elapsed:.1f}s)  →  {LR_PATH}")
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Fit a Random Forest multi-class classifier.

    Operates on **raw (unscaled)** features — tree ensembles are invariant
    to monotone feature transformations.  This is the **primary model** used
    by the Streamlit app and for SHAP explanations.

    Hyper-parameters
    ----------------
    n_estimators      : 300  — enough trees to stabilise variance on 1 k rows.
    max_depth         : 12   — limits over-fitting while allowing split depth.
    min_samples_split : 5    — avoids overly granular splits.
    min_samples_leaf  : 2    — each leaf must cover at least 2 samples.
    max_features      : sqrt — standard RF heuristic for classification.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 12)
        Raw (unscaled) training features.
    y_train : np.ndarray, shape (n_train,)
        Integer-encoded target labels.

    Returns
    -------
    RandomForestClassifier
        Fitted model, also saved to ``models/random_forest.pkl``.
    """
    print("[train] Random Forest … ", end="", flush=True)
    t0 = time.perf_counter()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    elapsed = time.perf_counter() - t0
    joblib.dump(model, RF_PATH)
    print(f"done ({elapsed:.1f}s)  →  {RF_PATH}")
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
) -> "XGBClassifier":
    """
    Fit an XGBoost multi-class classifier.

    Operates on **raw (unscaled)** features.  Requires ``libomp`` on macOS
    (install with ``brew install libomp``).

    Hyper-parameters
    ----------------
    n_estimators    : 300  — number of boosting rounds.
    learning_rate   : 0.1  — shrinkage per round.
    max_depth       : 6    — tree depth (typical default).
    subsample       : 0.8  — row sampling fraction per tree.
    colsample_bytree: 0.8  — column sampling fraction per tree.
    objective       : multi:softprob — outputs per-class probabilities.
    eval_metric     : mlogloss — cross-entropy for multi-class problems.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 12)
        Raw (unscaled) training features.
    y_train : np.ndarray, shape (n_train,)
        Integer-encoded target labels (0-indexed, contiguous).
    n_classes : int
        Number of distinct classes — required for multi:softprob.

    Returns
    -------
    XGBClassifier
        Fitted model, also saved to ``models/xgboost.pkl``.

    Raises
    ------
    RuntimeError
        If XGBoost could not be imported (missing libomp on macOS).
    """
    if not _XGB_AVAILABLE:
        raise RuntimeError(
            "[train] XGBoost could not be loaded.\n"
            f"  Original error: {_XGB_ERR_MSG}\n"
            "  Fix on macOS: brew install libomp\n"
            "  Then re-run: python -m src.train"
        )

    print("[train] XGBoost … ", end="", flush=True)
    t0 = time.perf_counter()

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_classes,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,       # suppress XGBoost progress output
    )
    model.fit(X_train, y_train)

    elapsed = time.perf_counter() - t0
    joblib.dump(model, XGB_PATH)
    print(f"done ({elapsed:.1f}s)  →  {XGB_PATH}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def train_all(data: dict) -> dict:
    """
    Train all three classifiers and augment the pipeline data dict.

    Reads inputs directly from the dict produced by
    ``src.preprocessing.build_pipeline``:

    +--------------+---------------------------------------------------+
    | Key used     | Purpose                                           |
    +==============+===================================================+
    | X_train      | Raw features  → RF, XGBoost                      |
    | X_train_sc   | Scaled feats  → Logistic Regression               |
    | y_train      | Integer labels → all models                       |
    | label_encoder| To count n_classes for XGBoost                    |
    +--------------+---------------------------------------------------+

    Parameters
    ----------
    data : dict
        Output of ``src.preprocessing.build_pipeline()``.

    Returns
    -------
    dict
        Same dict as input, with three new keys:
        ``"lr"``  — fitted LogisticRegression
        ``"rf"``  — fitted RandomForestClassifier
        ``"xgb"`` — fitted XGBClassifier (or ``None`` if XGBoost unavailable)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    n_classes = len(data["label_encoder"].classes_)

    print("\n" + "=" * 60)
    print("  MODEL TRAINING")
    print(f"  Training samples : {len(data['y_train']):,}")
    print(f"  Classes          : {n_classes}")
    print("=" * 60)

    lr  = train_logistic_regression(data["X_train_sc"], data["y_train"])
    rf  = train_random_forest(data["X_train"],          data["y_train"])

    # XGBoost is optional — if libomp missing, skip and warn.
    xgb = None
    if _XGB_AVAILABLE:
        xgb = train_xgboost(data["X_train"], data["y_train"], n_classes)
    else:
        print(
            f"[train] ⚠ XGBoost skipped — {_XGB_ERR_MSG}\n"
            "  Fix on macOS: brew install libomp"
        )

    print("\n[train] ✅ Training complete.  Models saved to models/\n")
    return {**data, "lr": lr, "rf": rf, "xgb": xgb}


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all ML models for 'Why Am I Procrastinating?'"
    )
    parser.add_argument(
        "--data", type=str,
        default=os.path.join("data", "raw", "demo_dataset.csv"),
        help="Path to the input CSV dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for preprocessing split (default: 42)",
    )
    args = parser.parse_args()

    # Auto-generate demo data if missing
    if not os.path.exists(args.data):
        print(f"[train] Dataset not found at '{args.data}' — generating …")
        from src.data_generator import generate_demo_dataset
        generate_demo_dataset(output_path=args.data)

    from src.preprocessing import build_pipeline
    data = build_pipeline(path=args.data, random_seed=args.seed)
    train_all(data)

"""
src/predict.py
===============
Single-sample inference engine used by the Streamlit application.

Responsibilities
----------------
- Load all saved artefacts from ``models/`` (models + preprocessing objects).
- Accept a dict of user-supplied feature values.
- Apply the SAME preprocessing transformations used during training via
  ``src.preprocessing.transform_input_for_inference`` (imputation, clipping,
  scaling) — preventing any training/inference mismatch.
- Return the predicted class, per-class probabilities, and SHAP values
  for the predicted class.
- Generate a SHAP waterfall figure ready for ``st.pyplot()``.

Public API
----------
load_artifacts()  -> tuple
predict(input_dict, ...)  -> dict
shap_waterfall_figure(result)  -> matplotlib.figure.Figure
"""

from __future__ import annotations  # PEP 563 — defers annotation evaluation (Python 3.9 compat)

import os
import sys
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Paths ──────────────────────────────────────────────────────────────────────
_MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
_RF_PATH      = os.path.join(_MODEL_DIR, "random_forest.pkl")
_LR_PATH      = os.path.join(_MODEL_DIR, "logistic_regression.pkl")
_XGB_PATH     = os.path.join(_MODEL_DIR, "xgboost.pkl")
_ENCODER_PATH = os.path.join(_MODEL_DIR, "label_encoder.pkl")
_SCALER_PATH  = os.path.join(_MODEL_DIR, "scaler.pkl")
_IMPUTER_PATH = os.path.join(_MODEL_DIR, "imputer.pkl")

# ── Feature metadata ───────────────────────────────────────────────────────────

#: Ordered feature column names — must match src.preprocessing.FEATURE_COLS.
FEATURE_COLS: list[str] = [
    "sleep_hours",
    "task_difficulty",
    "task_interest",
    "stress_level",
    "phone_distraction",
    "social_media_usage",
    "energy_level",
    "task_clarity",
    "confidence",
    "deadline_pressure",
    "perfectionism",
    "previous_procrastination",
]

#: Human-readable labels used in charts and tooltips.
FEATURE_LABELS: dict[str, str] = {
    "sleep_hours":               "Sleep Hours",
    "task_difficulty":           "Task Difficulty",
    "task_interest":             "Interest in Task",
    "stress_level":              "Stress Level",
    "phone_distraction":         "Phone Distraction",
    "social_media_usage":        "Social Media Usage (hrs)",
    "energy_level":              "Energy Level",
    "task_clarity":              "Task Clarity",
    "confidence":                "Confidence",
    "deadline_pressure":         "Deadline Pressure",
    "perfectionism":             "Perfectionism",
    "previous_procrastination":  "History of Procrastination",
}


# ── Artifact loading ───────────────────────────────────────────────────────────

def load_artifacts() -> tuple:
    """
    Load all model and preprocessing artefacts required for inference.

    Includes the imputer and scaler so that live Streamlit inputs are
    transformed identically to training data.

    Returns
    -------
    tuple : (rf, lr, xgb, label_encoder, scaler, imputer, shap_explainer)
        ``rf``              — RandomForestClassifier (primary model)
        ``lr``              — LogisticRegression (or None if missing)
        ``xgb``             — XGBClassifier (or None if missing)
        ``label_encoder``   — fitted LabelEncoder
        ``scaler``          — fitted StandardScaler (train-set only)
        ``imputer``         — fitted SimpleImputer  (train-set only)
        ``shap_explainer``  — shap.TreeExplainer built on the RF

    Raises
    ------
    FileNotFoundError
        If the RF model, encoder, scaler, or imputer are missing.
    """
    required = [_RF_PATH, _ENCODER_PATH, _SCALER_PATH, _IMPUTER_PATH]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Required artefacts not found:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nRun the full pipeline first:\n"
            "  python -m src.preprocessing\n"
            "  python -m src.train"
        )

    rf      = joblib.load(_RF_PATH)
    le      = joblib.load(_ENCODER_PATH)
    scaler  = joblib.load(_SCALER_PATH)
    imputer = joblib.load(_IMPUTER_PATH)

    lr  = joblib.load(_LR_PATH)  if os.path.exists(_LR_PATH)  else None
    xgb = joblib.load(_XGB_PATH) if os.path.exists(_XGB_PATH) else None

    explainer = shap.TreeExplainer(rf)

    return rf, lr, xgb, le, scaler, imputer, explainer


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict(
    input_dict: dict,
    rf,
    lr,
    xgb,
    label_encoder,
    scaler,
    imputer,
    explainer,
) -> dict:
    """
    Run inference for a single user input using the Random Forest model.

    The input dict is passed through the preprocessing pipeline
    (``src.preprocessing.transform_input_for_inference``) before prediction,
    so the model always receives data in the exact same format it was
    trained on — preventing any training/inference distribution mismatch.

    Parameters
    ----------
    input_dict : dict
        Mapping of ``{feature_name: value}`` for all 12 features.
    rf, lr, xgb :
        Fitted classifier objects (lr / xgb may be None).
    label_encoder :
        Fitted LabelEncoder.
    scaler :
        Fitted StandardScaler (train-set only; used for LR).
    imputer :
        Fitted SimpleImputer (train-set only).
    explainer :
        ``shap.TreeExplainer`` fitted on the RF.

    Returns
    -------
    dict with keys:
        ``"predicted_class"``  — str  human-readable label
        ``"predicted_idx"``    — int  label-encoder index
        ``"probabilities"``    — dict {class_name: probability}
        ``"shap_values"``      — np.ndarray shape (n_features,) for predicted class
        ``"feature_names"``    — list[str] ordered column names
        ``"feature_labels"``   — list[str] ordered human-readable names
        ``"feature_values"``   — list[float] original user-supplied values
    """
    # ── Apply identical preprocessing as training data ─────────────────────────
    from src.preprocessing import transform_input_for_inference
    x_raw, _x_scaled = transform_input_for_inference(input_dict, scaler, imputer)
    # x_raw shape: (1, 12) — unscaled → Random Forest / XGBoost
    # _x_scaled shape: (1, 12) — scaled   → Logistic Regression (not used here)

    # ── Primary prediction (Random Forest) ────────────────────────────────────
    pred_idx   = int(rf.predict(x_raw)[0])
    pred_proba = rf.predict_proba(x_raw)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    proba_dict = {
        label_encoder.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(pred_proba)
    }

    # ── SHAP values for the predicted class ───────────────────────────────────
    #
    # shap.TreeExplainer.shap_values() output format depends on SHAP version:
    #
    #   SHAP ≤ 0.41  — list of length n_classes, each element shape (n_samples, n_features)
    #                  → sv_for_class = shap_values[pred_idx][0]
    #
    #   SHAP 0.42+   — single ndarray, shape (n_samples, n_features, n_classes)
    #                  → sv_for_class = shap_values[0, :, pred_idx]
    #
    # We detect the format at runtime so the code survives any installed version.
    # If SHAP fails for any reason (import error, version mismatch, etc.) the
    # prediction and recommendation are still returned — only the explanation
    # chart is suppressed.
    sv_for_class: np.ndarray | None = None
    shap_error: str | None = None

    try:
        raw_sv = explainer.shap_values(x_raw)   # ← x_raw, NOT x

        if isinstance(raw_sv, list):
            # Old format: list[ndarray(n_samples, n_features)]
            sv_for_class = np.array(raw_sv[pred_idx])[0]   # shape (n_features,)
        else:
            # New format (SHAP ≥ 0.42): ndarray(n_samples, n_features, n_classes)
            arr = np.array(raw_sv)
            if arr.ndim == 3:
                sv_for_class = arr[0, :, pred_idx]          # shape (n_features,)
            elif arr.ndim == 2:
                # Edge case: 2-D means binary or single-output; take as-is
                sv_for_class = arr[0]
            else:
                shap_error = f"Unexpected SHAP output shape {arr.shape}; explanation unavailable."

    except Exception as exc:  # noqa: BLE001
        shap_error = (
            f"SHAP explanation unavailable ({type(exc).__name__}: {exc}). "
            "Prediction and recommendation are unaffected."
        )

    return {
        "predicted_class": pred_label,
        "predicted_idx":   pred_idx,
        "probabilities":   proba_dict,
        "shap_values":     sv_for_class,   # None when SHAP failed
        "shap_error":      shap_error,     # None when SHAP succeeded
        "feature_names":   FEATURE_COLS,
        "feature_labels":  [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "feature_values":  [input_dict[f] for f in FEATURE_COLS],
    }


# ── SHAP visualisation ────────────────────────────────────────────────────────

def shap_waterfall_figure(result: dict) -> "plt.Figure | None":
    """
    Create a horizontal bar chart showing each feature's SHAP contribution
    toward the predicted class.

    Red bars push the prediction *toward* the predicted class;
    blue bars push it *away*.

    Parameters
    ----------
    result : dict
        Output of :func:`predict`.

    Returns
    -------
    matplotlib.figure.Figure | None
        Ready to pass to ``st.pyplot(fig)``.
        Returns ``None`` when SHAP values are unavailable (e.g. SHAP
        computation failed); the caller should display a fallback message
        using ``result["shap_error"]``.
    """
    sv = result.get("shap_values")
    if sv is None:
        return None   # caller handles the fallback message

    labels = result["feature_labels"]
    vals   = result["feature_values"]

    sv = np.asarray(sv, dtype=float)

    # ── Design tokens (match light theme) ─────────────────────────────────────
    _NAVY  = "#172033"   # axis text / labels
    _SLATE = "#475569"   # secondary text
    _POS   = "#2563EB"   # blue  — pushes TOWARD this prediction
    _NEG   = "#DC2626"   # red   — pushes AWAY from this prediction

    # Sort by absolute SHAP value (largest impact at the top)
    order         = np.argsort(np.abs(sv))
    sorted_sv     = sv[order]
    sorted_labels = [f"{labels[i]}  =  {vals[i]}" for i in order]
    bar_colors    = [_POS if v > 0 else _NEG for v in sorted_sv]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Light background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F7F9FC")

    ax.barh(sorted_labels, sorted_sv, color=bar_colors, edgecolor="none", linewidth=0)
    ax.axvline(0, color=_NAVY, linewidth=0.9, alpha=0.5)

    # Axis labels + ticks: dark, readable
    ax.set_xlabel(
        "SHAP contribution  (→ increases likelihood of this class   ← decreases it)",
        fontsize=9, color=_SLATE,
    )
    ax.tick_params(axis="both", colors=_NAVY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER := "#DDE3ED")

    ax.set_title(
        f"Feature contributions for: \"{result['predicted_class']}\"\n"
        "Blue = increases likelihood   |   Red = decreases likelihood",
        fontsize=11, fontweight="bold", color=_NAVY, pad=12,
    )

    # Ensure long y-axis labels are not clipped
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    fig.subplots_adjust(left=0.30)
    return fig


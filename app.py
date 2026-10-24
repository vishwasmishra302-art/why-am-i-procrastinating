"""
app.py
=======
Streamlit application — "Why Am I Procrastinating?"

Entry point for the interactive web UI.  Imports the inference and
recommendation modules from ``src/`` and renders a two-tab layout:

  Tab 1 — Diagnose My Procrastination
      12 behavioural input sliders → model prediction → probability chart
      → SHAP waterfall → personalised recommendations

  Tab 2 — Model Insights
      Confusion matrices, feature importance and SHAP figures generated
      during the training pipeline, plus the full evaluation report.

Run:
    streamlit run app.py

Prerequisites:
    python -m src.preprocessing   # generate dataset
    python -m src.train           # train models
    python -m src.evaluate        # evaluation figures
    python -m src.explain         # SHAP figures

Or run everything at once:
    python -m src.train           # which internally calls preprocessing
"""

from __future__ import annotations  # PEP 563 — defers annotation evaluation (Python 3.9 compat)

import os
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Ensure the project root is on the Python path ─────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.predict import load_artifacts, predict, shap_waterfall_figure, FEATURE_LABELS, FEATURE_COLS
from src.recommendations import get_recommendation


# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Why Am I Procrastinating?",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# Design tokens (keep in sync with .streamlit/config.toml)
# ══════════════════════════════════════════════════════════════════════════════
_BG         = "#F7F9FC"   # page background
_CARD       = "#FFFFFF"   # card / panel background
_NAVY       = "#172033"   # primary text
_SLATE      = "#475569"   # secondary / caption text
_BLUE       = "#2563EB"   # accent / primary
_BLUE_LIGHT = "#EFF6FF"   # accent tint for cards
_BORDER     = "#DDE3ED"   # subtle card border
_GREEN      = "#16A34A"   # positive SHAP
_RED        = "#DC2626"   # negative SHAP / warning
_YELLOW_BG  = "#FFFBEB"   # demo banner background
_YELLOW_BD  = "#F59E0B"   # demo banner border


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Reset Streamlit's default dark overrides ───────────────────── */
  html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    color: {_NAVY};
  }}

  /* ── Page background ────────────────────────────────────────────── */
  .stApp {{
    background-color: {_BG};
  }}

  /* ── Main title ─────────────────────────────────────────────────── */
  .main-title {{
    font-size: 2.1rem;
    font-weight: 700;
    color: {_NAVY};
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
  }}

  /* ── Subtitle ───────────────────────────────────────────────────── */
  .subtitle {{
    font-size: 1.05rem;
    color: {_SLATE};
    margin-bottom: 1.2rem;
    line-height: 1.6;
  }}

  /* ── Demo data banner ───────────────────────────────────────────── */
  .demo-banner {{
    background: {_YELLOW_BG};
    border: 1px solid {_YELLOW_BD};
    border-radius: 8px;
    padding: 0.65rem 1rem;
    color: #92400E;
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
  }}

  /* ── Section headings ───────────────────────────────────────────── */
  .sec-header {{
    font-size: 1.05rem;
    font-weight: 600;
    color: {_NAVY};
    border-bottom: 2px solid {_BLUE};
    padding-bottom: 0.2rem;
    margin: 1.4rem 0 0.7rem;
  }}

  /* ── Questionnaire group headers ────────────────────────────────── */
  .group-label {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {_BLUE};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 1rem 0 0.35rem;
  }}

  /* ── Prediction result card ─────────────────────────────────────── */
  .pred-card {{
    background: {_BLUE_LIGHT};
    border: 1.5px solid {_BLUE};
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 1.4rem;
  }}
  .pred-card .pred-label {{
    font-size: 0.95rem;
    font-weight: 600;
    color: {_BLUE};
    margin: 0 0 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .pred-card .pred-class {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {_NAVY};
    margin: 0 0 0.3rem;
  }}
  .pred-card .pred-conf {{
    font-size: 1rem;
    color: {_SLATE};
    margin: 0;
  }}
  .pred-card .pred-conf strong {{
    color: {_NAVY};
  }}

  /* ── Tip cards (recommendation bullets) ────────────────────────── */
  .tip-card {{
    background: {_CARD};
    border: 1px solid {_BORDER};
    border-left: 4px solid {_BLUE};
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.97rem;
    color: {_NAVY};
    line-height: 1.55;
  }}

  /* ── Sidebar ────────────────────────────────────────────────────── */
  [data-testid="stSidebar"] {{
    background-color: {_CARD};
    border-right: 1px solid {_BORDER};
  }}
  [data-testid="stSidebar"] * {{
    color: {_NAVY} !important;
  }}
  [data-testid="stSidebar"] .sidebar-title {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {_NAVY};
  }}
  [data-testid="stSidebar"] caption,
  [data-testid="stSidebar"] .stCaption {{
    color: {_SLATE} !important;
  }}

  /* ── Slider labels ──────────────────────────────────────────────── */
  .stSlider label, .stSlider [data-testid="stWidgetLabel"] {{
    font-size: 0.9rem;
    color: {_NAVY};
    font-weight: 500;
  }}

  /* ── Tab labels ─────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab"] {{
    font-size: 0.95rem;
    font-weight: 600;
    color: {_SLATE};
  }}
  .stTabs [aria-selected="true"] {{
    color: {_BLUE} !important;
  }}

  /* ── st.metric ──────────────────────────────────────────────────── */
  [data-testid="metric-container"] {{
    background: {_CARD};
    border: 1px solid {_BORDER};
    border-radius: 10px;
    padding: 0.8rem 1rem;
  }}
  [data-testid="metric-container"] label {{
    font-size: 0.82rem;
    color: {_SLATE};
    font-weight: 500;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.4rem;
    color: {_NAVY};
    font-weight: 700;
  }}
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.82rem;
    color: {_SLATE};
  }}

  /* ── st.expander ────────────────────────────────────────────────── */
  .stExpander {{
    border: 1px solid {_BORDER};
    border-radius: 10px;
    background: {_CARD};
  }}
  .stExpander summary {{
    font-weight: 600;
    color: {_NAVY};
    font-size: 0.97rem;
  }}

  /* ── Streamlit code block ───────────────────────────────────────── */
  .stCode {{
    font-size: 0.82rem;
  }}

  /* ── st.caption override ────────────────────────────────────────── */
  .stCaption, .stMarkdown small {{
    color: {_SLATE} !important;
    font-size: 0.88rem;
  }}

  /* ── Ensure Plotly text stays dark on white ─────────────────────── */
  .js-plotly-plot .plotly text {{
    fill: {_NAVY} !important;
  }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Load model artefacts (cached so they are only read once)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading trained models …")
def _load():
    return load_artifacts()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="sidebar-title">🧠 Why Am I Procrastinating?</p>', unsafe_allow_html=True)
    st.markdown(
        "An **explainable AI/ML system** that predicts your most likely "
        "procrastination factor from 12 behavioural inputs."
    )
    st.divider()

    st.markdown("**Models**")
    st.markdown(
        "| Model | Role |\n"
        "|---|---|\n"
        "| Logistic Regression | Baseline |\n"
        "| **Random Forest** ⭐ | Primary |\n"
        "| XGBoost | Comparison |"
    )
    st.divider()

    st.markdown("**8 Procrastination Classes**")
    for cls in [
        "Digital Distraction", "Low Motivation", "Task Difficulty",
        "Fear of Failure", "Perfectionism", "Poor Sleep / Low Energy",
        "Lack of Clarity", "Poor Time Management",
    ]:
        st.markdown(f"- {cls}")

    st.divider()
    st.warning(
        "**Demo Disclaimer** — Models are trained on **synthetic data**. "
        "This is an educational project, not a clinical tool."
    )
    st.caption("Python · scikit-learn · XGBoost · SHAP · Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">Why Am I Procrastinating?</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Answer 12 behavioural questions and receive an AI-powered '
    'explanation of your most likely procrastination factor — with personalised advice.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="demo-banner">⚠️ <strong>Synthetic Demo Data:</strong> '
    'Models are trained on programmatically generated data for demonstration purposes. '
    'Not a clinical or psychological assessment.</div>',
    unsafe_allow_html=True,
)

# ── Try loading models ─────────────────────────────────────────────────────────
try:
    rf, lr, xgb, le, scaler, imputer, explainer = _load()
    models_ready = True
except FileNotFoundError as err:
    models_ready = False
    _err_msg = str(err)

if not models_ready:
    st.error(
        "**Trained models not found.**\n\n"
        "Run the training pipeline first:\n"
        "```bash\n"
        "python -m src.preprocessing   # generate synthetic data\n"
        "python -m src.train           # train all models\n"
        "python -m src.evaluate        # evaluation charts\n"
        "python -m src.explain         # SHAP charts\n"
        "```\n\n" + _err_msg
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_diagnose, tab_insights = st.tabs(
    ["Diagnose My Procrastination", "Model Insights"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Diagnose
# ══════════════════════════════════════════════════════════════════════════════
with tab_diagnose:
    st.markdown("#### Describe your current state")
    st.caption("Adjust each slider to reflect how you feel right now or in recent days.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="group-label">Sleep &amp; Energy</p>', unsafe_allow_html=True)
        sleep_hours  = st.slider("Sleep hours (last night)", 2.0, 10.0, 6.5, 0.5)
        energy_level = st.slider("Energy level  (1 = drained — 10 = charged)", 1, 10, 5)

        st.markdown('<p class="group-label">Distractions</p>', unsafe_allow_html=True)
        phone_distraction  = st.slider("Phone distraction  (1 = none — 10 = constant)", 1, 10, 5)
        social_media_usage = st.slider("Social media usage today (hrs)", 0.0, 8.0, 2.0, 0.5)

    with col2:
        st.markdown('<p class="group-label">Task &amp; Clarity</p>', unsafe_allow_html=True)
        task_difficulty = st.slider("Task difficulty  (1 = easy — 10 = very hard)", 1, 10, 5)
        task_interest   = st.slider("Interest in task (1 = boring — 10 = love it)", 1, 10, 5)
        task_clarity    = st.slider("Task clarity     (1 = no idea — 10 = crystal clear)", 1, 10, 5)

        st.markdown('<p class="group-label">Mindset</p>', unsafe_allow_html=True)
        confidence = st.slider("Confidence  (1 = none — 10 = very high)", 1, 10, 5)

    with col3:
        st.markdown('<p class="group-label">Stress &amp; Psychology</p>', unsafe_allow_html=True)
        stress_level             = st.slider("Stress level          (1 = calm — 10 = overwhelmed)", 1, 10, 5)
        deadline_pressure        = st.slider("Deadline pressure     (1 = none — 10 = overdue)", 1, 10, 5)
        perfectionism            = st.slider("Perfectionism         (1 = none — 10 = extreme)", 1, 10, 5)
        previous_procrastination = st.slider("History of procrastination (1 = rarely — 10 = always)", 1, 10, 5)

    st.divider()

    btn_col, _ = st.columns([2, 5])
    with btn_col:
        run = st.button(
            "Analyse My Procrastination",
            type="primary",
            use_container_width=True,
        )

    if run:
        # ── Build input dict (unchanged ML logic) ──────────────────────────────
        user_input = {
            "sleep_hours":               sleep_hours,
            "task_difficulty":           task_difficulty,
            "task_interest":             task_interest,
            "stress_level":              stress_level,
            "phone_distraction":         phone_distraction,
            "social_media_usage":        social_media_usage,
            "energy_level":              energy_level,
            "task_clarity":              task_clarity,
            "confidence":                confidence,
            "deadline_pressure":         deadline_pressure,
            "perfectionism":             perfectionism,
            "previous_procrastination":  previous_procrastination,
        }

        with st.spinner("Computing prediction and explanation …"):
            result = predict(user_input, rf, lr, xgb, le, scaler, imputer, explainer)

        pred_class = result["predicted_class"]
        proba_dict = result["probabilities"]
        top_prob   = proba_dict[pred_class]
        rec        = get_recommendation(pred_class)

        # ── Prediction card (light blue, dark text) ────────────────────────────
        st.markdown(
            f'<div class="pred-card">'
            f'<p class="pred-label">Your primary procrastination factor</p>'
            f'<p class="pred-class">{rec["emoji"]}  {pred_class}</p>'
            f'<p class="pred-conf">Model confidence: <strong>{top_prob * 100:.1f}%</strong></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Probability chart + SHAP waterfall ────────────────────────────────
        res1, res2 = st.columns(2)

        with res1:
            st.markdown('<p class="sec-header">Prediction Probabilities</p>', unsafe_allow_html=True)
            sorted_items  = sorted(proba_dict.items(), key=lambda kv: kv[1], reverse=True)
            sorted_labels = [k for k, _ in sorted_items]
            sorted_probs  = [v * 100 for _, v in sorted_items]
            bar_colors    = [_BLUE if c == pred_class else "#93C5FD" for c in sorted_labels]

            fig_prob = go.Figure(go.Bar(
                x=sorted_probs,
                y=sorted_labels,
                orientation="h",
                marker_color=bar_colors,
                marker_line_color=_BORDER,
                marker_line_width=0.5,
                text=[f"{p:.1f}%" for p in sorted_probs],
                textposition="outside",
                textfont=dict(color=_NAVY, size=12),
            ))
            fig_prob.update_layout(
                xaxis=dict(
                    title=dict(text="Probability (%)", font=dict(color=_SLATE, size=12)),
                    range=[0, 120],
                    tickfont=dict(color=_NAVY, size=11),
                    showgrid=True,
                    gridcolor="#E2E8F0",
                    zeroline=False,
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(color=_NAVY, size=11),
                    showgrid=False,
                ),
                margin=dict(l=5, r=60, t=15, b=15),
                height=340,
                plot_bgcolor=_CARD,
                paper_bgcolor=_CARD,
                font=dict(color=_NAVY),
                bargap=0.35,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with res2:
            st.markdown('<p class="sec-header">Feature Contributions (SHAP)</p>', unsafe_allow_html=True)
            st.caption("Positive values (blue) increase likelihood of this class; negative values (red) reduce it.")
            fig_shap = shap_waterfall_figure(result)
            if fig_shap is not None:
                st.pyplot(fig_shap, use_container_width=True)
                plt.close(fig_shap)
            else:
                shap_err = result.get("shap_error", "SHAP explanation unavailable.")
                st.info(f"ℹ️ {shap_err}")

        # ── Recommendations ────────────────────────────────────────────────────
        st.markdown('<p class="sec-header">Personalised Recommendations</p>', unsafe_allow_html=True)

        with st.expander(f"Why {pred_class}? — Understanding the pattern", expanded=True):
            st.markdown(rec["why"])

        st.markdown("**Actionable Tips**")
        for tip in rec["tips"]:
            st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

        if rec.get("resources"):
            st.markdown(
                "**Further Reading / Techniques:** "
                + " · ".join(f"`{r}`" for r in rec["resources"])
            )

        # ── Top 3 drivers ──────────────────────────────────────────────────────
        st.markdown('<p class="sec-header">Top 3 Contributing Factors</p>', unsafe_allow_html=True)
        sv          = result.get("shap_values")
        feat_labels = result["feature_labels"]
        feat_vals   = result["feature_values"]

        if sv is not None:
            sv   = np.asarray(sv)
            top3 = np.argsort(np.abs(sv))[::-1][:3]
            cols3 = st.columns(3)
            for rank, idx in enumerate(top3):
                direction = "↑ Increases likelihood" if sv[idx] > 0 else "↓ Reduces likelihood"
                with cols3[rank]:
                    st.metric(
                        label=f"#{rank + 1} — {feat_labels[idx]}",
                        value=str(feat_vals[idx]),
                        delta=direction,
                        delta_color="off",
                    )
        else:
            st.info(
                "Feature contribution ranking is unavailable because the SHAP "
                "explanation could not be computed. The prediction above is unaffected."
            )

        st.divider()
        st.caption(
            "Disclaimer: Results are generated by a model trained on synthetic demo data. "
            "This is not a clinical, psychological, or scientific assessment."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.markdown("#### Model Training Insights")
    st.markdown(
        '<div class="demo-banner">⚠️ All figures and metrics below are from models '
        'trained on <strong>synthetic demo data</strong> — not real survey results.</div>',
        unsafe_allow_html=True,
    )

    # Figures can be in reports/figures/ (new evaluate.py) or data/processed/figures/ (old)
    _FIG_DIRS = [
        os.path.join(ROOT, "reports", "figures"),
        os.path.join(ROOT, "data", "processed", "figures"),
    ]
    _REP_PATHS = [
        os.path.join(ROOT, "reports", "evaluation_report.txt"),
        os.path.join(ROOT, "data", "processed", "evaluation_report.txt"),
    ]

    def _find_fig(fname: str) -> str | None:
        """Return the first path where fname exists across known figure dirs."""
        for d in _FIG_DIRS:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return p
        return None

    def _show(fname: str, caption: str = "") -> None:
        """Display a figure, or a friendly info box if not yet generated."""
        path = _find_fig(fname)
        if path:
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.info(f"`{fname}` not found. Run the pipeline first.")

    # ── Model comparison + feature importance ──────────────────────────────────
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown('<p class="sec-header">Model Accuracy Comparison</p>', unsafe_allow_html=True)
        _show("model_comparison.png", "Accuracy across all three models (synthetic data)")
    with ic2:
        st.markdown('<p class="sec-header">RF Feature Importance</p>', unsafe_allow_html=True)
        _show("feature_importance_rf.png", "Mean decrease in impurity (built-in RF importances)")

    st.divider()

    # ── SHAP global charts ─────────────────────────────────────────────────────
    st.markdown('<p class="sec-header">SHAP Global Explainability</p>', unsafe_allow_html=True)
    st.caption(
        "SHAP values show how much each feature shifts the model's output. "
        "Higher |SHAP| = greater influence. These are model attributions, not causal effects."
    )
    sc1, sc2 = st.columns(2)
    with sc1:
        _show("shap_global_importance.png", "Mean |SHAP| across all classes")
    with sc2:
        _show("shap_beeswarm.png", "Multi-class beeswarm summary")

    st.divider()

    # ── Confusion matrices ─────────────────────────────────────────────────────
    st.markdown('<p class="sec-header">Confusion Matrices</p>', unsafe_allow_html=True)
    cm1, cm2, cm3 = st.columns(3)
    # Try both naming conventions (old slug vs new)
    with cm1:
        _show("cm_logistic_regression.png", "Logistic Regression") or \
        _show("logistic_regression_confusion_matrix.png", "Logistic Regression")
    with cm2:
        _show("cm_random_forest.png", "Random Forest") or \
        _show("random_forest_confusion_matrix.png", "Random Forest")
    with cm3:
        _show("cm_xgboost.png", "XGBoost") or \
        _show("xgboost_confusion_matrix.png", "XGBoost")

    st.divider()

    # ── Full text evaluation report ────────────────────────────────────────────
    st.markdown('<p class="sec-header">Full Evaluation Report</p>', unsafe_allow_html=True)
    rep_path = next((p for p in _REP_PATHS if os.path.exists(p)), None)
    if rep_path:
        with open(rep_path, "r", encoding="utf-8") as fh:
            st.code(fh.read(), language="text")
    else:
        st.info("Report not found. Run `python -m src.evaluate` first.")

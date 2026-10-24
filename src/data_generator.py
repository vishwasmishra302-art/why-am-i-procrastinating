"""
src/data_generator.py
======================
Synthetic / Demo Dataset Generator for "Why Am I Procrastinating?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  IMPORTANT — READ BEFORE USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This script generates ENTIRELY SYNTHETIC data for development and
testing of the ML pipeline.

    • NOT real student data.
    • NOT collected from any human subjects.
    • NOT validated against actual procrastination behaviour.
    • MUST NOT be presented in papers, reports, or the application
      as real or collected data.

Real project results MUST use a properly collected and labelled dataset
(e.g., via a validated survey instrument such as the Procrastination
Assessment Scale for Students, PASS).

This file may be removed or replaced once real data is available.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Design principles
-----------------
1. **No trivial single-feature determinism**
   Each class has a primary distinguishing feature cluster, but all
   features are sampled from a *mixture* of the class profile and the
   global population distribution (controlled by SIGNAL_WEIGHT).
   This means no single slider value maps deterministically to one label.

2. **Realistic inter-feature correlations**
   Real behaviour is correlated: tired people have low energy; stressed
   people have lower confidence; phone-heavy users also use more social
   media.  We inject these post-hoc so the synthetic data respects
   plausible structure.

3. **Label noise**
   A small fraction of labels (LABEL_NOISE_RATE) are randomly flipped
   to a neighbouring class.  This simulates real-world measurement noise
   and prevents a model from reaching artificially perfect accuracy.

4. **Balanced classes**
   Equal sample counts per class (±1 for rounding) to avoid naive
   majority-class bias during training.

Usage
-----
As a module:
    from src.data_generator import generate_demo_dataset
    df = generate_demo_dataset()

From the command line:
    python -m src.data_generator
    python -m src.data_generator --samples 2000 --seed 99 --output data/raw/demo_dataset.csv
"""

from __future__ import annotations  # PEP 563 — defers annotation evaluation (Python 3.9 compat)

import argparse
import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_N_SAMPLES: int  = 1000
DEFAULT_OUTPUT:    str  = os.path.join("data", "raw", "demo_dataset.csv")
DEFAULT_SEED:      int  = 42

# Fraction of each sample's features drawn from the *class* profile
# vs. the *global* (population-average) distribution.
# 0.0 → pure noise (no class signal)
# 1.0 → pure class profile (maximum predictability)
SIGNAL_WEIGHT: float = 0.72    # 72 % class signal, 28 % population noise

# Probability that a correctly-assigned label is randomly flipped
# to an adjacent class.  Simulates annotation/measurement noise.
LABEL_NOISE_RATE: float = 0.06  # 6 % of labels are flipped

#: Ordered feature names (must match preprocessing.FEATURE_COLS).
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

TARGET_COL: str = "procrastination_reason"

#: Valid class names in display order.
CLASS_LABELS: list[str] = [
    "Digital Distraction",
    "Low Motivation",
    "Task Difficulty",
    "Fear of Failure",
    "Perfectionism",
    "Poor Sleep / Low Energy",
    "Lack of Clarity",
    "Poor Time Management",
]

# Features that must be integers after generation.
_INT_FEATURES: frozenset[str] = frozenset({
    "task_difficulty", "task_interest", "stress_level", "phone_distraction",
    "energy_level", "task_clarity", "confidence", "deadline_pressure",
    "perfectionism", "previous_procrastination",
})

# Per-feature (min, max) clip bounds.
_CLIP: dict[str, tuple[float, float]] = {
    "sleep_hours":              (2.0, 10.0),
    "task_difficulty":          (1.0, 10.0),
    "task_interest":            (1.0, 10.0),
    "stress_level":             (1.0, 10.0),
    "phone_distraction":        (1.0, 10.0),
    "social_media_usage":       (0.0,  8.0),
    "energy_level":             (1.0, 10.0),
    "task_clarity":             (1.0, 10.0),
    "confidence":               (1.0, 10.0),
    "deadline_pressure":        (1.0, 10.0),
    "perfectionism":            (1.0, 10.0),
    "previous_procrastination": (1.0, 10.0),
}


# ══════════════════════════════════════════════════════════════════════════════
# Class feature profiles
# ══════════════════════════════════════════════════════════════════════════════
# Each entry is (mean, std) per feature for a given procrastination class.
#
# Design notes:
#   • std values are intentionally generous (1.2–1.8) so class distributions
#     heavily overlap — preventing single-feature trivial prediction.
#   • Each class has AT MOST 2–3 features with a strong mean shift; the rest
#     hover near the global population mean (~5.5).
#   • The global population profile below defines the "noise" distribution
#     that is mixed in via SIGNAL_WEIGHT.

_GLOBAL_PROFILE: dict[str, tuple[float, float]] = {
    "sleep_hours":              (6.5, 1.3),
    "task_difficulty":          (5.5, 1.8),
    "task_interest":            (5.0, 1.8),
    "stress_level":             (5.5, 1.8),
    "phone_distraction":        (5.5, 1.8),
    "social_media_usage":       (3.5, 1.5),
    "energy_level":             (5.0, 1.8),
    "task_clarity":             (5.0, 1.8),
    "confidence":               (5.0, 1.8),
    "deadline_pressure":        (5.5, 1.8),
    "perfectionism":            (5.0, 1.8),
    "previous_procrastination": (5.5, 1.8),
}

_CLASS_PROFILES: dict[str, dict[str, tuple[float, float]]] = {

    # Primary signal: very high phone_distraction + social_media_usage
    "Digital Distraction": {
        "sleep_hours":              (6.5, 1.3),
        "task_difficulty":          (5.0, 1.8),
        "task_interest":            (4.2, 1.8),
        "stress_level":             (5.5, 1.8),
        "phone_distraction":        (8.8, 0.9),   # ← strong signal
        "social_media_usage":       (6.2, 1.1),   # ← strong signal
        "energy_level":             (5.5, 1.8),
        "task_clarity":             (5.5, 1.8),
        "confidence":               (5.5, 1.8),
        "deadline_pressure":        (4.5, 1.8),
        "perfectionism":            (4.5, 1.8),
        "previous_procrastination": (6.8, 1.5),
    },

    # Primary signal: very low task_interest + energy_level + confidence
    "Low Motivation": {
        "sleep_hours":              (6.0, 1.3),
        "task_difficulty":          (5.0, 1.8),
        "task_interest":            (2.6, 1.2),   # ← strong signal
        "stress_level":             (6.0, 1.8),
        "phone_distraction":        (6.0, 1.8),
        "social_media_usage":       (4.5, 1.5),
        "energy_level":             (3.2, 1.2),   # ← strong signal
        "task_clarity":             (5.0, 1.8),
        "confidence":               (3.5, 1.3),   # ← moderate signal
        "deadline_pressure":        (5.0, 1.8),
        "perfectionism":            (4.0, 1.8),
        "previous_procrastination": (7.5, 1.3),
    },

    # Primary signal: very high task_difficulty + low confidence
    "Task Difficulty": {
        "sleep_hours":              (6.5, 1.3),
        "task_difficulty":          (8.9, 0.9),   # ← strong signal
        "task_interest":            (5.5, 1.8),
        "stress_level":             (7.0, 1.5),
        "phone_distraction":        (4.5, 1.8),
        "social_media_usage":       (3.5, 1.5),
        "energy_level":             (5.0, 1.8),
        "task_clarity":             (4.5, 1.8),
        "confidence":               (3.5, 1.3),   # ← moderate signal
        "deadline_pressure":        (7.0, 1.5),
        "perfectionism":            (5.5, 1.8),
        "previous_procrastination": (6.0, 1.5),
    },

    # Primary signal: very high stress_level + very low confidence
    "Fear of Failure": {
        "sleep_hours":              (5.5, 1.3),
        "task_difficulty":          (6.5, 1.8),
        "task_interest":            (5.0, 1.8),
        "stress_level":             (8.5, 0.9),   # ← strong signal
        "phone_distraction":        (5.0, 1.8),
        "social_media_usage":       (4.0, 1.5),
        "energy_level":             (4.5, 1.8),
        "task_clarity":             (5.5, 1.8),
        "confidence":               (2.5, 1.0),   # ← strong signal
        "deadline_pressure":        (8.0, 1.2),
        "perfectionism":            (7.0, 1.3),
        "previous_procrastination": (7.5, 1.3),
    },

    # Primary signal: extremely high perfectionism; moderate everything else
    "Perfectionism": {
        "sleep_hours":              (6.0, 1.3),
        "task_difficulty":          (6.5, 1.8),
        "task_interest":            (6.5, 1.8),
        "stress_level":             (7.5, 1.3),
        "phone_distraction":        (4.0, 1.8),
        "social_media_usage":       (3.0, 1.3),
        "energy_level":             (5.5, 1.8),
        "task_clarity":             (6.0, 1.8),
        "confidence":               (5.0, 1.8),
        "deadline_pressure":        (6.5, 1.8),
        "perfectionism":            (9.4, 0.6),   # ← strong signal
        "previous_procrastination": (7.0, 1.3),
    },

    # Primary signal: very low sleep_hours + very low energy_level
    "Poor Sleep / Low Energy": {
        "sleep_hours":              (3.6, 0.9),   # ← strong signal
        "task_difficulty":          (5.5, 1.8),
        "task_interest":            (5.0, 1.8),
        "stress_level":             (6.5, 1.5),
        "phone_distraction":        (5.5, 1.8),
        "social_media_usage":       (4.0, 1.5),
        "energy_level":             (2.1, 0.9),   # ← strong signal
        "task_clarity":             (5.0, 1.8),
        "confidence":               (4.0, 1.8),
        "deadline_pressure":        (5.5, 1.8),
        "perfectionism":            (4.5, 1.8),
        "previous_procrastination": (6.5, 1.5),
    },

    # Primary signal: extremely low task_clarity
    "Lack of Clarity": {
        "sleep_hours":              (6.5, 1.3),
        "task_difficulty":          (6.0, 1.8),
        "task_interest":            (5.5, 1.8),
        "stress_level":             (6.0, 1.8),
        "phone_distraction":        (5.0, 1.8),
        "social_media_usage":       (4.0, 1.5),
        "energy_level":             (5.0, 1.8),
        "task_clarity":             (1.6, 0.8),   # ← strong signal
        "confidence":               (4.0, 1.8),
        "deadline_pressure":        (5.0, 1.8),
        "perfectionism":            (5.0, 1.8),
        "previous_procrastination": (6.5, 1.5),
    },

    # Primary signal: very high deadline_pressure + high previous_procrastination
    "Poor Time Management": {
        "sleep_hours":              (6.5, 1.3),
        "task_difficulty":          (5.5, 1.8),
        "task_interest":            (5.0, 1.8),
        "stress_level":             (6.5, 1.5),
        "phone_distraction":        (6.5, 1.8),
        "social_media_usage":       (5.0, 1.5),
        "energy_level":             (5.0, 1.8),
        "task_clarity":             (4.5, 1.8),
        "confidence":               (4.5, 1.8),
        "deadline_pressure":        (8.5, 0.9),   # ← strong signal
        "perfectionism":            (5.0, 1.8),
        "previous_procrastination": (8.0, 1.1),   # ← strong signal
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Core generation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _sample_feature(
    feat: str,
    class_label: str,
    rng: np.random.Generator,
) -> float:
    """
    Sample a single feature value for one record using a mixture model.

    The value is drawn from a weighted average of:
      • The class-specific Gaussian profile  (weight = SIGNAL_WEIGHT)
      • The global population Gaussian        (weight = 1 - SIGNAL_WEIGHT)

    This intentional mixing prevents any single feature from being a
    perfect discriminator, forcing the ML model to use *combinations*
    of features to classify accurately.

    Parameters
    ----------
    feat : str
        Feature name.
    class_label : str
        Assigned class label for this record.
    rng : np.random.Generator
        Seeded random number generator.

    Returns
    -------
    float
        Sampled (pre-clip) feature value.
    """
    c_mean, c_std = _CLASS_PROFILES[class_label][feat]
    g_mean, g_std = _GLOBAL_PROFILE[feat]

    # Mixture: SIGNAL_WEIGHT fraction from class profile, rest from global
    class_val  = rng.normal(c_mean, c_std)
    global_val = rng.normal(g_mean, g_std)
    return SIGNAL_WEIGHT * class_val + (1.0 - SIGNAL_WEIGHT) * global_val


def _apply_correlations(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Inject realistic inter-feature correlations via small additive adjustments.

    Correlations modelled:
      • sleep_hours  ↔  energy_level      (positive, r ≈ 0.5)
      • stress_level ↔  confidence         (negative, r ≈ −0.4)
      • phone_distraction ↔ social_media  (positive, r ≈ 0.55)
      • task_difficulty  ↔ stress_level   (positive, r ≈ 0.35)

    The adjustments are noisy so they do not create perfect correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-only frame (float, pre-clip).
    rng : np.random.Generator

    Returns
    -------
    pd.DataFrame
        Modified in-place and returned.
    """
    n = len(df)
    noise = lambda scale: rng.normal(0, scale, size=n)   # noqa: E731

    # sleep → energy: more sleep → slightly more energy
    df["energy_level"] += 0.30 * (df["sleep_hours"] - 6.0) + noise(0.4)

    # stress → confidence: more stress → slightly less confidence
    df["confidence"] -= 0.25 * (df["stress_level"] - 5.5) + noise(0.4)

    # phone → social media: phone addicts tend to use more social media
    df["social_media_usage"] += 0.20 * (df["phone_distraction"] - 5.5) + noise(0.3)

    # task_difficulty → stress: harder task → more stress
    df["stress_level"] += 0.18 * (df["task_difficulty"] - 5.5) + noise(0.4)

    return df


def _clip_and_round(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-feature range clipping and round integer features.

    Parameters
    ----------
    df : pd.DataFrame
        Feature frame (may contain out-of-range values after correlation
        adjustments).

    Returns
    -------
    pd.DataFrame
        In-range, correctly-typed feature frame.
    """
    for feat in FEATURE_COLS:
        lo, hi = _CLIP[feat]
        df[feat] = df[feat].clip(lo, hi)
        if feat in _INT_FEATURES:
            df[feat] = df[feat].round().astype(int).clip(int(lo), int(hi))
        else:
            df[feat] = df[feat].round(1)
    return df


def _apply_label_noise(
    labels: list[str],
    rng: np.random.Generator,
    rate: float = LABEL_NOISE_RATE,
) -> list[str]:
    """
    Randomly flip a fraction of labels to a neighbouring class.

    This simulates real-world annotation ambiguity (e.g., a student who
    is both sleep-deprived *and* distracted might be labelled either way).
    It also prevents models from achieving artificially high accuracy on
    trivially generated data.

    Parameters
    ----------
    labels : list[str]
        Original class labels (one per record).
    rng : np.random.Generator
    rate : float
        Fraction of labels to flip (default: LABEL_NOISE_RATE).

    Returns
    -------
    list[str]
        Labels with noise applied.
    """
    labels = labels.copy()
    n = len(labels)
    flip_mask = rng.random(n) < rate
    for i in np.where(flip_mask)[0]:
        # Pick any other class uniformly at random
        current = labels[i]
        others  = [c for c in CLASS_LABELS if c != current]
        labels[i] = rng.choice(others)
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_demo_dataset(
    n_samples: int  = DEFAULT_N_SAMPLES,
    output_path: str = DEFAULT_OUTPUT,
    random_seed: int = DEFAULT_SEED,
    label_noise_rate: float = LABEL_NOISE_RATE,
    signal_weight: float = SIGNAL_WEIGHT,
) -> pd.DataFrame:
    """
    Generate and save the synthetic demo dataset.

    ⚠️  This produces SYNTHETIC data only — see module docstring.

    Algorithm
    ---------
    1. Assign balanced class labels (equal counts per class, ±1).
    2. For each record, sample each feature from a mixture of the
       class-specific Gaussian and the global population Gaussian
       (controlled by ``signal_weight``).
    3. Inject realistic inter-feature correlations.
    4. Clip values to valid ranges; round integer features.
    5. Apply label noise (randomly flip ``label_noise_rate`` fraction
       of labels to a different class).
    6. Shuffle all records.
    7. Add a ``data_source`` sentinel column = ``"SYNTHETIC_DEMO_ONLY"``.
    8. Write to CSV.

    Parameters
    ----------
    n_samples : int
        Total number of rows (balanced across 8 classes).
    output_path : str
        Destination CSV path (parent dirs created if absent).
    random_seed : int
        NumPy seed for reproducibility.
    label_noise_rate : float
        Fraction of labels randomly flipped (0.0 = no noise).
    signal_weight : float
        Class-profile mixture weight (0.0 = all noise, 1.0 = pure profile).

    Returns
    -------
    pd.DataFrame
        The generated dataset (also written to ``output_path``).
    """
    rng = np.random.default_rng(random_seed)

    # ── Step 1: balanced class assignment ────────────────────────────────────
    base, remainder = divmod(n_samples, len(CLASS_LABELS))
    labels: list[str] = []
    for i, cls in enumerate(CLASS_LABELS):
        count = base + (1 if i < remainder else 0)
        labels.extend([cls] * count)

    # ── Step 2: sample features (mixture model) ───────────────────────────────
    records: list[dict] = []
    for label in labels:
        row = {
            feat: _sample_feature(feat, label, rng)
            for feat in FEATURE_COLS
        }
        records.append(row)

    df = pd.DataFrame(records)

    # ── Step 3: inter-feature correlations ────────────────────────────────────
    df = _apply_correlations(df, rng)

    # ── Step 4: clip + round ──────────────────────────────────────────────────
    df = _clip_and_round(df)

    # ── Step 5: label noise ───────────────────────────────────────────────────
    noisy_labels = _apply_label_noise(labels, rng, rate=label_noise_rate)
    df[TARGET_COL] = noisy_labels

    # ── Step 6: shuffle ───────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=int(random_seed)).reset_index(drop=True)

    # ── Step 7: sentinel column ───────────────────────────────────────────────
    # This column travels with the CSV so anyone reading it knows it is fake.
    df.insert(0, "data_source", "SYNTHETIC_DEMO_ONLY")

    # ── Step 8: write CSV ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(df: pd.DataFrame, output_path: str, elapsed_ms: float) -> None:
    """Print a generation summary to stdout."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  ⚠️   SYNTHETIC DEMO DATASET — NOT REAL DATA")
    print(sep)
    print(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Records    : {len(df):,}")
    print(f"  Features   : {len(FEATURE_COLS)}")
    print(f"  Classes    : {len(CLASS_LABELS)}")
    print(f"  Saved to   : {output_path}")
    print(f"  Time taken : {elapsed_ms:.0f} ms")
    print(f"\n  Class distribution (after label noise):")
    counts = df[TARGET_COL].value_counts()
    for cls in CLASS_LABELS:
        bar = "█" * int(counts.get(cls, 0) / max(counts) * 25)
        print(f"    {cls:<30} {counts.get(cls, 0):>4}  {bar}")
    print(f"\n  Feature statistics (first 4):")
    print(df[FEATURE_COLS[:4]].describe().round(2).to_string())
    print(f"\n{sep}")
    print("  ⚠️  IMPORTANT DISCLAIMER")
    print(sep)
    print(textwrap.fill(
        "This dataset is ENTIRELY SYNTHETIC. It was programmatically "
        "generated to test the ML pipeline. It must NOT be used in any "
        "publication, report, or application as evidence of real student "
        "behaviour or procrastination patterns.",
        width=63, initial_indent="  ", subsequent_indent="  ",
    ))
    print(textwrap.fill(
        "Real project results MUST use a properly collected and labelled "
        "dataset (e.g., via a validated survey instrument such as the "
        "Procrastination Assessment Scale for Students, PASS).",
        width=63, initial_indent="  ", subsequent_indent="  ",
    ))
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic demo dataset for the "
            "'Why Am I Procrastinating?' ML pipeline.\n\n"
            "⚠️  Output is SYNTHETIC data only — not real student data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples", type=int, default=DEFAULT_N_SAMPLES,
        metavar="N",
        help=f"Number of rows to generate (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        metavar="INT",
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--noise", type=float, default=LABEL_NOISE_RATE,
        metavar="FLOAT",
        help=f"Label noise rate 0.0–1.0 (default: {LABEL_NOISE_RATE})",
    )
    parser.add_argument(
        "--signal", type=float, default=SIGNAL_WEIGHT,
        metavar="FLOAT",
        help=(
            f"Class-signal mixture weight 0.0–1.0 (default: {SIGNAL_WEIGHT}). "
            "Lower = more overlap between classes."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    import time

    args = _parse_args()

    t0 = time.perf_counter()
    df = generate_demo_dataset(
        n_samples        = args.samples,
        output_path      = args.output,
        random_seed      = args.seed,
        label_noise_rate = args.noise,
        signal_weight    = args.signal,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _print_summary(df, args.output, elapsed_ms)

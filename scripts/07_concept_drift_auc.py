# -*- coding: utf-8 -*-
"""
07_concept_drift_auc.py - Concept drift detection (rolling-window AUC)
======================================================================
Extension 2: score each rolling window with the baseline model and compute
its true AUC, observing how the model's discrimination changes over time.
A sustained AUC decline is the key signal of CONCEPT drift and PERFORMANCE
drift.

Distinguishing the three drift types (glossary):
    - Covariate Shift: P(X) changes, P(Y|X) fixed - feature distributions
    - Concept Drift: P(Y|X) changes - the feature-label relationship
    - Prior Probability Shift: P(Y) changes - overall default rate
AUC decline is a proxy for concept drift: if feature distributions change
but AUC stays stable, the model still separates good / bad in the new
population (covariate shift without harming the model); if AUC clearly
drops, the feature-to-label mapping is changing (concept drift) and the
model needs retraining.

Outputs (under output/rolling/):
    rolling_auc_trend.csv   per-window AUC / bad rate / performance drop
    auc_trend.png           AUC trend + baseline reference + alert lines
"""
import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ensure_dir, load_csv, now_str, setup_utf8

setup_utf8()

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")
WINDOW_DIR = os.path.join(PROC_DIR, "rolling_windows")
MODEL_DIR = os.path.join(ROOT, "output", "model")
ROLL_DIR = os.path.join(ROOT, "output", "rolling")

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
TARGET = "SeriousDlqin2yrs"

# AUC-drop alert thresholds (relative to baseline): >=1% -> watch,
# >=3% -> trigger retraining review
WARN_DROP = 0.01
RED_DROP = 0.03


def auc_score(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(auc(fpr, tpr))


def main():
    ensure_dir(ROLL_DIR)

    print(f"[{now_str()}] Loading baseline model and rolling windows")
    ref_val = pd.read_csv(os.path.join(PROC_DIR, "reference_val.csv"), encoding="utf-8-sig")
    model = joblib.load(os.path.join(MODEL_DIR, "baseline_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    meta = json.load(open(os.path.join(WINDOW_DIR, "meta.json"), encoding="utf-8"))

    ref_score = scaler.transform(ref_val[FEATURES].values)
    baseline_auc = auc_score(ref_val[TARGET].values, model.predict_proba(ref_score)[:, 1])

    rows = []
    for m in meta:
        wdf = load_csv(m["file"])
        X = scaler.transform(wdf[FEATURES].values)
        y_prob = model.predict_proba(X)[:, 1]
        a = auc_score(wdf[TARGET].values, y_prob)
        drop = baseline_auc - a
        if drop >= RED_DROP:
            level = "SevereDegradation_Retrain"
        elif drop >= WARN_DROP:
            level = "MildDegradation_Watch"
        else:
            level = "Normal"
        rows.append({
            "Window": m["name"],
            "Date": m["date"],
            "DriftIntensity": m["intensity"],
            "Size": len(wdf),
            "BadRate": round(float(wdf[TARGET].mean()), 4),
            "AUC": round(a, 4),
            "AUC_Drop": round(drop, 4),
            "Status": level,
        })

    trend = pd.DataFrame(rows)
    trend.to_csv(os.path.join(ROLL_DIR, "rolling_auc_trend.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # AUC trend chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.0))
    x = np.arange(len(trend))
    ax.plot(x, trend["AUC"], marker="o", color="#4C72B0", lw=2, label="Window AUC")
    ax.axhline(baseline_auc, color="#2E9E5B", ls="--", lw=1.2, label=f"Baseline AUC = {baseline_auc:.4f}")
    ax.axhline(baseline_auc - WARN_DROP, color="#E8A33D", ls=":", lw=1.2)
    ax.axhline(baseline_auc - RED_DROP, color="#D9534F", ls=":", lw=1.2)
    ax.text(len(x) - 0.4, baseline_auc - WARN_DROP - 0.004, "Watch line -1%", color="#E8A33D", fontsize=9, ha="right")
    ax.text(len(x) - 0.4, baseline_auc - RED_DROP - 0.004, "Retrain line -3%", color="#D9534F", fontsize=9, ha="right")

    # Mark performance status
    for i, r in trend.iterrows():
        if "Severe" in r["Status"]:
            ax.scatter(i, r["AUC"], s=120, facecolors="none", edgecolors="#D9534F", linewidths=2)
        elif "Mild" in r["Status"]:
            ax.scatter(i, r["AUC"], s=120, facecolors="none", edgecolors="#E8A33D", linewidths=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Window']}\n{r['Date'][5:]}" for _, r in trend.iterrows()], fontsize=9)
    # y range accommodates the lowest window AUC (0.615) and the threshold lines
    y_lo = min(trend["AUC"].min(), baseline_auc - RED_DROP) - 0.02
    y_hi = max(trend["AUC"].max(), baseline_auc) + 0.015
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Monitoring period (rolling windows, simulated monthly)")
    ax.set_ylabel("AUC")
    ax.set_title("Rolling-Window AUC Performance Drift (Concept / Performance Drift)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROLL_DIR, "auc_trend.png"), dpi=120)
    plt.close(fig)

    # Conclusion
    worst = trend.loc[trend["AUC_Drop"].idxmax()]
    conclusion = {
        "baseline_auc": round(baseline_auc, 4),
        "worst_window": worst["Window"],
        "worst_auc": worst["AUC"],
        "worst_drop": worst["AUC_Drop"],
        "severe_windows": trend.loc[trend["Status"] == "SevereDegradation_Retrain", "Window"].tolist(),
        "conclusion": (
            f"AUC hit its lowest value {worst['AUC']:.4f} in {worst['Window']}, a drop of "
            f"{worst['AUC_Drop']:.4f} vs baseline - combined with the feature-distribution "
            f"drift (rising PSI) this is concept / performance drift; a model retraining "
            f"review is recommended."
        ),
    }
    with open(os.path.join(ROLL_DIR, "concept_drift_summary.json"), "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=2)

    print("\n===== Rolling AUC trend =====")
    print(trend.to_string(index=False))
    print(f"\nConclusion: {conclusion['conclusion']}")
    print(f"[{now_str()}] Done -> {ROLL_DIR}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
06_rolling_psi_trend.py - Rolling-window PSI trend monitoring (Early Warning)
==============================================================================
Extension 1: split the current-period data into 8 rolling windows (simulating
monthly / weekly monitoring). For each window, compute the model SCORE PSI and
the FEATURE-AVERAGE PSI against the reference period, then plot the PSI trend
line over time with the 0.10 / 0.25 alert-threshold lines.

Business value: trends reveal drift earlier than a single snapshot PSI
(trend is more sensitive than snapshot).

Window design (progressive worsening, mimicking real business):
    W1-W4  normal borrower mix (no drift injected)
    W5      mild drift, intensity=0.25
    W6      moderate drift, intensity=0.50
    W7      heavier drift, intensity=0.75
    W8      severe drift, intensity=1.00

Outputs (under output/rolling/):
    rolling_psi_trend.csv   per-window PSI / KL / alert level
    psi_trend.png           trend chart (score PSI + feature-avg PSI + lines)
    early_warning.json      first moderate / severe warning window + conclusion
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Feature list (consistent with 01-05)
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

from utils import (
    build_rolling_windows,
    classify_psi,
    compute_kld,
    compute_psi,
    ensure_dir,
    now_str,
    save_rolling_windows,
    setup_utf8,
)

setup_utf8()

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")
WINDOW_DIR = os.path.join(PROC_DIR, "rolling_windows")
MODEL_DIR = os.path.join(ROOT, "output", "model")
ROLL_DIR = os.path.join(ROOT, "output", "rolling")

COLOR_GREEN, COLOR_YELLOW, COLOR_RED = "#2E9E5B", "#E8A33D", "#D9534F"


def score_of(df: pd.DataFrame, model, scaler) -> np.ndarray:
    """Output P(bad) with the baseline model."""
    X = scaler.transform(df[FEATURES].values)
    return model.predict_proba(X)[:, 1]


def feature_avg_psi(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> float:
    """Mean PSI across the 10 features (overall feature-drift level)."""
    return float(np.mean([compute_psi(ref_df[c].values, cur_df[c].values)["psi"] for c in FEATURES]))


def main():
    ensure_dir(ROLL_DIR)
    ensure_dir(WINDOW_DIR)

    print(f"[{now_str()}] Loading reference data and model")
    ref_val = pd.read_csv(os.path.join(PROC_DIR, "reference_val.csv"), encoding="utf-8-sig")
    current_raw = pd.read_csv(os.path.join(PROC_DIR, "current_raw.csv"), encoding="utf-8-sig")
    model = joblib.load(os.path.join(MODEL_DIR, "baseline_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

    ref_score = score_of(ref_val, model, scaler)

    # ------------------------------------------------------------------
    # 1) Build 8 rolling windows (normal early, progressive drift later)
    # ------------------------------------------------------------------
    windows = build_rolling_windows(current_raw, n_windows=8)
    save_rolling_windows(windows, WINDOW_DIR)
    print(f"[{now_str()}] Built {len(windows)} rolling windows -> {WINDOW_DIR}")

    # ------------------------------------------------------------------
    # 2) Per-window PSI / KL
    # ------------------------------------------------------------------
    rows = []
    for w in windows:
        s = score_of(w["df"], model, scaler)
        psi_res = compute_psi(ref_score, s)
        kld_res = compute_kld(ref_score, s)
        fpsi = feature_avg_psi(ref_val, w["df"])
        rows.append({
            "Window": w["name"],
            "Date": w["date"],
            "DriftIntensity": w["intensity"],
            "Size": len(w["df"]),
            "ScorePSI": round(psi_res["psi"], 6),
            "ScoreKL": round(kld_res["kld"], 6),
            "FeatureAvgPSI": round(fpsi, 6),
            "ScorePSI_Level": classify_psi(psi_res["psi"]),
            "FirstAlert": "",  # flagged below
        })

    trend = pd.DataFrame(rows)

    # Flag the first window triggering moderate (0.10) / severe (0.25) alert
    first_yellow = trend[trend["ScorePSI"] >= 0.10]["Window"].min()
    first_red = trend[trend["ScorePSI"] >= 0.25]["Window"].min()
    trend["FirstAlert"] = np.where(trend["Window"] == first_yellow, "First Moderate", trend["FirstAlert"])
    trend["FirstAlert"] = np.where(trend["Window"] == first_red, "First Severe", trend["FirstAlert"])

    trend.to_csv(os.path.join(ROLL_DIR, "rolling_psi_trend.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 3) Trend chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(trend))
    ax.plot(x, trend["ScorePSI"], marker="o", color="#D9534F", label="Model Score PSI", lw=2)
    ax.plot(x, trend["FeatureAvgPSI"], marker="s", color="#4C72B0", label="Feature Avg PSI", lw=1.6, ls="--")
    ax.axhline(0.10, color=COLOR_YELLOW, ls="--", lw=1.2)
    ax.axhline(0.25, color=COLOR_RED, ls="--", lw=1.2)
    ax.text(len(x) - 0.4, 0.103, "Moderate line 0.10", color=COLOR_YELLOW, fontsize=9, ha="right")
    ax.text(len(x) - 0.4, 0.255, "Severe line 0.25", color=COLOR_RED, fontsize=9, ha="right")

    # Shade the alert windows
    for i, r in trend.iterrows():
        if "Severe" in r["ScorePSI_Level"]:
            ax.axvspan(i - 0.4, i + 0.4, color=COLOR_RED, alpha=0.10)
        elif "Moderate" in r["ScorePSI_Level"]:
            ax.axvspan(i - 0.4, i + 0.4, color=COLOR_YELLOW, alpha=0.10)

    # Annotate the first warning windows
    for wname, note, color in [(first_yellow, "First Moderate", COLOR_YELLOW), (first_red, "First Severe", COLOR_RED)]:
        if pd.notna(wname):
            idx = trend[trend["Window"] == wname].index[0]
            ax.annotate(note, xy=(idx, trend.loc[idx, "ScorePSI"]), xytext=(idx, trend.loc[idx, "ScorePSI"] + 0.4),
                        arrowprops=dict(arrowstyle="->", color=color), color=color, fontsize=10, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Window']}\n{r['Date'][5:]}" for _, r in trend.iterrows()], fontsize=9)
    ax.set_xlabel("Monitoring period (rolling windows, simulated monthly)")
    ax.set_ylabel("PSI")
    ax.set_title("Rolling-Window PSI Trend Monitoring (Early Warning)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    trend_path = os.path.join(ROLL_DIR, "psi_trend.png")
    fig.savefig(trend_path, dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4) Early-warning conclusion
    # ------------------------------------------------------------------
    early_warning = {
        "first_moderate_window": str(first_yellow) if pd.notna(first_yellow) else None,
        "first_moderate_date": str(trend.loc[trend["Window"] == first_yellow, "Date"].iloc[0]) if pd.notna(first_yellow) else None,
        "first_severe_window": str(first_red) if pd.notna(first_red) else None,
        "first_severe_date": str(trend.loc[trend["Window"] == first_red, "Date"].iloc[0]) if pd.notna(first_red) else None,
        "conclusion": (
            f"Drift first triggered a moderate alert (PSI>=0.10) in window {first_yellow} and "
            f"escalated to a severe alert (PSI>=0.25) in window {first_red} - "
            f"exposing the risk earlier than fixed snapshot monitoring."
            if pd.notna(first_yellow) else "No alert triggered during the monitored period"
        ),
    }
    with open(os.path.join(ROLL_DIR, "early_warning.json"), "w", encoding="utf-8") as f:
        json.dump(early_warning, f, ensure_ascii=False, indent=2)

    print("\n===== Rolling PSI trend =====")
    print(trend[["Window", "Date", "ScorePSI", "ScorePSI_Level", "FeatureAvgPSI", "FirstAlert"]].to_string(index=False))
    print(f"\nConclusion: {early_warning['conclusion']}")
    print(f"[{now_str()}] Done -> {ROLL_DIR}")


if __name__ == "__main__":
    main()

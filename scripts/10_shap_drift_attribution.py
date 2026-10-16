# -*- coding: utf-8 -*-
"""
10_shap_drift_attribution.py - SHAP drift attribution
======================================================
Extension 4b: use SHAP (SHapley Additive exPlanations) to locate which
feature's drift contributes most to the model-score change.

Core idea:
    PSI only tells us "the feature distribution changed", not "what that
    change means for the model". Combining SHAP (each feature's contribution
    to the model output, per sample) answers:
        - How did a feature's model influence (mean|SHAP|) change between the
          reference and drifted periods?
        - A feature that drifts hard (high PSI) AND that the model relies on
          (high SHAP) = the top-priority item to investigate.

Attribution metric:
    Drift Impact = normalized PSI x normalized drifted mean|SHAP|
    - combines "drift magnitude" and "model sensitivity", outputs a ranked
      top-priority investigation list

Outputs (under output/gbdt/):
    shap_drift_report.csv         per feature: ref/drifted mean|SHAP|, change
                                  rate, PSI, attribution level
    shap_importance_shift.png     two-period model-influence comparison bar
    shap_beeswarm_drifted.png     drifted-period SHAP beeswarm plot
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
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import compute_psi, ensure_dir, now_str, setup_utf8

setup_utf8()

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")
GBDT_DIR = os.path.join(ROOT, "output", "gbdt")

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


def main():
    ensure_dir(GBDT_DIR)

    print(f"[{now_str()}] Loading LightGBM model and data")
    model = joblib.load(os.path.join(GBDT_DIR, "lgb_model.joblib"))
    ref_val = pd.read_csv(os.path.join(PROC_DIR, "reference_val.csv"), encoding="utf-8-sig")
    cur_drifted = pd.read_csv(os.path.join(PROC_DIR, "current_drifted.csv"), encoding="utf-8-sig")

    X_ref = ref_val[FEATURES]
    X_cur = cur_drifted[FEATURES]

    # ------------------------------------------------------------------
    # 1) TreeExplainer (exact SHAP for tree models)
    # ------------------------------------------------------------------
    print(f"[{now_str()}] Computing SHAP values "
          f"(reference {len(X_ref):,} rows / drifted {len(X_cur):,} rows) ...")
    explainer = shap.TreeExplainer(model)
    # New API: returns an Explanation object; for binary classification take
    # the positive-class contribution (values may be a list)
    shap_ref = explainer(X_ref, check_additivity=False).values
    shap_cur = explainer(X_cur, check_additivity=False).values
    if isinstance(shap_ref, list):
        shap_ref = shap_ref[1]
    if isinstance(shap_cur, list):
        shap_cur = shap_cur[1]

    # ------------------------------------------------------------------
    # 2) Attribution metrics
    # ------------------------------------------------------------------
    mean_abs_ref = np.abs(shap_ref).mean(axis=0)
    mean_abs_cur = np.abs(shap_cur).mean(axis=0)
    change = (mean_abs_cur - mean_abs_ref) / (mean_abs_ref + 1e-9)

    rows = []
    for i, f in enumerate(FEATURES):
        psi = compute_psi(X_ref[f].values, X_cur[f].values)["psi"]
        rows.append({
            "Feature": f,
            "Ref_MeanAbsSHAP": round(float(mean_abs_ref[i]), 6),
            "Drifted_MeanAbsSHAP": round(float(mean_abs_cur[i]), 6),
            "ImportanceChange": round(float(change[i]), 4),
            "FeaturePSI": round(float(psi), 6),
        })
    report = pd.DataFrame(rows)

    # Normalize and combine into "Drift Impact" (50/50 weight)
    psi_norm = (report["FeaturePSI"] - report["FeaturePSI"].min()) / (report["FeaturePSI"].max() - report["FeaturePSI"].min() + 1e-9)
    imp_norm = (report["Drifted_MeanAbsSHAP"] - report["Drifted_MeanAbsSHAP"].min()) / (
            report["Drifted_MeanAbsSHAP"].max() - report["Drifted_MeanAbsSHAP"].min() + 1e-9)
    report["DriftImpact"] = ((psi_norm + imp_norm) / 2).round(4)

    # Attribution level: top-3 by impact = high priority; others by PSI level
    top3 = report.nlargest(3, "DriftImpact")["Feature"].tolist()
    report["Priority"] = np.where(report["Feature"].isin(top3), "HighPriority",
                                  np.where(report["FeaturePSI"] >= 0.25, "Watch", "Normal"))
    report = report.sort_values("DriftImpact", ascending=False).reset_index(drop=True)
    report.to_csv(os.path.join(GBDT_DIR, "shap_drift_report.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # 3) Charts: two-period influence comparison + drifted beeswarm
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_df = report.sort_values("Drifted_MeanAbsSHAP")
    y = np.arange(len(plot_df))
    ax.barh(y + 0.18, plot_df["Ref_MeanAbsSHAP"], height=0.36, color="#9BB7D4", label="Reference mean|SHAP|")
    ax.barh(y - 0.18, plot_df["Drifted_MeanAbsSHAP"], height=0.36, color="#D9534F", label="Drifted mean|SHAP|")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Feature"], fontsize=9)
    ax.set_xlabel("mean |SHAP| (model influence)")
    ax.set_title("Feature Model-Influence Shift (SHAP Importance Shift)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(GBDT_DIR, "shap_importance_shift.png"), dpi=120)
    plt.close(fig)

    # Drifted-period beeswarm (highly-dependent features at the top)
    fig2 = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_cur, X_cur, max_display=10, show=False)
    plt.title("SHAP Beeswarm - Drifted Period (red = high value, blue = low)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(GBDT_DIR, "shap_beeswarm_drifted.png"), dpi=120)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # 4) Conclusion
    # ------------------------------------------------------------------
    conclusion = {
        "top_priority_features": top3,
        "conclusion": (
            f"Combining feature drift (PSI) with model sensitivity (SHAP), the "
            f"top-priority features to investigate are: {', '.join(top3)} - they "
            f"both drift significantly and have the largest influence on the model "
            f"output, making them the main attribution channels for model failure "
            f"during the drifted period."
        ),
    }
    with open(os.path.join(GBDT_DIR, "shap_drift_conclusion.json"), "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=2)

    print("\n===== SHAP drift attribution report =====")
    print(report.to_string(index=False))
    print(f"\nConclusion: {conclusion['conclusion']}")
    print(f"[{now_str()}] Done -> {GBDT_DIR}")


if __name__ == "__main__":
    main()

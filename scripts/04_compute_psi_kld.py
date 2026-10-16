# -*- coding: utf-8 -*-
"""
04_compute_psi_kld.py - PSI / KL drift metric engine
=====================================================
For "reference vs early post-launch" and "reference vs drifted", compute:

    1. Feature-level:
       - PSI (Population Stability Index)
       - KL (Kullback-Leibler Divergence)
    2. Score-level:
       - PSI of the model output P(bad) (score drift - core monitoring metric)
    3. Performance-level:
       - AUC / KS comparison across the three datasets
         (did drift degrade the model?)

Outputs (under output/):
    psi_kld_feature_report.csv   per-feature PSI/KL detail (with alert class)
    psi_kld_score_report.csv     score-level PSI/KL
    monitoring_metrics.json      aggregated metrics
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import classify_psi, compute_kld, compute_psi, now_str, setup_utf8

setup_utf8()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "output", "model")
OUT_DIR = os.path.join(ROOT, "output")

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


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC: area under the ROC curve - the model's power to separate good/bad."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(auc(fpr, tpr))


def build_feature_report(ref: pd.DataFrame, current: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute PSI + KL per feature and output a detail table."""
    rows = []
    for col in FEATURES:
        psi_res = compute_psi(ref[col].values, current[col].values)
        kld_res = compute_kld(ref[col].values, current[col].values)
        rows.append({
            "Comparison": label,
            "Feature": col,
            "PSI": round(psi_res["psi"], 6),
            "PSI_Level": classify_psi(psi_res["psi"]),
            "KL_Divergence": round(kld_res["kld"], 6),
            "Ref_Median": round(float(ref[col].median()), 4),
            "Cur_Median": round(float(current[col].median()), 4),
            "Ref_Mean": round(float(ref[col].mean()), 4),
            "Cur_Mean": round(float(current[col].mean()), 4),
        })
    return pd.DataFrame(rows)


def build_score_report(ref_score: np.ndarray, cur_scores: dict) -> pd.DataFrame:
    """Compute PSI / KL on the model score distribution (top monitoring priority)."""
    rows = []
    for label, cur in cur_scores.items():
        psi_res = compute_psi(ref_score, cur)
        kld_res = compute_kld(ref_score, cur)
        rows.append({
            "Comparison": label,
            "Ref_Size": len(ref_score),
            "Cur_Size": len(cur),
            "PSI": round(psi_res["psi"], 6),
            "PSI_Level": classify_psi(psi_res["psi"]),
            "KL_Divergence": round(kld_res["kld"], 6),
            "Ref_MeanScore": round(float(np.mean(ref_score)), 4),
            "Cur_MeanScore": round(float(np.mean(cur)), 4),
        })
    return pd.DataFrame(rows)


def main():
    print(f"[{now_str()}] Loading data and model")
    ref_train = pd.read_csv(os.path.join(PROC_DIR, "reference_train.csv"), encoding="utf-8-sig")
    ref_val = pd.read_csv(os.path.join(PROC_DIR, "reference_val.csv"), encoding="utf-8-sig")
    cur_initial = pd.read_csv(os.path.join(PROC_DIR, "current_initial.csv"), encoding="utf-8-sig")
    cur_drifted = pd.read_csv(os.path.join(PROC_DIR, "current_drifted.csv"), encoding="utf-8-sig")

    model = joblib.load(os.path.join(MODEL_DIR, "baseline_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

    # ------------------------------------------------------------------
    # 1) Feature-level PSI/KL: reference(train) vs early / drifted
    # ------------------------------------------------------------------
    feat_initial = build_feature_report(ref_train, cur_initial, "ref_vs_initial")
    feat_drifted = build_feature_report(ref_train, cur_drifted, "ref_vs_drifted")

    # ------------------------------------------------------------------
    # 2) Score-level PSI/KL: model default-probability distribution
    # ------------------------------------------------------------------
    def score_of(df: pd.DataFrame) -> np.ndarray:
        X = scaler.transform(df[FEATURES].values)
        return model.predict_proba(X)[:, 1]  # P(bad)

    ref_score_train = score_of(ref_train)
    ref_score_val = score_of(ref_val)
    cur_score_initial = score_of(cur_initial)
    cur_score_drifted = score_of(cur_drifted)

    # Score reference uses the validation set (samples not used in training),
    # closer to the production monitoring convention
    score_report = build_score_report(
        ref_score_val,
        {"current_initial": cur_score_initial, "current_drifted": cur_score_drifted},
    )

    # ------------------------------------------------------------------
    # 3) Performance monitoring: did AUC degrade? (Model Degradation)
    # ------------------------------------------------------------------
    perf_rows = []
    for name, df in [("ref_val", ref_val), ("current_initial", cur_initial), ("current_drifted", cur_drifted)]:
        s = score_of(df)
        perf_rows.append({
            "Dataset": name,
            "Size": len(df),
            "BadRate": round(float(df[TARGET].mean()), 4),
            "AUC": round(auc_score(df[TARGET].values, s), 4),
            "MeanScoreP_bad": round(float(s.mean()), 4),
        })
    perf = pd.DataFrame(perf_rows)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    feat_report = pd.concat([feat_initial, feat_drifted], ignore_index=True)
    feat_report.to_csv(os.path.join(OUT_DIR, "psi_kld_feature_report.csv"), index=False, encoding="utf-8-sig")
    score_report.to_csv(os.path.join(OUT_DIR, "psi_kld_score_report.csv"), index=False, encoding="utf-8-sig")
    perf.to_csv(os.path.join(OUT_DIR, "performance_report.csv"), index=False, encoding="utf-8-sig")

    # JSON summary (consumed by 05 report / alerts)
    summary = {
        "feature_report": feat_report.to_dict(orient="records"),
        "score_report": score_report.to_dict(orient="records"),
        "performance_report": perf.to_dict(orient="records"),
    }
    with open(os.path.join(OUT_DIR, "monitoring_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n===== Score-level drift (core monitoring metric) =====")
    print(score_report.to_string(index=False))
    print("\n===== Model performance monitoring =====")
    print(perf.to_string(index=False))
    print("\n===== Top5 drifted features (ref vs drifted) =====")
    top5 = feat_drifted.sort_values("PSI", ascending=False).head(5)
    print(top5[["Feature", "PSI", "PSI_Level", "KL_Divergence"]].to_string(index=False))
    print(f"\n[{now_str()}] Metrics computed -> {OUT_DIR}")


if __name__ == "__main__":
    main()

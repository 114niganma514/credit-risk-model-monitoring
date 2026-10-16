# -*- coding: utf-8 -*-
"""
09_train_gbdt.py - Train GBDT baselines (LightGBM / XGBoost)
=============================================================
Extension 4a: replace the logistic-regression baseline with gradient boosting
decision trees (GBDT) and compare the discrimination of three models (AUC/KS):

    - Logistic Regression - original baseline, interpretable, regulator-friendly
    - LightGBM - mainstream risk GBDT, fast and accurate
    - XGBoost - classic GBDT

Key points (glossary):
    - Class Imbalance: bad samples are only ~6.7%; compensate with
      scale_pos_weight / class_weight (negatives / positives)
    - Tree models need no feature standardization (insensitive to feature
      scale); train on raw features directly
    - Early Stopping: monitor AUC on the validation set to prevent
      overfitting (native in LGB; XGB uses eval_set + early_stopping_rounds)

Outputs (under output/gbdt/):
    lgb_model.joblib / xgb_model.joblib
    model_comparison.csv     three-model AUC/KS comparison
    model_comparison.png     comparison bar chart
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
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ensure_dir, now_str, setup_utf8

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
TARGET = "SeriousDlqin2yrs"


def auc_score(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(auc(fpr, tpr))


def ks_statistic(y_true, y_prob) -> float:
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    good = df[df.y == 0]["p"]
    bad = df[df.y == 1]["p"]
    grid = np.linspace(0, 1, 1000)
    return float(np.max(np.abs(np.array([(good <= g).mean() for g in grid]) -
                               np.array([(bad <= g).mean() for g in grid]))))


def main():
    ensure_dir(GBDT_DIR)

    print(f"[{now_str()}] Loading reference-period data")
    train = pd.read_csv(os.path.join(PROC_DIR, "reference_train.csv"), encoding="utf-8-sig")
    val = pd.read_csv(os.path.join(PROC_DIR, "reference_val.csv"), encoding="utf-8-sig")

    X_tr, y_tr = train[FEATURES], train[TARGET]
    X_va, y_va = val[FEATURES], val[TARGET]

    # Imbalance compensation weight: negatives / positives ~= 14
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    sw = neg / pos
    print(f"    Train size: {len(train):,} | bad rate {y_tr.mean():.4%} | scale_pos_weight ~= {sw:.1f}")

    # ------------------------------------------------------------------
    # 1) Logistic Regression (original baseline; needs standardization)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_tr_lr = scaler.fit_transform(X_tr)
    X_va_lr = scaler.transform(X_va)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_tr_lr, y_tr)
    lr_prob = lr.predict_proba(X_va_lr)[:, 1]
    lr_res = {"Model": "Logistic Regression", "AUC": round(auc_score(y_va, lr_prob), 4),
              "KS": round(ks_statistic(y_va, lr_prob), 4)}

    # ------------------------------------------------------------------
    # 2) LightGBM (tree model: no standardization; early stopping)
    # ------------------------------------------------------------------
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=5, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=sw, random_state=42, verbose=-1,
    )
    lgb_model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
                  eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_prob = lgb_model.predict_proba(X_va)[:, 1]
    lgb_res = {"Model": "LightGBM", "AUC": round(auc_score(y_va, lgb_prob), 4),
               "KS": round(ks_statistic(y_va, lgb_prob), 4)}

    # ------------------------------------------------------------------
    # 3) XGBoost
    # ------------------------------------------------------------------
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=sw, random_state=42, eval_metric="auc",
        early_stopping_rounds=50, verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_prob = xgb_model.predict_proba(X_va)[:, 1]
    xgb_res = {"Model": "XGBoost", "AUC": round(auc_score(y_va, xgb_prob), 4),
               "KS": round(ks_statistic(y_va, xgb_prob), 4)}

    # ------------------------------------------------------------------
    # Aggregate and persist
    # ------------------------------------------------------------------
    comp = pd.DataFrame([lr_res, lgb_res, xgb_res])
    comp.to_csv(os.path.join(GBDT_DIR, "model_comparison.csv"), index=False, encoding="utf-8-sig")
    joblib.dump(lgb_model, os.path.join(GBDT_DIR, "lgb_model.joblib"))
    joblib.dump(xgb_model, os.path.join(GBDT_DIR, "xgb_model.joblib"))
    joblib.dump(scaler, os.path.join(GBDT_DIR, "lr_scaler.joblib"))  # keep LR scaler as backup

    # Comparison chart
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xpos = np.arange(len(comp))
    w = 0.32
    ax.bar(xpos - w / 2, comp["AUC"], width=w, label="AUC", color="#4C72B0")
    ax.bar(xpos + w / 2, comp["KS"], width=w, label="KS", color="#8BC8EA")
    for i in xpos:
        ax.text(i - w / 2, comp["AUC"][i] + 0.005, f"{comp['AUC'][i]:.4f}", ha="center", fontsize=9)
        ax.text(i + w / 2, comp["KS"][i] + 0.005, f"{comp['KS'][i]:.4f}", ha="center", fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(comp["Model"], fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Metric")
    ax.set_title("Baseline Model Comparison (AUC / KS)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(GBDT_DIR, "model_comparison.png"), dpi=120)
    plt.close(fig)

    print("\n===== Model comparison =====")
    print(comp.to_string(index=False))
    best = comp.loc[comp["AUC"].idxmax(), "Model"]
    print(f"\nBest model: {best} (SHAP attribution later is demonstrated on LightGBM)")
    print(f"[{now_str()}] Done -> {GBDT_DIR}")


if __name__ == "__main__":
    main()

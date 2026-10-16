# -*- coding: utf-8 -*-
"""
02_train_baseline.py - Train and save the baseline model
=========================================================
Business context:
    Before launch, train an interpretable baseline model on reference data.
    Logistic Regression is chosen - the standard risk-industry baseline:
    interpretable, stable and regulator-friendly.

Outputs (under output/model/):
    baseline_model.joblib   trained logistic regression model
    scaler.joblib           StandardScaler fitted on training features
    baseline_metrics.json   initial model performance (AUC / KS / confusion)

Glossary:
- Logistic Regression
- StandardScaler (rescales features to mean 0, variance 1)
- AUC (Area Under the ROC Curve): discrimination metric
- KS statistic (Kolmogorov-Smirnov): common risk-industry discrimination
  metric (max gap between good / bad score CDFs)
- Confusion Matrix (TP/FP/TN/FN)
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ensure_dir, now_str, setup_utf8

setup_utf8()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "output", "model")

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


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    KS statistic: maximum gap between the cumulative distributions of the
    good and bad groups. Risk-industry convention: KS > 0.3 indicates a
    reasonably discriminative model.
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    good = df[df.y == 0]["p"]
    bad = df[df.y == 1]["p"]
    grid = np.linspace(0, 1, 1000)
    cdf_good = np.array([(good <= g).mean() for g in grid])
    cdf_bad = np.array([(bad <= g).mean() for g in grid])
    return float(np.max(np.abs(cdf_good - cdf_bad)))


def main():
    ensure_dir(MODEL_DIR)

    print(f"[{now_str()}] Loading reference train / validation data")
    train = pd.read_csv(os.path.join(PROC_DIR, "reference_train.csv"), encoding="utf-8-sig")
    val = pd.read_csv(os.path.join(PROC_DIR, "reference_val.csv"), encoding="utf-8-sig")

    X_train, y_train = train[FEATURES].values, train[TARGET].values
    X_val, y_val = val[FEATURES].values, val[TARGET].values

    # 1) Standardization: logistic regression is scale-sensitive
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)   # fit on train
    X_val_s = scaler.transform(X_val)           # transform val with the same params

    # 2) Train logistic regression
    #    class_weight='balanced': compensates class imbalance (bad ~6.7%)
    #    max_iter raised to guarantee convergence
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )
    print(f"[{now_str()}] Training Logistic Regression ...")
    model.fit(X_train_s, y_train)

    # 3) Evaluate on the validation set
    y_prob = model.predict_proba(X_val_s)[:, 1]  # P(bad | features)
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc_val = auc(fpr, tpr)
    ks_val = ks_statistic(y_val, y_prob)

    # Confusion matrix at decision threshold 0.5 (production usually tunes
    # the threshold to business cost)
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    metrics = {
        "auc": round(auc_val, 4),
        "ks": round(ks_val, 4),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "bad_rate_val": float(y_val.mean()),
    }
    print(f"    AUC = {metrics['auc']:.4f} | KS = {metrics['ks']:.4f}")
    print(f"    Confusion (threshold 0.5): TP={tp} FP={fp} FN={fn} TN={tn}")

    # 4) Save model and scaler
    joblib.dump(model, os.path.join(MODEL_DIR, "baseline_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "baseline_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[{now_str()}] Baseline model saved -> {MODEL_DIR}")


if __name__ == "__main__":
    main()

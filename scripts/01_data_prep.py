# -*- coding: utf-8 -*-
"""
01_data_prep.py - Data preparation and reference / current period split
========================================================================
Business context (Model Lifecycle):
    A consumer finance company deployed a "credit default prediction model".
    - Reference Period: historical data used to develop the model
      (here: first 80% of rows, simulating chronological order)
    - Current Period: new data flowing in after launch
      (here: last 20% of rows)

Outputs (under data/processed/):
    reference_train.csv   reference period - training set (baseline model)
    reference_val.csv     reference period - validation set (initial perf.)
    current_raw.csv       current period - raw new data (later split into
                          an initial normal segment + a drifted segment)

Glossary:
- Missing Value
- Median Imputation
- Outlier / Winsorization
- Class Imbalance
"""
import os
import sys

import numpy as np
import pandas as pd

# Allow importing utils directly from the scripts directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ensure_dir, now_str, setup_utf8

setup_utf8()

# ---------------------------------------------------------------------------
# Path configuration (rooted at the project directory)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")

RAW_CSV = os.path.join(RAW_DIR, "cs-training.csv")
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",  # ratio of used to total credit limit
    "age",                                   # age in years
    "NumberOfTime30-59DaysPastDueNotWorse",  # times 30-59 days past due (2y)
    "DebtRatio",                             # monthly debt / monthly income
    "MonthlyIncome",                         # monthly income (USD)
    "NumberOfOpenCreditLinesAndLoans",       # open credit lines / loans
    "NumberOfTimes90DaysLate",               # times 90+ days past due (2y)
    "NumberRealEstateLoansOrLines",          # real-estate loans / lines
    "NumberOfTime60-89DaysPastDueNotWorse",  # times 60-89 days past due (2y)
    "NumberOfDependents",                    # number of dependents
]
TARGET = "SeriousDlqin2yrs"  # seriously delinquent within 2 years (1=bad, 0=good)


def load_and_clean() -> pd.DataFrame:
    """Load the raw data and apply basic cleaning."""
    print(f"[{now_str()}] Loading raw data: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")

    # 1) Drop the meaningless index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 2) Missing value imputation
    #    MonthlyIncome -> median imputation (median is robust to outliers)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    #    NumberOfDependents -> fill with 0 (business-wise: no dependents)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # 3) Drop impossible delinquency counts (96/98 exist in the raw data)
    for col in [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60-89DaysPastDueNotWorse",
    ]:
        n_before = len(df)
        df = df[df[col] <= 80]
        if len(df) < n_before:
            print(f"    Dropped {n_before - len(df)} rows in {col} (impossible values)")

    # 4) Winsorization for heavy-tailed features: clip values above the 99th
    #    percentile so extreme values do not stretch the bin edges and distort PSI
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=q99)

    # 5) Unify dtypes
    df[FEATURES] = df[FEATURES].astype(float)
    df[TARGET] = df[TARGET].astype(int)

    print(f"    Cleaned size: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"    Bad rate: {df[TARGET].mean():.4%}")
    return df


def split_periods(df: pd.DataFrame, ref_ratio: float = 0.8):
    """
    Split into reference / current periods by row order.
    Note: the raw data has no timestamps; row order simulates chronology -
    the first ref_ratio is treated as development data, the rest as
    post-launch data.
    """
    n = len(df)
    n_ref = int(n * ref_ratio)
    reference = df.iloc[:n_ref].copy()
    current = df.iloc[n_ref:].copy()
    print(f"    Reference: {len(reference):,} rows | Current: {len(current):,} rows")
    return reference, current


def main():
    ensure_dir(PROC_DIR)

    df = load_and_clean()

    # Reference / current period split
    reference, current = split_periods(df, ref_ratio=0.8)

    # Split reference into train / validation (7:3) for baseline development
    n_train = int(len(reference) * 0.7)
    reference_train = reference.iloc[:n_train].copy()
    reference_val = reference.iloc[n_train:].copy()

    # Persist (utf-8-sig so Excel can open directly)
    reference_train.to_csv(os.path.join(PROC_DIR, "reference_train.csv"), index=False, encoding="utf-8-sig")
    reference_val.to_csv(os.path.join(PROC_DIR, "reference_val.csv"), index=False, encoding="utf-8-sig")
    current.to_csv(os.path.join(PROC_DIR, "current_raw.csv"), index=False, encoding="utf-8-sig")

    print(f"[{now_str()}] Data preparation done -> {PROC_DIR}")
    print("    Output files:")
    for f in ["reference_train.csv", "reference_val.csv", "current_raw.csv"]:
        print(f"      - {f}")


if __name__ == "__main__":
    main()

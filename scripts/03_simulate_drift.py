# -*- coding: utf-8 -*-
"""
03_simulate_drift.py - Simulate post-launch data drift
=======================================================
Business context:
    After launch, new data flows in as two batches:
    - Early post-launch (first 3 months): borrower mix roughly matches the
      development period -> current_initial (expected: no / slight drift)
    - From month 4 (economic downturn): borrower quality changes
      -> current_drifted (expected: clear drift)

Injection (artificial drift signals, to demo how monitoring "catches" drift):
    1) Income shift: MonthlyIncome x 0.6 (overall income drop)
    2) Utilization rise: RevolvingUtilizationOfUnsecuredLines x 1.8 + 0.05
       (borrowers are more cash-strapped)
    3) Delinquency worsening: NumberOfTimes90DaysLate + 2
    4) Age structure shift: age + 3

Glossary:
- Covariate Shift: P(X) changes (feature distribution) while P(Y|X) is fixed
- Concept Drift: P(Y|X) changes (feature-label relationship)
- Prior Probability Shift: P(Y) changes (overall default rate)
This script mainly simulates Covariate Shift (+ a bit of prior shift).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import now_str, setup_utf8

setup_utf8()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")


def apply_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Inject artificial drift transformations (operate on a copy; do not
    pollute the raw data). Per-row magnitude carries random noise to mimic
    real-world individual variation.
    """
    d = df.copy()

    # 1) Income drop: x 0.55~0.65 random coefficient
    d["MonthlyIncome"] = d["MonthlyIncome"] * rng.uniform(0.55, 0.65, size=len(d))

    # 2) Utilization rise: x 1.7~1.9 plus 0.03~0.07
    d["RevolvingUtilizationOfUnsecuredLines"] = (
        d["RevolvingUtilizationOfUnsecuredLines"] * rng.uniform(1.7, 1.9, size=len(d))
        + rng.uniform(0.03, 0.07, size=len(d))
    )

    # 3) 90+ days past due: +1~3
    d["NumberOfTimes90DaysLate"] = d["NumberOfTimes90DaysLate"] + rng.integers(1, 4, size=len(d))

    # 4) Age shift right: +2~4 years
    d["age"] = d["age"] + rng.integers(2, 5, size=len(d))

    # Lower-bound protection for transformed values
    d["MonthlyIncome"] = d["MonthlyIncome"].clip(lower=100)
    d["RevolvingUtilizationOfUnsecuredLines"] = d["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0, upper=1.2)
    d["age"] = d["age"].clip(lower=18)
    return d


def main():
    print(f"[{now_str()}] Loading current-period raw data")
    current_raw = pd.read_csv(os.path.join(PROC_DIR, "current_raw.csv"), encoding="utf-8-sig")

    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    # Split 50/50 into "early post-launch" and "drifted period"
    n = len(current_raw)
    n_initial = int(n * 0.5)
    idx = rng.permutation(n)               # shuffle indices
    initial_idx = idx[:n_initial]
    drifted_idx = idx[n_initial:]

    current_initial = current_raw.iloc[initial_idx].copy()
    current_drifted_raw = current_raw.iloc[drifted_idx].copy()

    # Inject drift into the drifted-period data
    current_drifted = apply_drift(current_drifted_raw, rng)

    current_initial.to_csv(os.path.join(PROC_DIR, "current_initial.csv"), index=False, encoding="utf-8-sig")
    current_drifted.to_csv(os.path.join(PROC_DIR, "current_drifted.csv"), index=False, encoding="utf-8-sig")

    print(f"    Early post-launch (normal): {len(current_initial):,} rows")
    print(f"    Drifted (injected): {len(current_drifted):,} rows")

    # Print distribution comparison to confirm drift was injected
    ref = pd.read_csv(os.path.join(PROC_DIR, "reference_train.csv"), encoding="utf-8-sig")
    for col in ["MonthlyIncome", "RevolvingUtilizationOfUnsecuredLines", "NumberOfTimes90DaysLate", "age"]:
        print(f"    {col}: ref median={ref[col].median():.2f} | "
              f"initial median={current_initial[col].median():.2f} | "
              f"drifted median={current_drifted[col].median():.2f}")

    print(f"[{now_str()}] Drift simulation done -> {PROC_DIR}")


if __name__ == "__main__":
    main()

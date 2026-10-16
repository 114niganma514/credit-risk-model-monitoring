# -*- coding: utf-8 -*-
"""
utils.py - Shared utility module for the Model Monitoring project
=================================================================
Public functions for the PSI / KL drift-detection framework:

1. compute_psi():  Population Stability Index - core metric for
                   feature / score distribution drift
2. compute_kld():  Kullback-Leibler Divergence (relative entropy) -
                   information-theoretic view of distribution shift
3. classify_psi(): Map a PSI value to NoDrift / ModerateDrift / SevereDrift
                   using industry-standard thresholds
4. ensure_dir():   Recursively create a directory if missing
5. setup_utf8():   Force UTF-8 on the Windows console to avoid mojibake

Glossary (English terms kept throughout the codebase):
- Reference Distribution: feature / score distribution at model development
  time (pre-launch); used as the "expected" distribution E
- Current Distribution: distribution of new incoming data after launch;
  used as the "actual" distribution A
- Equal-frequency Binning: bin edges from quantiles so each bin holds a
  roughly equal share of samples
- Epsilon Smoothing: tiny constant added to bin proportions to avoid
  division-by-zero / log(0) when a bin is empty
- Drift: a distribution changes over time
"""
import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 0. Bin-edge construction (shared by PSI / KL)
# ---------------------------------------------------------------------------
def _build_edges(reference: np.ndarray, bins: int = 10) -> np.ndarray:
    """
    Build bin edges for PSI / KL computation.

    Two strategies (auto-selected):
      1) Sparse / discrete features (unique values <= bins * 3): bin by
         UNIQUE VALUES. Equal-frequency binning fails on such features -
         many duplicated values collapse quantile edges (e.g. 95% of samples
         are 0, so the 0th-90th percentiles are all 0), degenerating the
         bins and distorting PSI.
      2) Continuous features: EQUAL-FREQUENCY binning.

    Returns:
        Strictly increasing edge array (open ends: -inf / +inf)
    """
    uniq = np.unique(reference)
    if len(uniq) <= bins * 3:
        # Discrete feature: one bin per unique value (when count is manageable)
        return np.concatenate([[-np.inf], uniq.astype(float), [np.inf]])

    # Continuous feature: quantile-based equal-frequency edges from reference
    percentiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(reference, percentiles)
    edges = np.unique(edges)          # dedupe to avoid degenerate edges
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


# ---------------------------------------------------------------------------
# 1. PSI (Population Stability Index)
# ---------------------------------------------------------------------------
def compute_psi(reference: np.ndarray,
                current: np.ndarray,
                bins: int = 10,
                epsilon: float = 1e-6) -> dict:
    """
    Compute the Population Stability Index.

    Formula: PSI = sum_i (A_i - E_i) * ln(A_i / E_i)

    Where:
        E_i = share of the reference (expected) distribution in bin i
        A_i = share of the current (actual) distribution in bin i

    Intuition:
        - If both distributions are identical, every bin has A_i == E_i,
          ln(A_i/E_i) = 0, hence PSI = 0
        - The larger the distribution gap, the larger the PSI

    Binning strategy (critical):
        1) Compute 10 quantile edges on the REFERENCE distribution
           (equal-frequency binning)
        2) Apply the SAME edges to the current distribution
        - Both distributions must be measured on the same scale, otherwise
          PSI is meaningless.

    Parameters:
        reference: reference distribution (1-D array)
        current:   current distribution (1-D array)
        bins:      number of bins (default 10)
        epsilon:   smoothing term; prevents log(0) when a bin share is 0

    Returns:
        dict: psi value, per-bin shares, bin edges
    """
    reference = np.asarray(reference, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()

    # 1) Bin edges: unique values for discrete, quantiles for continuous
    edges = _build_edges(reference, bins=bins)

    # 2) Sample counts per bin for both distributions
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)

    # 3) Convert to shares and apply epsilon smoothing
    ref_pct = ref_hist / ref_hist.sum() + epsilon
    cur_pct = cur_hist / cur_hist.sum() + epsilon

    # 4) Accumulate the PSI formula bin by bin
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    return {
        "psi": psi,
        "ref_pct": ref_pct,
        "cur_pct": cur_pct,
        "edges": edges,
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# 2. KL Divergence (Kullback-Leibler Divergence / relative entropy)
# ---------------------------------------------------------------------------
def compute_kld(reference: np.ndarray,
                current: np.ndarray,
                bins: int = 10,
                epsilon: float = 1e-6) -> dict:
    """
    Compute the Kullback-Leibler Divergence (relative entropy).

    Formula (discrete form):
        KL(P || Q) = sum_i P(i) * ln( P(i) / Q(i) )

    Direction note:
        - Here P = reference, Q = current, i.e. KL(reference || current)
        - KL is NOT symmetric: KL(P||Q) != KL(Q||P); fix the direction
          when interpreting results
        - Relation to PSI: PSI is essentially a symmetrized variant of KL;
          roughly PSI = KL(current||reference) + KL(reference||current) scaled
          (industry approximation)

    Binning strategy:
        Use quantiles of BOTH distributions when building edges, so that a
        narrow reference range does not push most current samples into the
        open-ended first / last bins, which would distort the metric.

    Parameters:
        reference: reference distribution (1-D array)
        current:   current distribution (1-D array)
        bins:      number of bins (default 10)
        epsilon:   smoothing term

    Returns:
        dict: kld value, both distributions' probabilities, bin edges
    """
    reference = np.asarray(reference, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()

    # Unified edges: unique values for discrete; merged quantiles otherwise
    unique_ref = np.unique(reference)
    if len(unique_ref) <= bins * 3:
        edges = np.concatenate([[-np.inf], unique_ref.astype(float), [np.inf]])
    else:
        combined = np.concatenate([reference, current])
        percentiles = np.linspace(0, 100, bins + 1)
        edges = np.percentile(combined, percentiles)
        edges = np.unique(edges)
        edges[0], edges[-1] = -np.inf, np.inf

    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)

    ref_p = ref_hist / ref_hist.sum() + epsilon
    cur_p = cur_hist / cur_hist.sum() + epsilon

    # KL(P||Q), P = reference
    kld = float(np.sum(ref_p * np.log(ref_p / cur_p)))

    return {
        "kld": kld,
        "ref_p": ref_p,
        "cur_p": cur_p,
        "edges": edges,
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# 3. PSI classification (industry-standard thresholds)
# ---------------------------------------------------------------------------
def classify_psi(psi: float) -> str:
    """
    Classify a PSI value using industry-standard thresholds (used for alerts):

        PSI < 0.10          -> No Significant Drift   (GREEN)
        0.10 <= PSI < 0.25  -> Moderate Drift         (YELLOW)
        PSI >= 0.25         -> Severe Drift           (RED)

    Note: these are empirical values (commonly 0.1 / 0.25 in the industry);
    real deployments should calibrate them against business tolerance and
    the cost of false positives / false negatives.
    """
    if psi < 0.10:
        return "NoDrift"
    elif psi < 0.25:
        return "ModerateDrift"
    else:
        return "SevereDrift"


# ---------------------------------------------------------------------------
# 4. Utility functions
# ---------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """Ensure a directory exists (create recursively if missing)."""
    os.makedirs(path, exist_ok=True)


def setup_utf8() -> None:
    """Force UTF-8 on the Windows console to avoid mojibake in output."""
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def save_json(obj: dict, path: str) -> None:
    """Save a dict as JSON (UTF-8)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def now_str() -> str:
    """Current timestamp string, used in logs / filenames."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_csv(path: str) -> pd.DataFrame:
    """Read a CSV (utf-8-sig tolerant of the BOM added by Excel)."""
    return pd.read_csv(path, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 5. Rolling-window generation (shared by extensions 1 / 2):
#    progressive drift simulation
# ---------------------------------------------------------------------------
DRIFT_FEATURES = [
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


def apply_drift_intensity(df: pd.DataFrame, intensity: float, rng: np.random.Generator,
                          concept_flip: float = 0.0) -> pd.DataFrame:
    """
    Inject artificial drift into data according to a drift intensity (0~1).
    intensity=0 returns the data unchanged; intensity=1 approximates the
    fixed drift used by 03_simulate_drift.

    Two kinds of drift are injected (mimicking an economic downturn):
        [Covariate Shift] - feature distributions change:
            - MonthlyIncome declines: x (1 - 0.42*intensity)
            - RevolvingUtilization rises: x(1+0.75*intensity) + 0.08*intensity
            - NumberOfTimes90DaysLate increases: + round(2.5*intensity)
            - age shifts right: + round(3*intensity)
        [Concept Drift] - feature-label relationship changes:
            - concept_flip: randomly flips a proportion of "good" customers
              (y=0) to "bad" (y=1). Business meaning: in a downturn, some
              customers with otherwise normal features actually default;
              for the same X, P(Y=1|X) rises -> P(Y|X) changes.
            - The flip only affects AUC evaluation; feature / score
              distributions (PSI) stay unpolluted.
    """
    d = df.copy()
    i = float(intensity)

    # ---- Covariate shift ----
    d["MonthlyIncome"] = d["MonthlyIncome"] * rng.uniform(1 - 0.42 * i - 0.03, 1 - 0.42 * i + 0.03, size=len(d))
    d["RevolvingUtilizationOfUnsecuredLines"] = (
        d["RevolvingUtilizationOfUnsecuredLines"] * (1 + 0.75 * i) + 0.08 * i
    )
    d["NumberOfTimes90DaysLate"] = d["NumberOfTimes90DaysLate"] + int(round(2.5 * i))
    d["age"] = d["age"] + int(round(3 * i))
    d["MonthlyIncome"] = d["MonthlyIncome"].clip(lower=100)
    d["RevolvingUtilizationOfUnsecuredLines"] = d["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0, upper=1.2)
    d["age"] = d["age"].clip(lower=18)

    # ---- Concept drift (label flip) ----
    if concept_flip > 0:
        good_idx = d.index[d["SeriousDlqin2yrs"] == 0].to_numpy()
        n_flip = int(round(len(good_idx) * concept_flip))
        if n_flip > 0:
            flip_idx = rng.choice(good_idx, size=n_flip, replace=False)
            d.loc[flip_idx, "SeriousDlqin2yrs"] = 1
    return d


def build_rolling_windows(current_raw: pd.DataFrame,
                          n_windows: int = 8,
                          drift_schedule: list = None,
                          start_date: str = "2020-01-01",
                          period_days: int = 30,
                          rng_seed: int = 42) -> list:
    """
    Build rolling-window data:
        split the current-period data row-wise into n_windows segments
        (each segment simulates one monitoring period: week / month),
        then inject a different drift intensity per segment according to
        drift_schedule (default: normal early, progressively worsening later).

    Parameters:
        current_raw:    current-period raw data (data/processed/current_raw.csv)
        n_windows:      number of windows (default 8; think 8 months / 8 weeks)
        drift_schedule: per-window drift intensities (0~1); length must equal
                        n_windows; default [0,0,0,0, 0.25, 0.5, 0.75, 1.0] -
                        first 4 windows normal, deterioration from window 5
        start_date:     date of the first window (x-axis of the trend chart)
        period_days:    days spanned by each window (month=30, week=7)
        rng_seed:       random seed for reproducibility

    Returns:
        windows: list[dict]; each item has name / date / df / intensity /
                 concept_flip
    """
    if drift_schedule is None:
        drift_schedule = [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(drift_schedule) == n_windows, "drift_schedule length must equal n_windows"

    rng = np.random.default_rng(rng_seed)
    n = len(current_raw)
    per_window = n // n_windows

    windows = []
    start = pd.Timestamp(start_date)
    for k in range(n_windows):
        seg = current_raw.iloc[k * per_window:(k + 1) * per_window].copy()
        intensity = drift_schedule[k]
        # Concept-drift flip ratio grows with intensity (max 15% good -> bad)
        concept_flip = round(0.15 * intensity, 4)
        if intensity > 0 or concept_flip > 0:
            seg = apply_drift_intensity(seg, intensity, rng, concept_flip=concept_flip)
        windows.append({
            "name": f"W{k + 1}",
            "date": (start + pd.Timedelta(days=period_days * k)).strftime("%Y-%m-%d"),
            "df": seg.reset_index(drop=True),
            "intensity": intensity,
            "concept_flip": concept_flip,
        })
    return windows


def save_rolling_windows(windows: list, out_dir: str) -> None:
    """Save rolling windows to CSV + meta.json (reused by extensions 1 / 2)."""
    ensure_dir(out_dir)
    meta = []
    for w in windows:
        path = os.path.join(out_dir, f"{w['name'].lower()}.csv")
        w["df"].to_csv(path, index=False, encoding="utf-8-sig")
        meta.append({"name": w["name"], "date": w["date"], "intensity": w["intensity"],
                     "concept_flip": w["concept_flip"], "file": path})
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta

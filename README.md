# Credit Risk Model Monitoring & Drift Detection System

> Model Lifecycle Management (MLOps) - demonstrating what happens **after a model
> goes live**. This resume project monitors a deployed credit-default model with
> PSI / KL divergence, rolling-window trends, concept-drift detection, automatic
> alerts and SHAP attribution, building a complete
> **Monitor -> Alert -> Attribute -> Retrain** loop.

---

## 0. Resume Summary

| Item | Detail |
|---|---|
| **Project Name** | Credit Risk Model Monitoring & Drift Detection System |
| **Positioning** | Credit-risk model lifecycle management (Model Lifecycle / MLOps) |
| **Tech Stack** | Python - pandas - scikit-learn - LightGBM - XGBoost - SHAP - matplotlib |
| **Core Metrics** | PSI (Population Stability Index) - KL Divergence - AUC/KS - rolling-window trend - drift attribution |
| **Data** | Kaggle Give Me Some Credit (FICO credit-default dataset, 150K samples, 10 features) |
| **Key Results** | Built a Monitor->Alert->Attribute->Retrain loop: rolling PSI warned one period earlier; concept-drift AUC dropped 24% triggering a retraining review; SHAP attribution identified days-past-due-90 as the top drift channel |

## 1. One-Sentence Introduction

A consumer finance company deployed a credit-default prediction model. After
launch the borrower mix changes (income drops, delinquencies rise). This project
catches drift in time and attributes it, using **PSI (Population Stability
Index)**, **KL Divergence (Kullback-Leibler Divergence)**, **rolling-window
trends**, **AUC performance-drift detection**, **automatic alerts** and
**SHAP drift attribution** - then decides whether the model needs retraining.

## 2. Folder Structure

```
credit-risk-model-monitoring/
├── README.md
├── requirements.txt                 # dependencies (incl. lightgbm/xgboost/shap)
├── config/
│   └── alert_config.json            # alert config (Feishu webhook / SMTP / thresholds)
├── data/
│   ├── raw/cs-training.csv          # raw data (Kaggle: Give Me Some Credit, 150K rows)
│   └── processed/                   # cleaned splits + rolling windows (auto-generated)
│       ├── reference_train/val.csv  # reference train / validation sets
│       ├── current_raw/initial/drifted.csv  # current-period data
│       └── rolling_windows/         # 8 rolling windows (built by 06, reused by 07)
├── scripts/
│   ├── utils.py                     # PSI/KL engine + rolling-window generation
│   ├── 01_data_prep.py              # data cleaning + reference/current split
│   ├── 02_train_baseline.py         # logistic-regression baseline
│   ├── 03_simulate_drift.py         # fixed drift simulation (for snapshot monitoring)
│   ├── 04_compute_psi_kld.py        # feature-level / score-level PSI, KL
│   ├── 05_monitor_report.py         # snapshot monitoring report (HTML)
│   ├── 06_rolling_psi_trend.py      # *Ext.1 rolling-window PSI trend (early warning)
│   ├── 07_concept_drift_auc.py      # *Ext.2 concept-drift detection (rolling AUC)
│   ├── 08_alert_notification.py     # *Ext.3 Feishu / email alerts
│   ├── 09_train_gbdt.py             # *Ext.4a LightGBM/XGBoost baselines
│   └── 10_shap_drift_attribution.py # *Ext.4b SHAP drift attribution
└── output/
    ├── model/                       # logistic-regression baseline + scaler
    ├── gbdt/                        # LGB/XGB models + SHAP attribution report & charts
    ├── rolling/                     # rolling PSI/AUC trend CSV + charts + warning conclusions
    ├── charts/                      # snapshot monitoring charts
    ├── monitoring_report.html       # snapshot monitoring report (open in a browser)
    ├── monitoring_summary.json      # alert summary
    ├── alert_log.json               # alert history (written by 08)
    └── *_report.csv                 # PSI/KL/performance detail
```

## 3. Quick Start

```bash
# Option A: one-click run (Windows: double-click run_all.bat, or from cmd)
run_all.bat

# Option B: run step by step
pip install -r requirements.txt
python scripts/01_data_prep.py
python scripts/02_train_baseline.py
python scripts/03_simulate_drift.py
python scripts/04_compute_psi_kld.py
python scripts/05_monitor_report.py         # snapshot monitoring report
python scripts/06_rolling_psi_trend.py      # Ext.1 rolling PSI trend
python scripts/07_concept_drift_auc.py      # Ext.2 concept-drift AUC
python scripts/08_alert_notification.py     # Ext.3 alerts (simulate by default)
python scripts/09_train_gbdt.py             # Ext.4a GBDT baselines
python scripts/10_shap_drift_attribution.py # Ext.4b SHAP attribution

# 3. Open the report
# Browser: output/monitoring_report.html; charts in output/charts, output/rolling, output/gbdt
```

## 4. Extensions in Detail

### Extension 1: Rolling-window PSI trend (Early Warning)

`06_rolling_psi_trend.py` splits the current-period data into **8 rolling
windows** (simulating monthly monitoring, 2020-01 ~ 2020-07). The first 4
windows are normal; from window 5 the drift intensity ramps up. Each window is
compared against the reference to compute the **model-score PSI**, plotted as a
trend line with **0.10 / 0.25 threshold lines**:

- Values: W1-W4 score PSI < 0.01 (normal) -> from W5 it jumps to 1.94 (severe
  alert) -> W8 reaches 8.55
- Value: **a trend line exposes the risk earlier than a single snapshot**
  (the first severe alert fires in period 5)
- Outputs: `output/rolling/rolling_psi_trend.csv`, `psi_trend.png`,
  `early_warning.json`

### Extension 2: Concept-drift detection (rolling-window AUC)

`07_concept_drift_auc.py` computes the **true AUC** on the same rolling
windows to detect performance drift:

- Three drift types (glossary):
  - **Covariate Shift**: P(X) changes (feature distributions; reflected by PSI)
  - **Concept Drift**: P(Y|X) changes (feature-label relationship; reflected by
    AUC decline)
  - **Prior Probability Shift**: P(Y) changes (overall default rate)
- Demo: from W5 a label flip is injected (some good customers actually
  default); AUC drops from 0.87 to 0.62, default rate rises 6% -> 21%,
  **cross-validated with the PSI trend**, triggering a retraining suggestion
- Outputs: `output/rolling/rolling_auc_trend.csv`, `auc_trend.png`,
  `concept_drift_summary.json`

### Extension 3: Automatic alerts when PSI exceeds thresholds (Feishu / email)

`08_alert_notification.py` aggregates the latest monitoring results and sends
an alert when thresholds are exceeded:

- Trigger rules (tunable in `config/alert_config.json`): score PSI >= 0.10 /
  0.25; AUC drop >= 3%; severe-drift features >= 1
- Channels:
  - Feishu custom-bot Webhook (fill `feishu.webhook_url` in
    `config/alert_config.json`)
  - SMTP email (fill the `email` section; QQ/163/Gmail use an authorization code)
  - **Simulate mode (default)**: only prints the alert card, sends nothing -
    handy for debugging
- Usage:
  ```bash
  python scripts/08_alert_notification.py                  # simulate
  python scripts/08_alert_notification.py --channel feishu # actually send to Feishu
  python scripts/08_alert_notification.py --channel all    # Feishu + email
  ```
- Scheduling (Windows Task Scheduler, daily 09:00 check):
  ```powershell
  schtasks /Create /TN "ModelMonitorAlert" /TR "python D:\Users\jiayi\credit-risk-model-monitoring\scripts\08_alert_notification.py --channel all" /SC DAILY /ST 09:00
  ```
- Output: `output/alert_log.json` (alert history)

### Extension 4: LightGBM/XGBoost baselines + SHAP drift attribution

`09_train_gbdt.py` trains **LightGBM / XGBoost** and compares them with
logistic regression (AUC/KS):

| Model | AUC | KS |
|---|---|---|
| Logistic Regression | 0.8560 | 0.5547 |
| LightGBM | 0.8530 | 0.5496 |
| **XGBoost** | **0.8633** | **0.5683** |

> Key points: tree models need no feature standardization; use
> `scale_pos_weight` (~ negatives/positives = 14.3) to compensate class
> imbalance; early stopping prevents overfitting.

`10_shap_drift_attribution.py` uses **SHAP (SHapley Additive exPlanations)** to
answer "which feature's drift affects the model output the most":

- Metric: **Drift Impact = normalized PSI x normalized drifted mean|SHAP|**
  (combining "drift magnitude" with "model sensitivity")
- Result: `NumberOfTimes90DaysLate` influence surges **12.5x**
  (mean|SHAP| 0.043 -> 0.576) with PSI = 16.2 -> highest priority
- Outputs: `output/gbdt/shap_drift_report.csv`, `shap_importance_shift.png`,
  `shap_beeswarm_drifted.png`

## 5. Core Concepts (glossary)

| Term | English (full name) | Plain explanation |
|---|---|---|
| Population Stability Index | PSI | Measures how much the reference and current distributions differ: PSI = sum((actual share - expected share) * ln(actual share / expected share)) |
| KL Divergence | KLD (Kullback-Leibler Divergence) / Relative Entropy | Information-theoretic distribution difference; not symmetric, fix the direction |
| Rolling Window | Rolling Window | Split time into weekly/monthly windows and compute metrics per window to form a trend |
| Covariate Shift | Covariate Shift | P(X) changes while P(Y\|X) is fixed (feature distributions change) |
| Concept Drift | Concept Drift | P(Y\|X) changes (feature-label relationship changes) |
| Prior Probability Shift | Prior Probability Shift | P(Y) changes (overall default rate) |
| Performance Drift | Performance Drift | Model AUC/KS metrics decay over time |
| Early Stopping | Early Stopping | Stop training when the validation metric stops improving - prevents overfitting |
| Class Imbalance | Class Imbalance | Bad samples are only ~6.7%; compensate with scale_pos_weight / class_weight |
| SHAP | SHapley Additive exPlanations | Shapley-value explanation: each feature's contribution to each sample's prediction |
| Discrimination | AUC / KS | The model's power to separate good and bad customers |

## 6. PSI Alert Thresholds (industry-standard empirical values)

| PSI range | Level | Suggested action |
|---|---|---|
| PSI < 0.10 | No Significant Drift | Normal - keep monitoring |
| 0.10 <= PSI < 0.25 | Moderate Drift | Investigate causes, increase attention |
| PSI >= 0.25 | Severe Drift | Trigger an alert, evaluate retraining |

## 7. Pitfalls & Error Cases

**Pitfall 1: an empty bin -> log(0) error**
```
Wrong: a feature's current distribution shifts entirely, one bin has 0 current
       samples, plugging ln(0/share) -> ValueError: divide by zero
Fixed: add a tiny epsilon smoothing term (1e-6) to the shares
       (built into utils.compute_psi), and keep the first/last edges open
       (-inf / +inf) to catch extreme values.
```

**Pitfall 2: equal-width binning instead of equal-frequency**
```
Wrong: equal-width 10 bins on a heavy-tailed feature (e.g. MonthlyIncome);
       most samples pile into bin 1 and PSI is distorted.
Fixed: equal-frequency binning (quantile-based), and "fix edges first, then
       apply": edges come from the reference, current is counted on them.
```

**Pitfall 2.5: equal-frequency binning on sparse count features -> distorted
PSI (actually hit in this project)**
```
Wrong: 96% of NumberOfTimes90DaysLate are 0; the 0th-90th percentiles are all
       0, edges collapse and bins degenerate: PSI = 0.000 while KL = 1.038
       (contradictory results).
Fixed: detect the unique-value count before binning; sparse/discrete features
       (unique <= 3 x bins) switch to "bin by unique values"
       (auto-switched in utils._build_edges).
```

**Pitfall 3: missing values fed into the model without handling**
```
Wrong: MonthlyIncome has ~5% missing; training directly ->
       ValueError: Input contains NaN
Fixed: median imputation, plus 99th-percentile winsorization for heavy tails.
```

**Pitfall 4: mixing up the KL direction**
```
Wrong: someone says "larger KL means worse drift", but KL(P||Q) != KL(Q||P).
Fixed: fix the direction KL(reference||current) and state it; PSI is
       symmetric, so its direction is unambiguous.
```

**Pitfall 5: pure covariate shift != AUC necessarily dropping (actually hit)**
```
Wrong: assuming "feature distributions changed, so performance must crash";
       after injecting covariate shift aligned with the model's judgment,
       AUC did not drop (even rose slightly).
Fixed: AUC is insensitive to a uniform score shift; to detect concept drift
       you must inject label-relationship changes (some good customers actually
       default given the same features), and also monitor P(Y) (default rate).
```

## 8. How to Extend (resume bonus points)

1. **Rolling monitoring -> alerts**: feed the 06/07 results straight into 08;
   thresholds trigger alerts (loop already closed).
2. **Add drift attribution**: SHAP already locates top features; drill down
   further with per-feature PSI and business interpretation.
3. **Add a retraining pipeline**: after a retraining review, automatically
   retrain on the latest window data and A/B-compare old vs new models.
4. **Add a monitoring dashboard**: push PSI/AUC trends into Grafana / Feishu
   Bitable for a live dashboard.
5. **Add multi-model attribution**: run SHAP on XGBoost too, comparing how
   different models react to drift.

## 9. Demo Results (run on this machine)

| Layer | Metric | Early post-launch | Drifted (W8) |
|---|---|---|---|
| Feature drift | Score PSI | < 0.01 (no drift) | 8.55 (severe) |
| Concept drift | AUC | 0.856 | 0.615 (-24%) |
| Prior shift | Default rate | 6.1% | 21.1% |
| Attribution | Top-priority features | - | days-past-due-90, utilization, income |
| Alert | Suggested action | - | trigger a retraining review |

> Numbers come from the actual run (see `output/rolling/`, `output/gbdt/`,
> `output/alert_log.json`).

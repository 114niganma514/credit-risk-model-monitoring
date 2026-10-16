# -*- coding: utf-8 -*-
"""
05_monitor_report.py - Generate monitoring charts and the HTML report
=====================================================================
Responsibilities:
    1. Generate 4 static charts with matplotlib (offline, embeddable):
       - score_distribution.png   model score distribution comparison
         (reference vs early post-launch vs drifted)
       - feature_drift_top.png    distribution comparison of the top
         drifted features
       - psi_bar.png              feature PSI bar chart for the drifted
         period (red / yellow / green by alert level)
       - auc_compare.png          model AUC degradation comparison
    2. Generate the HTML monitoring report (monitoring_report.html):
       - embedded ECharts interactive charts (precise data visualization)
       - falls back to embedded PNGs if the ECharts CDN fails to load
       - includes conclusion summary, alert levels and a glossary

Outputs (under output/): charts/*.png, monitoring_report.html,
monitoring_summary.json
"""
import base64
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import classify_psi, ensure_dir, now_str, setup_utf8

setup_utf8()

# English-only project: use the default DejaVu Sans family
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "output", "model")
OUT_DIR = os.path.join(ROOT, "output")
CHART_DIR = os.path.join(OUT_DIR, "charts")

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

# PSI alert colors (red / yellow / green, consistent with classify_psi)
COLOR_GREEN = "#2E9E5B"
COLOR_YELLOW = "#E8A33D"
COLOR_RED = "#D9534F"


def psi_color(psi: float) -> str:
    """Return the chart color for a PSI value by alert level."""
    if psi < 0.10:
        return COLOR_GREEN
    elif psi < 0.25:
        return COLOR_YELLOW
    return COLOR_RED


# ---------------------------------------------------------------------------
# 1. matplotlib static charts
# ---------------------------------------------------------------------------
def chart_score_distribution(ref_score, init_score, drift_score) -> str:
    """Model score distribution comparison chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 60)
    for data, label, color in [
        (ref_score, "Reference", "#4C72B0"),
        (init_score, "Initial", "#2E9E5B"),
        (drift_score, "Drifted", "#D9534F"),
    ]:
        ax.hist(data, bins=bins, alpha=0.45, density=True, label=label, color=color)
    ax.set_xlabel("Model score P(bad)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution Shift")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(CHART_DIR, "score_distribution.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def chart_feature_drift_top(ref_df, drift_df, top_feats) -> str:
    """Distribution comparison of the top drifted features (subplot grid)."""
    n = len(top_feats)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.2 * rows))
    axes = np.array(axes).ravel()
    for i, col in enumerate(top_feats):
        ax = axes[i]
        r = ref_df[col].values
        d = drift_df[col].values
        lo = min(np.percentile(r, 1), np.percentile(d, 1))
        hi = max(np.percentile(r, 99), np.percentile(d, 99))
        bins = np.linspace(lo, hi, 40)
        ax.hist(r, bins=bins, alpha=0.5, density=True, label="Reference", color="#4C72B0")
        ax.hist(d, bins=bins, alpha=0.5, density=True, label="Drifted", color="#D9534F")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Top Drifted Features - Distribution Shift", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(CHART_DIR, "feature_drift_top.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def chart_psi_bar(drift_feat_report) -> str:
    """Feature PSI bar chart for the drifted period (sorted, color-coded)."""
    df = drift_feat_report.sort_values("PSI", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [psi_color(v) for v in df["PSI"]]
    ax.barh(df["Feature"], df["PSI"], color=colors)
    ax.axvline(0.10, color="#E8A33D", ls="--", lw=1, label="Threshold 0.10 (Moderate)")
    ax.axvline(0.25, color="#D9534F", ls="--", lw=1, label="Threshold 0.25 (Severe)")
    ax.set_xlabel("PSI (Population Stability Index)")
    ax.set_title("Feature PSI Alerts in Drifted Period (Red=Severe / Yellow=Moderate / Green=Normal)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "psi_bar.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def chart_auc_compare(perf_df) -> str:
    """AUC degradation comparison chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = perf_df["Dataset"].tolist()
    aucs = perf_df["AUC"].tolist()
    colors = ["#4C72B0", "#2E9E5B", "#D9534F"]
    bars = ax.bar(names, aucs, color=colors, width=0.5)
    ax.axhline(aucs[0], color="#4C72B0", ls="--", lw=1)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title("Model Discrimination Monitoring (AUC Degradation)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(CHART_DIR, "auc_compare.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def img_to_base64(path: str) -> str:
    """Encode an image as base64 for HTML embedding fallback."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------------------------------------------------------------------------
# 2. ECharts options (following the doubao-visualization spec)
# ---------------------------------------------------------------------------
def echarts_score_psi(score_report) -> dict:
    """Score-level PSI comparison bar chart."""
    names = [r["Comparison"] for r in score_report]
    values = [r["PSI"] for r in score_report]
    return {
        "backgroundColor": "transparent",
        "title": {"text": "Model Score Distribution PSI", "left": "center",
                  "textStyle": {"color": "#1A1B1C", "fontSize": 15, "fontWeight": 600}},
        "tooltip": {"trigger": "axis", "triggerOn": "click", "renderMode": "richText",
                    "confine": True, "textStyle": {"fontSize": 10, "lineHeight": 14}, "padding": [6, 8]},
        "legend": {"top": 36, "itemWidth": 14, "itemHeight": 8, "textStyle": {"color": "#6B7280", "fontSize": 11}},
        "grid": {"left": 44, "right": 20, "top": 84, "bottom": 32, "containLabel": True},
        "xAxis": {"type": "category", "data": names, "axisLabel": {"color": "#555", "fontSize": 11, "hideOverlap": True}},
        "yAxis": {"type": "value", "name": "PSI", "axisLabel": {"color": "#555", "fontSize": 11}},
        "series": [{
            "name": "PSI", "type": "bar", "barWidth": "38%",
            "data": [{"value": v, "itemStyle": {"color": psi_color(v)}} for v in values],
            "label": {"show": True, "position": "top", "fontSize": 10,
                      "formatter": "function(p){if(p.value==null){return '';}return p.value.toFixed(4);}"},
        }],
    }


def echarts_feature_psi(feat_report) -> dict:
    """Top-10 feature PSI horizontal bar chart (ascending, color-coded)."""
    df = feat_report.sort_values("PSI", ascending=True).tail(10)
    cats = df["Feature"].tolist()
    vals = df["PSI"].tolist()
    return {
        "backgroundColor": "transparent",
        "title": {"text": "Top-10 Feature PSI (Drifted Period)", "left": "center",
                  "textStyle": {"color": "#1A1B1C", "fontSize": 15, "fontWeight": 600}},
        "tooltip": {"trigger": "axis", "triggerOn": "click", "renderMode": "richText",
                    "confine": True, "textStyle": {"fontSize": 10, "lineHeight": 14}, "padding": [6, 8]},
        "grid": {"left": 10, "right": 60, "top": 48, "bottom": 24, "containLabel": True},
        "xAxis": {"type": "value", "name": "PSI", "axisLabel": {"color": "#555", "fontSize": 11}},
        "yAxis": {"type": "category", "data": cats, "axisLabel": {"color": "#555", "fontSize": 10}},
        "series": [{
            "name": "PSI", "type": "bar", "barWidth": "60%",
            "data": [{"value": v, "itemStyle": {"color": psi_color(v)}} for v in vals],
            "label": {"show": True, "position": "right", "fontSize": 10,
                      "formatter": "function(p){if(p.value==null){return '';}return p.value.toFixed(4);}"},
        }],
    }


def echarts_auc(perf_df) -> dict:
    """AUC comparison bar chart."""
    names = perf_df["Dataset"].tolist()
    aucs = perf_df["AUC"].tolist()
    return {
        "backgroundColor": "transparent",
        "title": {"text": "Model AUC Performance Monitoring", "left": "center",
                  "textStyle": {"color": "#1A1B1C", "fontSize": 15, "fontWeight": 600}},
        "tooltip": {"trigger": "axis", "triggerOn": "click", "renderMode": "richText",
                    "confine": True, "textStyle": {"fontSize": 10, "lineHeight": 14}, "padding": [6, 8]},
        "grid": {"left": 44, "right": 20, "top": 84, "bottom": 32, "containLabel": True},
        "xAxis": {"type": "category", "data": names, "axisLabel": {"color": "#555", "fontSize": 11, "hideOverlap": True}},
        "yAxis": {"type": "value", "name": "AUC", "min": 0.5, "max": 1.0, "axisLabel": {"color": "#555", "fontSize": 11}},
        "series": [{
            "name": "AUC", "type": "bar", "barWidth": "38%",
            "data": [{"value": v, "itemStyle": {"color": c}} for v, c in zip(aucs, ["#4C72B0", "#2E9E5B", "#D9534F"])],
            "label": {"show": True, "position": "top", "fontSize": 10,
                      "formatter": "function(p){if(p.value==null){return '';}return p.value.toFixed(4);}"},
        }],
    }


# ---------------------------------------------------------------------------
# 3. HTML report assembly
# ---------------------------------------------------------------------------
def build_html(score_report, feat_report, perf_df, png_map: dict, summary: dict) -> str:
    """Assemble the monitoring report HTML (ECharts + PNG fallback)."""
    # JSON records -> DataFrame
    feat_report = pd.DataFrame(feat_report)
    e_opt_score = echarts_score_psi(score_report)
    e_opt_feat = echarts_feature_psi(feat_report)
    e_opt_auc = echarts_auc(perf_df)

    def opt2js(opt: dict) -> str:
        return json.dumps(opt, ensure_ascii=False)

    # Alert statistics (only for the ref_vs_drifted comparison group)
    drifted = feat_report[feat_report["Comparison"] == "ref_vs_drifted"]
    n_severe = int((drifted["PSI_Level"] == "SevereDrift").sum())
    n_moderate = int((drifted["PSI_Level"] == "ModerateDrift").sum())
    score_drift = score_report[1]["PSI_Level"] if len(score_report) > 1 else "NoDrift"
    auc_drop = round((perf_df["AUC"].iloc[0] - perf_df["AUC"].iloc[2]) * 100, 2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Monitoring / Drift Detection Report (PSI - KL)</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
  :root {{ --green:#2E9E5B; --yellow:#E8A33D; --red:#D9534F; --ink:#1A1B1C; --sub:#6B7280; --line:#E5E7EB; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: var(--ink); background:#F7F8FA; line-height:1.6; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 24px 16px 60px; }}
  header {{ background: linear-gradient(135deg,#1F3A5F,#2E5E8C); color:#fff; border-radius: 12px; padding: 28px 32px; }}
  header h1 {{ font-size: 24px; margin-bottom: 6px; }}
  header p {{ color:#C9D8E8; font-size: 13px; }}
  .badges {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }}
  .badge {{ background:rgba(255,255,255,0.14); border-radius:20px; padding:4px 14px; font-size:12px; }}
  h2 {{ font-size: 18px; margin: 34px 0 12px; border-left: 4px solid #2E5E8C; padding-left: 10px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }}
  .kpis {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:18px 0 6px; }}
  .kpi {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .kpi .num {{ font-size: 26px; font-weight: 700; }}
  .kpi .lbl {{ font-size: 12px; color: var(--sub); margin-top:2px; }}
  .green {{ color: var(--green); }} .yellow {{ color: var(--yellow); }} .red {{ color: var(--red); }}
  .chart {{ width: 100%; height: 380px; }}
  .chart-static {{ width:100%; height:auto; border-radius:6px; margin-top:8px; }}
  .note {{ font-size: 12px; color: var(--sub); margin-top: 6px; }}
  table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align:left; }}
  th {{ background:#F3F4F6; }}
  .pill {{ display:inline-block; padding:1px 10px; border-radius:12px; font-size:11px; color:#fff; }}
  .pill.g {{ background: var(--green); }} .pill.y {{ background: var(--yellow); }} .pill.r {{ background: var(--red); }}
  footer {{ margin-top:40px; color:var(--sub); font-size:12px; text-align:center; }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>Model Monitoring / Drift Detection Report</h1>
  <p>Credit Default Prediction Model - Post-launch Drift Monitoring (PSI / KL Visualization) - Generated: {now_str()}</p>
  <div class="badges">
    <span class="badge">Baseline: Logistic Regression</span>
    <span class="badge">Dataset: Give Me Some Credit (FICO / Kaggle)</span>
    <span class="badge">Reference ~120K samples -&gt; Current ~30K samples</span>
  </div>
</header>

<div class="kpis">
  <div class="kpi"><div class="num {('red' if 'Severe' in score_drift else 'yellow' if 'Moderate' in score_drift else 'green')}">{score_drift}</div><div class="lbl">Model score PSI (drifted period)</div></div>
  <div class="kpi"><div class="num red">{n_severe}</div><div class="lbl">Severe-drift features (PSI &gt;= 0.25)</div></div>
  <div class="kpi"><div class="num yellow">{n_moderate}</div><div class="lbl">Moderate-drift features (0.10-0.25)</div></div>
  <div class="kpi"><div class="num {('red' if auc_drop >= 3 else 'yellow' if auc_drop >= 1 else 'green')}">-{auc_drop}%</div><div class="lbl">AUC drop (reference -&gt; drifted)</div></div>
</div>

<h2>1. Model Score Drift (Core Metric)</h2>
<div class="card">
  <div id="chart_score" class="chart"></div>
  <img class="chart-static" src="data:image/png;base64,{png_map['score_distribution']}" alt="Score distribution comparison">
  <p class="note">Interpretation: score PSI measures whether the model's P(bad) distribution has shifted overall. PSI &gt;= 0.25 (red) means the score distribution changed significantly, usually accompanied by a borrower-mix change or model-degradation risk.</p>
</div>

<h2>2. Feature Drift Top (PSI Alerts)</h2>
<div class="card">
  <div id="chart_feat" class="chart"></div>
  <img class="chart-static" src="data:image/png;base64,{png_map['psi_bar']}" alt="Feature PSI alert bar chart">
  <p class="note">Interpretation: PSI &lt; 0.10 no significant drift (green); 0.10 &lt;= PSI &lt; 0.25 moderate drift (yellow); PSI &gt;= 0.25 severe drift (red).</p>
</div>
<div class="card">
  <img class="chart-static" src="data:image/png;base64,{png_map['feature_drift']}" alt="Top feature distribution comparison">
  <p class="note">Above: distribution comparison of the most drifted features (blue = reference, red = drifted).</p>
</div>

<h2>3. Model Performance Degradation</h2>
<div class="card">
  <div id="chart_auc" class="chart"></div>
  <img class="chart-static" src="data:image/png;base64,{png_map['auc_compare']}" alt="AUC comparison chart">
  <p class="note">Interpretation: the AUC drop from reference to drifted measures whether the model's power to separate good / bad borrowers has degraded - the key evidence for triggering a retraining review.</p>
</div>

<h2>4. Feature-level Drift Detail</h2>
<div class="card">
  <table>
    <tr><th>Comparison</th><th>Feature</th><th>PSI</th><th>PSI Level</th><th>KL Divergence</th><th>Ref Median</th><th>Cur Median</th></tr>
    {''.join(f"<tr><td>{r['Comparison']}</td><td>{r['Feature']}</td><td>{r['PSI']:.4f}</td>"
             f"<td><span class='pill {'g' if r['PSI_Level'] == 'NoDrift' else 'y' if r['PSI_Level'] == 'ModerateDrift' else 'r'}'>{r['PSI_Level']}</span></td>"
             f"<td>{r['KL_Divergence']:.4f}</td><td>{r['Ref_Median']:.2f}</td><td>{r['Cur_Median']:.2f}</td></tr>"
             for _, r in feat_report.iterrows())}
  </table>
</div>

<h2>5. Glossary & Methodology</h2>
<div class="card">
  <table>
    <tr><th>Term</th><th>Description</th></tr>
    <tr><td>PSI (Population Stability Index)</td><td>Classic risk metric for distribution difference; PSI = sum((actual_share - expected_share) * ln(actual_share / expected_share))</td></tr>
    <tr><td>KLD (Kullback-Leibler Divergence)</td><td>Information-theoretic measure of distribution difference, aka relative entropy; KL(P||Q) = sum P * ln(P/Q)</td></tr>
    <tr><td>Covariate Shift</td><td>Feature distribution P(X) changes while P(Y|X) stays fixed - the main drift type simulated in this report</td></tr>
    <tr><td>Reference / Current</td><td>Development-period distribution (expected) vs post-launch new-data distribution (actual)</td></tr>
    <tr><td>Equal-frequency Binning</td><td>Quantile-based 10 bins with roughly equal shares - the standard practice for PSI</td></tr>
    <tr><td>AUC / KS</td><td>Discrimination metrics: area under the ROC curve / max gap between good and bad score CDFs</td></tr>
  </table>
  <p class="note">Data source: Give Me Some Credit (provided by FICO, Kaggle competition), 150,000 rows. This report splits "pre/post launch" by row order and injects artificial drift into the current period to demonstrate monitoring capabilities. Thresholds use the industry-standard values 0.10 / 0.25.</p>
</div>

<footer>Model Monitoring - Drift Detection - PSI/KL Visualization - Model Lifecycle (MLOps) demo project</footer>
</div>

<script>
(function () {{
  function render(id, option) {{
    var el = document.getElementById(id);
    if (!el) return;
    if (typeof echarts === 'undefined') return;   // CDN failed -> keep static PNG below
    try {{
      var chart = echarts.init(el);
      chart.setOption(option);
    }} catch (e) {{
      // silently keep the static image on render failure
    }}
  }}
  render('chart_score', {opt2js(e_opt_score)});
  render('chart_feat', {opt2js(e_opt_feat)});
  render('chart_auc', {opt2js(e_opt_auc)});
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def main():
    ensure_dir(CHART_DIR)

    print(f"[{now_str()}] Loading monitoring metrics and data")
    with open(os.path.join(OUT_DIR, "monitoring_metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    feat_report = metrics["feature_report"]
    score_report = metrics["score_report"]
    perf_df = pd.DataFrame(metrics["performance_report"])

    ref_val = pd.read_csv(os.path.join(PROC_DIR, "reference_val.csv"), encoding="utf-8-sig")
    cur_initial = pd.read_csv(os.path.join(PROC_DIR, "current_initial.csv"), encoding="utf-8-sig")
    cur_drifted = pd.read_csv(os.path.join(PROC_DIR, "current_drifted.csv"), encoding="utf-8-sig")

    import joblib
    model = joblib.load(os.path.join(MODEL_DIR, "baseline_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

    def score_of(df: pd.DataFrame) -> np.ndarray:
        X = scaler.transform(df[FEATURES].values)
        return model.predict_proba(X)[:, 1]

    # Generate the 4 static charts
    print(f"[{now_str()}] Generating matplotlib charts")
    png_score = chart_score_distribution(score_of(ref_val), score_of(cur_initial), score_of(cur_drifted))

    feat_drifted = [r for r in feat_report if r["Comparison"] == "ref_vs_drifted"]
    top6 = sorted(feat_drifted, key=lambda r: -r["PSI"])[:6]
    png_feat = chart_feature_drift_top(ref_val, cur_drifted, [r["Feature"] for r in top6])
    png_psi = chart_psi_bar(pd.DataFrame(feat_drifted))
    png_auc = chart_auc_compare(perf_df)

    # Assemble the HTML report (embedded PNG fallback)
    print(f"[{now_str()}] Assembling HTML monitoring report")
    png_map = {
        "score_distribution": img_to_base64(png_score),
        "feature_drift": img_to_base64(png_feat),
        "psi_bar": img_to_base64(png_psi),
        "auc_compare": img_to_base64(png_auc),
    }
    html = build_html(score_report, feat_report, perf_df, png_map, metrics)
    html_path = os.path.join(OUT_DIR, "monitoring_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Alert summary JSON
    summary = {
        "score_psi_drifted": score_report[1]["PSI"],
        "score_psi_level": score_report[1]["PSI_Level"],
        "severe_features": [r["Feature"] for r in feat_drifted if r["PSI"] >= 0.25],
        "moderate_features": [r["Feature"] for r in feat_drifted if 0.10 <= r["PSI"] < 0.25],
        "auc_drop": round(perf_df["AUC"].iloc[0] - perf_df["AUC"].iloc[2], 4),
        "action": "Trigger model retraining review" if any(r["PSI"] >= 0.25 for r in feat_drifted) else "Keep watching",
    }
    with open(os.path.join(OUT_DIR, "monitoring_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"    HTML report: {html_path}")
    print(f"    Charts dir: {CHART_DIR}")
    print(f"[{now_str()}] Report generated")


if __name__ == "__main__":
    main()

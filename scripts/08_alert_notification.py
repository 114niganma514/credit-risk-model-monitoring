# -*- coding: utf-8 -*-
"""
08_alert_notification.py - Automatic alerts when PSI / AUC exceed thresholds
============================================================================
Extension 3: read the rolling-monitoring results (rolling_psi_trend.csv /
rolling_auc_trend.csv / monitoring_summary.json), decide whether to trigger
an alert, and send it through the configured channel:

    Channel 1 Feishu custom bot (Webhook):
        POST {webhook_url}  {"msg_type":"text","content":{"text":...}}
    Channel 2 Email (SMTP):
        smtplib + email.mime to send an alert digest
    Channel 3 Simulate mode (default):
        print the alert only, do not really send - for local debugging / demo

Trigger rules (consistent with config/alert_config.json -> alert_rules):
    - Latest window model-score PSI >= 0.10 (moderate) or >= 0.25 (severe)
    - Latest window AUC drop vs baseline >= 3%
    - Severe-drift feature count >= 1

Usage:
    python scripts/08_alert_notification.py                 # simulate (default)
    python scripts/08_alert_notification.py --channel feishu
    python scripts/08_alert_notification.py --channel email
    python scripts/08_alert_notification.py --channel all

Scheduling (Windows example, daily 09:00 check and alert):
    schtasks /Create /TN "ModelMonitorAlert" /TR "python C:\\...\\scripts\\08_alert_notification.py --channel all" /SC DAILY /ST 09:00
"""
import argparse
import json
import os
import smtplib
import sys
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import now_str, setup_utf8

setup_utf8()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "alert_config.json")
ROLL_DIR = os.path.join(ROOT, "output", "rolling")
OUT_DIR = os.path.join(ROOT, "output")


def load_rules(cfg: dict) -> dict:
    return cfg.get("alert_rules", {})


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_alert_content() -> dict:
    """
    Aggregate monitoring results; returns
    {triggered: bool, title, body_lines, details: dict}
    """
    rules = load_rules(load_config())
    psi_trend = pd.read_csv(os.path.join(ROLL_DIR, "rolling_psi_trend.csv"), encoding="utf-8-sig")
    auc_trend = pd.read_csv(os.path.join(ROLL_DIR, "rolling_auc_trend.csv"), encoding="utf-8-sig")
    summary = json.load(open(os.path.join(OUT_DIR, "monitoring_summary.json"), encoding="utf-8"))

    latest_psi = psi_trend.iloc[-1]
    latest_auc = auc_trend.iloc[-1]

    triggers = []
    # 1) Score PSI
    psi = float(latest_psi["ScorePSI"])
    if psi >= rules["psi_severe"]:
        triggers.append(f"Score PSI = {psi:.4f} (severe, >= {rules['psi_severe']})")
    elif psi >= rules["psi_moderate"]:
        triggers.append(f"Score PSI = {psi:.4f} (moderate, >= {rules['psi_moderate']})")

    # 2) AUC drop
    drop = float(latest_auc["AUC_Drop"])
    if drop >= rules["auc_drop_severe"]:
        triggers.append(f"AUC drop = {drop:.4f} (>= {rules['auc_drop_severe']}, retrain suggested)")

    # 3) Severe-drift feature count
    n_severe = len(summary.get("severe_features", []))
    if n_severe >= rules["severe_feature_count"]:
        triggers.append(f"{n_severe} severe-drift feature(s): {', '.join(summary['severe_features'])}")

    triggered = len(triggers) > 0
    title = "[Model Monitoring Alert] Credit scoring model detected data drift" if triggered \
        else "[Model Monitoring] Check passed - no drift"

    lines = [
        f"Check time: {now_str()}",
        f"Latest window: {latest_psi['Window']} ({latest_psi['Date']})",
        "",
    ]
    if triggered:
        lines += ["Triggered items:", *[f"  [WARN] {t}" for t in triggers], ""]
    lines += [
        f"Score PSI: {psi:.4f} ({latest_psi['ScorePSI_Level']})",
        f"Feature avg PSI: {latest_psi['FeatureAvgPSI']:.4f}",
        f"Latest window AUC: {latest_auc['AUC']:.4f} (drop vs baseline {drop:.4f})",
        f"Severe-drift features: {', '.join(summary.get('severe_features', [])) or 'none'}",
        "",
        "Suggested action: " + ("start a model retraining review" if triggered
                                 else "keep watching; see you at the next check"),
    ]
    return {"triggered": triggered, "title": title, "body_lines": lines, "details": {
        "psi": psi, "psi_level": latest_psi["ScorePSI_Level"], "auc_drop": drop,
        "severe_features": summary.get("severe_features", []),
    }}


# ---------------------------------------------------------------------------
# Channel implementations
# ---------------------------------------------------------------------------
def send_feishu(webhook_url: str, title: str, body: str) -> bool:
    """Feishu custom bot: POST a text message."""
    payload = {"msg_type": "text", "content": {"text": f"{title}\n{body}"}}
    resp = requests.post(webhook_url, json=payload, timeout=10)
    resp.raise_for_status()
    return True


def send_email(cfg: dict, title: str, body: str) -> bool:
    """SMTP email: alert body sent as UTF-8 plain text."""
    host = cfg["smtp_host"]
    port = cfg["smtp_port"]
    user = cfg["username"]
    pwd = cfg["password"]
    from_addr = cfg.get("from_addr") or user
    to_addrs = cfg["to_addrs"]

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(title, "utf-8")
    msg["From"] = formataddr(("Model Monitoring Alert", from_addr))
    msg["To"] = ", ".join(to_addrs)

    if cfg.get("use_ssl", True):
        server = smtplib.SMTP_SSL(host, port, timeout=15)
    else:
        server = smtplib.SMTP(host, port, timeout=15)
        server.starttls()
    try:
        server.login(user, pwd)
        server.sendmail(from_addr, to_addrs, msg.as_string())
    finally:
        server.quit()
    return True


def append_alert_log(details: dict, channel: str) -> str:
    """Append an alert record to output/alert_log.json (all modes log, for audit)."""
    alert_log = os.path.join(OUT_DIR, "alert_log.json")
    history = []
    if os.path.exists(alert_log):
        with open(alert_log, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append({"time": now_str(), "channel": channel, **details})
    with open(alert_log, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return alert_log


def main():
    ap = argparse.ArgumentParser(description="Model monitoring alert sender")
    ap.add_argument("--channel", choices=["simulate", "feishu", "email", "all"], default="simulate",
                    help="sending channel: simulate(default, print only) / feishu / email / all")
    args = ap.parse_args()

    cfg = load_config()
    content = build_alert_content()
    title, body = content["title"], "\n".join(content["body_lines"])

    print("=" * 60)
    print(title)
    print("=" * 60)
    print(body)

    if not content["triggered"]:
        print("\nNo alert triggered; check passed.")
        if args.channel in ("feishu", "email", "all"):
            print("(No alert triggered; no channel message sent)")
        # keep an audit trail even for normal checks
        append_alert_log({**content["details"], "triggered": False}, "normal")
        return

    channels = {"feishu": args.channel in ("feishu", "all"),
                "email": args.channel in ("email", "all")}

    if args.channel == "simulate" or not (channels["feishu"] or channels["email"]):
        print("\n[Simulate mode] Alert triggered - simulated send (nothing was sent). "
              "Configure a real channel and run --channel all.")
        append_alert_log(content["details"], "simulate")
        return

    if channels["feishu"]:
        url = cfg["feishu"]["webhook_url"]
        if not url:
            print("\n[Skipped] Feishu webhook_url not configured (config/alert_config.json)")
        else:
            send_feishu(url, title, body)
            print("\n[Sent] Feishu alert")

    if channels["email"]:
        ecfg = cfg["email"]
        if not (ecfg["smtp_host"] and ecfg["username"] and ecfg["password"] and ecfg["to_addrs"]):
            print("\n[Skipped] Email SMTP config incomplete (config/alert_config.json)")
        else:
            send_email(ecfg, title, body)
            print("\n[Sent] Email alert ->", ecfg["to_addrs"])

    alert_log = append_alert_log(content["details"], args.channel)
    print(f"\nAlert record appended -> {alert_log}")


if __name__ == "__main__":
    main()

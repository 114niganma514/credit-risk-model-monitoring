@echo off
chcp 65001 >nul
REM ============================================================
REM Credit Risk Model Monitoring & Drift Detection System
REM One-click pipeline: Snapshot Monitor + Rolling Early Warning
REM                     + Alert + SHAP Attribution (01-10)
REM Usage: double-click run_all.bat or run in cmd
REM ============================================================
cd /d "%~dp0"

echo ============================================================
echo  [1/10] Data Preparation
echo ============================================================
python scripts\01_data_prep.py || goto :err

echo ============================================================
echo  [2/10] Train Baseline (Logistic Regression)
echo ============================================================
python scripts\02_train_baseline.py || goto :err

echo ============================================================
echo  [3/10] Simulate Drift
echo ============================================================
python scripts\03_simulate_drift.py || goto :err

echo ============================================================
echo  [4/10] Compute PSI / KLD
echo ============================================================
python scripts\04_compute_psi_kld.py || goto :err

echo ============================================================
echo  [5/10] Snapshot Monitoring Report
echo ============================================================
python scripts\05_monitor_report.py || goto :err

echo ============================================================
echo  [6/10] Rolling PSI Trend (Early Warning)
echo ============================================================
python scripts\06_rolling_psi_trend.py || goto :err

echo ============================================================
echo  [7/10] Concept Drift AUC
echo ============================================================
python scripts\07_concept_drift_auc.py || goto :err

echo ============================================================
echo  [8/10] Alert Notification (simulate mode)
echo ============================================================
python scripts\08_alert_notification.py || goto :err

echo ============================================================
echo  [9/10] Train GBDT (LightGBM / XGBoost)
echo ============================================================
python scripts\09_train_gbdt.py || goto :err

echo ============================================================
echo  [10/10] SHAP Drift Attribution
echo ============================================================
python scripts\10_shap_drift_attribution.py || goto :err

echo.
echo ============================================================
echo  ALL DONE!
echo  Open output\monitoring_report.html for the report.
echo  Charts: output\charts , output\rolling , output\gbdt
echo ============================================================
pause
exit /b 0

:err
echo.
echo  [ERROR] A step failed. Check the message above and retry.
pause
exit /b 1

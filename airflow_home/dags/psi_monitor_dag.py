"""
Monthly PSI drift check against logged live decisions.

Separate from credit_risk_pipeline (training) because the two run on
fundamentally different clocks: training runs once whenever the model is
retrained; this DAG re-checks the *same* frozen reference against newly
accumulated /predict traffic every time it runs, so it belongs on its own
schedule (see credit_risk_dag.py's train_task for where the reference
this DAG reads -- models/psi_reference.json -- gets produced and frozen).

Only fully closed calendar months are scored (the current in-progress
month is skipped) -- an incomplete month can look artificially different
just from in-month seasonality, not real population drift, so scoring it
would be a false signal, not a leading indicator.

Drift here is a flag for a human to review, not an auto-retrain trigger
-- consistent with how this repo already treats calibration and model
choice as human-reviewed decisions (see MODEL_CARD.md). A run that finds
significant drift still succeeds; it writes reports/psi_alerts.json
(and tries to open it) rather than failing the task.

Trigger manually: airflow dags trigger credit_risk_psi_monitor
"""
from datetime import datetime

from airflow.sdk import dag, task

PROJECT_ROOT = "/home/nati/Documents/credit-risk"
PROJECT_VENV_PYTHON = f"{PROJECT_ROOT}/venv/bin/python"


@dag(
    dag_id="credit_risk_psi_monitor",
    schedule=None,  # manually triggered for this exercise, same reasoning
                     # as credit_risk_pipeline's schedule=None
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["credit-risk", "monitoring"],
)
def credit_risk_psi_monitor():

    @task.external_python(python=PROJECT_VENV_PYTHON)
    def load_reference_task() -> str:
        # Returns the path, not the parsed contents -- the reference stats
        # contain actual +-inf floats (the outermost bin edges, see
        # monitor.py), and Airflow 3's XCom push serializes with strict JSON
        # (allow_nan=False), which rejects infinities. Same "pass paths, not
        # data" convention credit_risk_dag.py already uses, for a slightly
        # different reason there (payload size, not JSON-compliance).
        psi_reference_path = "/home/nati/Documents/credit-risk/models/psi_reference.json"
        import os
        if not os.path.exists(psi_reference_path):
            raise FileNotFoundError(
                f"{psi_reference_path} not found -- run credit_risk_pipeline "
                "at least once first, it's produced by train_task."
            )
        return psi_reference_path

    @task.external_python(python=PROJECT_VENV_PYTHON)
    def compute_psi_task(psi_reference_path: str) -> dict:
        import sys
        sys.path.insert(0, "/home/nati/Documents/credit-risk/src")
        import json
        import subprocess
        from datetime import datetime

        import pandas as pd
        import decision_log
        from monitor import psi_from_reference, interpret, PSI_MODERATE_SHIFT

        with open(psi_reference_path) as f:
            reference_stats = json.load(f)

        decisions_df = pd.read_sql(
            decision_log.decisions.select(), decision_log.engine, parse_dates=["timestamp"]
        )

        if decisions_df.empty:
            print("No decisions logged yet -- nothing to score.")
            return {"results": [], "alerts": []}

        decisions_df["year_month"] = decisions_df["timestamp"].dt.to_period("M")
        current_month = pd.Timestamp.now().to_period("M")
        closed_months = sorted(m for m in decisions_df["year_month"].unique() if m < current_month)

        if not closed_months:
            print("No fully closed months of decisions yet -- nothing to score.")
            return {"results": [], "alerts": []}

        results = []
        for month in closed_months:
            month_df = decisions_df[decisions_df["year_month"] == month]
            for col, stats in reference_stats.items():
                psi = psi_from_reference(month_df[col], stats)
                results.append({
                    "month": str(month),
                    "column": col,
                    "n": len(month_df),
                    "psi": round(psi, 4),
                    "interpretation": interpret(psi),
                })
                print(f"{str(month)}  {col:25s} PSI = {psi:.4f}  ({interpret(psi)})")

        alerts = [r for r in results if r["psi"] >= PSI_MODERATE_SHIFT]

        report_path = "/home/nati/Documents/credit-risk/reports/psi_monitor.json"
        with open(report_path, "w") as f:
            json.dump({"generated_at": datetime.now().isoformat(), "results": results}, f, indent=2)

        if alerts:
            alert_path = "/home/nati/Documents/credit-risk/reports/psi_alerts.json"
            with open(alert_path, "w") as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "message": "Population drift at or above the moderate-shift threshold -- "
                               "human review recommended before deciding whether to retrain.",
                    "alerts": alerts,
                }, f, indent=2)
            print(f"ALERT: {len(alerts)} (month, column) pair(s) at/above moderate shift -- see {alert_path}")

            # Best-effort -- a monitoring task shouldn't fail just because
            # there's no desktop session to open a viewer in.
            try:
                subprocess.run(["xdg-open", alert_path], check=False, timeout=5)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        else:
            print("No alerts -- all closed months within the moderate-shift threshold.")

        return {"results": results, "alerts": alerts}

    reference = load_reference_task()
    compute_psi_task(reference)


credit_risk_psi_monitor()

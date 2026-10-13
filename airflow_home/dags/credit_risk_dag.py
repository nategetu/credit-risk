"""
Airflow DAG version of src/pipeline.py.

Each step is an independently tracked, independently retriable task, unlike
the plain-Python pipeline where one process either fully succeeds or fully
crashes. Every task's real work runs via ExternalPythonOperator in the
PROJECT'S venv (which has pandas/lightgbm/etc.) rather than Airflow's own venv
(which deliberately does not, to avoid dependency conflicts) -- so every
import of pandas/ingest/etc. has to happen INSIDE each task function, not at
the top of this file, since this file itself is parsed by Airflow's venv.

Intermediate output is written to data/processed/ as CSV and passed between
tasks as a file path via XCom -- XCom is for small values (a path is ~60
bytes), not for shipping a quarter-million-row DataFrame through it directly.

Trigger manually: airflow dags trigger credit_risk_pipeline
"""
from datetime import datetime

from airflow.sdk import dag, task

PROJECT_ROOT = "/home/nati/Documents/credit-risk"
PROJECT_VENV_PYTHON = f"{PROJECT_ROOT}/venv/bin/python"


@dag(
    dag_id="credit_risk_pipeline",
    schedule=None,  # manually triggered for this exercise -- see README for
                     # the logical-date/backfill discussion this deliberately
                     # sidesteps by not being on an automatic schedule
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["credit-risk"],
)
def credit_risk_pipeline():

    @task.external_python(python=PROJECT_VENV_PYTHON)
    def ingest_task() -> str:
        import sys
        sys.path.insert(0, "/home/nati/Documents/credit-risk/src")
        from ingest import load_raw

        df = load_raw()
        out_path = "/home/nati/Documents/credit-risk/data/processed/01_ingested.csv"
        df.to_csv(out_path, index=False)
        print(f"Ingested {len(df):,} rows -> {out_path}")
        return out_path

    @task.external_python(python=PROJECT_VENV_PYTHON)
    def validate_task(ingested_path: str) -> str:
        import sys
        sys.path.insert(0, "/home/nati/Documents/credit-risk/src")
        import pandas as pd
        from validate import validate_loans, clean_invalid_rows, ValidationError

        df = pd.read_csv(ingested_path, parse_dates=["origination_date"])
        try:
            validate_loans(df)
            print("Raw data passed validation with no issues")
        except ValidationError as e:
            print(f"Raw data failed validation, quarantining bad rows:\n{e}")
            df = clean_invalid_rows(df)
            validate_loans(df)
            print("Post-quarantine data passed validation")

        out_path = "/home/nati/Documents/credit-risk/data/processed/02_validated.csv"
        df.to_csv(out_path, index=False)
        return out_path

    @task.external_python(python=PROJECT_VENV_PYTHON)
    def engineer_features_task(validated_path: str) -> str:
        import sys
        sys.path.insert(0, "/home/nati/Documents/credit-risk/src")
        import pandas as pd
        from features import engineer_features

        df = pd.read_csv(validated_path, parse_dates=["origination_date"])
        df = engineer_features(df)
        out_path = "/home/nati/Documents/credit-risk/data/processed/03_features.csv"
        df.to_csv(out_path, index=False)
        print(f"Engineered features, {len(df):,} rows retained -> {out_path}")
        return out_path

    @task.external_python(python=PROJECT_VENV_PYTHON)
    def train_task(features_path: str) -> dict:
        import sys
        sys.path.insert(0, "/home/nati/Documents/credit-risk/src")
        import json
        import pandas as pd
        from train_gbm import time_based_split, train_model, fit_calibrator
        from monitor import fit_reference_distribution, MONITORED_COLUMNS

        df = pd.read_csv(features_path, parse_dates=["origination_date"])
        train_df, calib_df, test_df = time_based_split(df)
        model = train_model(train_df)
        calibrator = fit_calibrator(model, calib_df)

        model_path = "/home/nati/Documents/credit-risk/models/gbm_model.txt"
        model.booster_.save_model(model_path)

        calibrator_path = "/home/nati/Documents/credit-risk/models/gbm_calibrator.json"
        with open(calibrator_path, "w") as f:
            json.dump({
                "coef": float(calibrator.coef_[0][0]),
                "intercept": float(calibrator.intercept_[0]),
            }, f)

        # Frozen once, here, against this exact train_df -- see monitor.py's
        # module docstring for why this never needs recomputing after today.
        psi_reference_path = "/home/nati/Documents/credit-risk/models/psi_reference.json"
        psi_reference = {col: fit_reference_distribution(train_df[col]) for col in MONITORED_COLUMNS}
        with open(psi_reference_path, "w") as f:
            json.dump(psi_reference, f, indent=2)

        test_path = "/home/nati/Documents/credit-risk/data/processed/04_test.csv"
        test_df.to_csv(test_path, index=False)
        return {
            "model_path": model_path,
            "calibrator_path": calibrator_path,
            "test_path": test_path,
            "psi_reference_path": psi_reference_path,
        }

    @task.external_python(python=PROJECT_VENV_PYTHON)
    def evaluate_task(train_result: dict) -> dict:
        import sys
        sys.path.insert(0, "/home/nati/Documents/credit-risk/src")
        import json
        import numpy as np
        import pandas as pd
        import lightgbm as lgb
        from train_gbm import evaluate_model

        # evaluate_model expects an sklearn-style .predict_proba(); the
        # booster we saved/reloaded only has the raw .predict() -- adapt it
        # rather than change evaluate_model's interface.
        class _BoosterAsSklearn:
            def __init__(self, booster):
                self._booster = booster

            def predict_proba(self, X):
                p1 = self._booster.predict(X)
                return np.column_stack([1 - p1, p1])

        # Same adaptation for the calibrator -- reloaded from its 2 saved
        # floats rather than passed through XCom, matching how the model
        # itself is reloaded from disk rather than pickled through XCom.
        class _CalibratorFromJSON:
            def __init__(self, coef, intercept):
                self._coef = coef
                self._intercept = intercept

            def predict_proba(self, X):
                log_odds = self._coef * X[:, 0] + self._intercept
                p1 = 1 / (1 + np.exp(-log_odds))
                return np.column_stack([1 - p1, p1])

        # Module-level constants don't cross the ExternalPythonOperator
        # boundary -- only the function body itself (plus its explicit
        # arguments) gets shipped to the subprocess, so these have to be
        # defined here, not at the top of this file.
        min_test_auc = 0.60
        max_calibration_error = 0.08

        booster = lgb.Booster(model_file=train_result["model_path"])
        with open(train_result["calibrator_path"]) as f:
            calibrator_params = json.load(f)
        calibrator = _CalibratorFromJSON(calibrator_params["coef"], calibrator_params["intercept"])

        test_df = pd.read_csv(train_result["test_path"], parse_dates=["origination_date"])
        model = _BoosterAsSklearn(booster)

        # Gates check the RAW metrics on purpose: api.py serves the raw score
        # (see predict() in api.py), so that's the number CI needs to validate.
        # The calibrated metrics are computed too, purely for a human to
        # review (see MODEL_CARD.md, section 6, for why raw currently ships).
        raw_metrics = evaluate_model(model, test_df)
        calibrated_metrics = evaluate_model(model, test_df, calibrator=calibrator)

        if calibrated_metrics["calibration_error"] < raw_metrics["calibration_error"]:
            print(
                f"\nNote: calibrated calibration error ({calibrated_metrics['calibration_error']:.4f}) "
                f"beats raw ({raw_metrics['calibration_error']:.4f}) on this run. CI still gates on the "
                f"raw model, since that's what api.py currently serves -- if this holds up across "
                f"multiple runs, consider wiring the calibrator into api.py's /predict."
            )

        with open("/home/nati/Documents/credit-risk/reports/pipeline_metrics.json", "w") as f:
            json.dump({"raw": raw_metrics, "calibrated": calibrated_metrics}, f, indent=2)

        if raw_metrics["auc"] < min_test_auc:
            raise ValueError(f"GATE FAILED: AUC {raw_metrics['auc']:.4f} below minimum {min_test_auc}")
        if raw_metrics["calibration_error"] > max_calibration_error:
            raise ValueError(
                f"GATE FAILED: calibration error {raw_metrics['calibration_error']:.4f} "
                f"exceeds max {max_calibration_error}"
            )
        print(f"All gates passed: {raw_metrics}")
        return raw_metrics

    ingested = ingest_task()
    validated = validate_task(ingested)
    features = engineer_features_task(validated)
    trained = train_task(features)
    evaluate_task(trained)


credit_risk_pipeline()

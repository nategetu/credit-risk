"""
Phase 3: the actual pipeline. Explicit, ordered, fail-loud steps.

This is intentionally plain Python rather than an orchestration framework --
get comfortable with the DAG concept here (each step has clear inputs,
outputs, and can fail independently) before adding Prefect/Dagster on top.

Run: python src/pipeline.py
Exit code is nonzero if any gate fails -- this is what CI checks.
"""
import sys
import json
from pathlib import Path

from ingest import load_raw
from validate import validate_loans, clean_invalid_rows, ValidationError
from features import engineer_features
from train_gbm import time_based_split, train_model, fit_calibrator, evaluate_model

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# CI performance gates -- tune these once you have a real baseline.
# 0.60 reflects the actual achievable AUC on the synthetic dataset's baked-in
# noise level; ratchet this up once you swap in real data and a stronger model.
MIN_TEST_AUC = 0.60
MAX_CALIBRATION_ERROR = 0.08


def run_pipeline() -> dict:
    print("STEP 1/5: Ingest")
    df = load_raw()
    print(f"  Loaded {len(df):,} raw rows")

    print("\nSTEP 2/5: Validate")
    try:
        validate_loans(df)
        print("  Raw data passed validation with no issues")
    except ValidationError as e:
        print(f"  Raw data failed validation, quarantining bad rows:\n{e}")
        df = clean_invalid_rows(df)
        validate_loans(df)
        print("  Post-quarantine data passed validation")

    print("\nSTEP 3/5: Feature engineering")
    df = engineer_features(df)
    print(f"  Engineered features, {len(df):,} rows retained")

    print("\nSTEP 4/5: Train (time-based split)")
    train_df, calib_df, test_df = time_based_split(df)
    model = train_model(train_df)
    calibrator = fit_calibrator(model, calib_df)

    print("\nSTEP 5/5: Evaluate against CI gates")
    # Gates check the RAW metrics on purpose: api.py serves the raw score
    # (see predict() in api.py), so that's the number CI needs to validate.
    # The calibrated metrics are computed too, purely for a human to review
    # (see MODEL_CARD.md, section 6, for why raw currently ships).
    raw_metrics = evaluate_model(model, test_df)
    calibrated_metrics = evaluate_model(model, test_df, calibrator=calibrator)

    if calibrated_metrics["calibration_error"] < raw_metrics["calibration_error"]:
        print(
            f"\nNote: calibrated calibration error ({calibrated_metrics['calibration_error']:.4f}) "
            f"beats raw ({raw_metrics['calibration_error']:.4f}) on this run. CI still gates on the "
            f"raw model, since that's what api.py currently serves -- if this holds up across "
            f"multiple runs, consider wiring the calibrator into api.py's /predict."
        )

    MODELS_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODELS_DIR / "gbm_model.txt"))
    with open(MODELS_DIR / "gbm_calibrator.json", "w") as f:
        json.dump({
            "coef": float(calibrator.coef_[0][0]),
            "intercept": float(calibrator.intercept_[0]),
        }, f)

    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "pipeline_metrics.json", "w") as f:
        json.dump({"raw": raw_metrics, "calibrated": calibrated_metrics}, f, indent=2)

    return raw_metrics


def check_gates(metrics: dict) -> bool:
    passed = True
    if metrics["auc"] < MIN_TEST_AUC:
        print(f"\n✗ GATE FAILED: AUC {metrics['auc']:.4f} below minimum {MIN_TEST_AUC}")
        passed = False
    else:
        print(f"\n✓ AUC gate passed: {metrics['auc']:.4f} >= {MIN_TEST_AUC}")

    if metrics["calibration_error"] > MAX_CALIBRATION_ERROR:
        print(f"✗ GATE FAILED: calibration error {metrics['calibration_error']:.4f} "
              f"exceeds max {MAX_CALIBRATION_ERROR}")
        passed = False
    else:
        print(f"✓ Calibration gate passed: {metrics['calibration_error']:.4f} <= {MAX_CALIBRATION_ERROR}")

    return passed


if __name__ == "__main__":
    metrics = run_pipeline()
    ok = check_gates(metrics)
    if not ok:
        print("\nPipeline completed but failed one or more quality gates.")
        sys.exit(1)
    print("\nPipeline completed successfully, all gates passed.")
    sys.exit(0)

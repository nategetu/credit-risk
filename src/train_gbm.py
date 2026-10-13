"""
Phase 2: Predictive modeling done right.

Key habits this script forces on purpose:
  1. TIME-BASED split, not random -- you can't leak 2023 loan outcomes into
     a model deployed to score 2021 applications. Random splits hide this
     bug; time splits expose it.
  2. Calibration is checked, and corrected, not just AUC. A lender prices
     risk off the predicted probability, so "0.15 predicted, 15% actually
     defaulted" matters as much as rank-ordering ability. GBM's raw scores
     are systematically overconfident (boosting keeps chasing a residual
     that's only zero at 0/1), so a Platt-scaling correction is fit on its
     own held-out calibration slice -- reusing the training or test set for
     that fit would just relaunder the same overfitting it's meant to fix.

Run this file directly: `python src/train_gbm.py`
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

from ingest import load_raw
from validate import clean_invalid_rows
from features import engineer_features, FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def time_based_split(df: pd.DataFrame, calib_frac: float = 0.15, test_frac: float = 0.15):
    df = df.sort_values("origination_date")
    n = len(df)
    train_end_idx = int(n * (1 - calib_frac - test_frac))
    calib_end_idx = int(n * (1 - test_frac))
    train_cutoff = df.iloc[train_end_idx]["origination_date"]
    calib_cutoff = df.iloc[calib_end_idx]["origination_date"]

    train = df[df["origination_date"] < train_cutoff]
    calib = df[(df["origination_date"] >= train_cutoff) & (df["origination_date"] < calib_cutoff)]
    test = df[df["origination_date"] >= calib_cutoff]

    print(f"Train:       {len(train):,} loans before {train_cutoff.date()}")
    print(f"Calibration: {len(calib):,} loans {train_cutoff.date()} to {calib_cutoff.date()}")
    print(f"Test:        {len(test):,} loans on/after {calib_cutoff.date()}")
    return train, calib, test


def train_model(train_df: pd.DataFrame, params: dict | None = None):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["defaulted"]

    default_params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "min_child_samples": 30,
        "verbosity": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train, y_train)
    return model


def fit_calibrator(model, calib_df: pd.DataFrame) -> LogisticRegression:
    """Platt scaling: a 1-D logistic regression mapping the GBM's raw score
    to a corrected probability, fit on a calibration slice the GBM never
    trained on."""
    X_calib = calib_df[FEATURE_COLUMNS]
    y_calib = calib_df["defaulted"]
    raw_probs = model.predict_proba(X_calib)[:, 1]

    calibrator = LogisticRegression()
    calibrator.fit(raw_probs.reshape(-1, 1), y_calib)
    return calibrator


def evaluate_model(model, test_df: pd.DataFrame, calibrator: LogisticRegression | None = None) -> dict:
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["defaulted"]
    probs = model.predict_proba(X_test)[:, 1]
    if calibrator is not None:
        probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]

    auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))

    label = "calibrated" if calibrator is not None else "raw"
    print(f"\n[{label}] Test AUC: {auc:.4f}")
    print(f"[{label}] Brier score (lower is better): {brier:.4f}")
    print(f"[{label}] Mean calibration error across bins: {calibration_error:.4f}")
    print(f"[{label}] Calibration curve (predicted vs actual, by decile):")
    for pt, pp in zip(prob_true, prob_pred):
        print(f"  predicted={pp:.3f}  actual={pt:.3f}")

    return {"auc": auc, "brier": brier, "calibration_error": calibration_error}


if __name__ == "__main__":
    df = clean_invalid_rows(load_raw())
    df = engineer_features(df)

    train_df, calib_df, test_df = time_based_split(df)
    model = train_model(train_df)
    calibrator = fit_calibrator(model, calib_df)

    raw_metrics = evaluate_model(model, test_df)
    calibrated_metrics = evaluate_model(model, test_df, calibrator=calibrator)

    MODELS_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODELS_DIR / "gbm_model.txt"))
    calibrator_path = MODELS_DIR / "gbm_calibrator.json"
    calibrator_path.write_text(json.dumps({
        "coef": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
    }))
    print(f"\nModel saved to {MODELS_DIR / 'gbm_model.txt'}")
    print(f"Calibrator saved to {calibrator_path}")

    REPORTS_DIR.mkdir(exist_ok=True)
    # Both are saved, not just whichever looks better -- deciding which one
    # ships is a model-validation call for a person to make (see
    # MODEL_CARD.md, section 6), not something this script should silently
    # auto-select.
    metrics_path = REPORTS_DIR / "gbm_metrics.json"
    metrics_path.write_text(json.dumps({"raw": raw_metrics, "calibrated": calibrated_metrics}, indent=2))
    print(f"Metrics saved to {metrics_path}")

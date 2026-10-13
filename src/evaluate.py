"""
Compares the Phase 1 logistic model against the Phase 2 GBM model on the
same test set. This is a real conversation in consumer lending: an
interpretable model you can fully explain vs. a black-box model with better
raw performance. Having both numbers side by side lets you make that
tradeoff with data instead of vibes.
"""
from pathlib import Path
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from ingest import load_raw
from validate import clean_invalid_rows
from features import engineer_features, FEATURE_COLUMNS
from train_logistic import fit_logistic
from train_gbm import time_based_split, train_model

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def compare_models(df):
    train_df, _calib_df, test_df = time_based_split(df)

    logit_model = fit_logistic(train_df)
    X_test_const = sm.add_constant(test_df[FEATURE_COLUMNS].astype(float), has_constant="add")
    logit_probs = logit_model.predict(X_test_const)
    logit_auc = roc_auc_score(test_df["defaulted"], logit_probs)

    gbm_model = train_model(train_df)
    gbm_probs = gbm_model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    gbm_auc = roc_auc_score(test_df["defaulted"], gbm_probs)

    print(f"Logistic regression AUC: {logit_auc:.4f}  (fully interpretable)")
    print(f"LightGBM AUC:            {gbm_auc:.4f}  (better rank-ordering, harder to explain)")
    print(f"AUC gap: {gbm_auc - logit_auc:+.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, probs in [("Logistic", logit_probs), ("LightGBM", gbm_probs)]:
        prob_true, prob_pred = calibration_curve(test_df["defaulted"], probs, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Predicted default probability")
    ax.set_ylabel("Actual default rate")
    ax.set_title("Calibration: Logistic vs LightGBM")
    ax.legend()
    REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORTS_DIR / "calibration_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nCalibration plot saved to {REPORTS_DIR / 'calibration_comparison.png'}")

    return {"logit_auc": logit_auc, "gbm_auc": gbm_auc}


if __name__ == "__main__":
    df = clean_invalid_rows(load_raw())
    df = engineer_features(df)
    compare_models(df)

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_gbm import time_based_split, train_model, fit_calibrator, evaluate_model
from features import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# time_based_split: a pure, deterministic function of its input -- testable
# with hand-built data and exact assertions, same pattern as test_features.py.
# ---------------------------------------------------------------------------

def _dated_df(n=10):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    # shuffle on purpose: the function must sort internally, not assume
    # its caller already did.
    shuffled = dates.to_series().sample(frac=1, random_state=0).reset_index(drop=True)
    return pd.DataFrame({"origination_date": shuffled, "row_id": range(n)})


def test_time_based_split_no_leakage():
    train, calib, test = time_based_split(_dated_df(10), calib_frac=0.2, test_frac=0.2)
    assert train["origination_date"].max() < calib["origination_date"].min()
    assert calib["origination_date"].max() < test["origination_date"].min()


def test_time_based_split_sizes_and_no_dropped_rows():
    train, calib, test = time_based_split(_dated_df(10), calib_frac=0.2, test_frac=0.2)
    assert len(train) == 6
    assert len(calib) == 2
    assert len(test) == 2
    # every original row_id appears in exactly one of the three splits
    all_ids = train["row_id"].tolist() + calib["row_id"].tolist() + test["row_id"].tolist()
    assert sorted(all_ids) == list(range(10))


# ---------------------------------------------------------------------------
# evaluate_model: doesn't need a real trained model -- it only calls
# model.predict_proba(). A fake stand-in with hand-picked outputs makes the
# whole calculation exactly hand-verifiable, same as the tests above.
# ---------------------------------------------------------------------------

class _FakeModel:
    """Stands in for a trained model: returns fixed, known probabilities
    regardless of X, so evaluate_model()'s math becomes exactly checkable."""

    def __init__(self, probs_class1):
        self.probs_class1 = np.array(probs_class1)

    def predict_proba(self, X):
        return np.column_stack([1 - self.probs_class1, self.probs_class1])


def _eval_test_df(labels):
    # evaluate_model slices test_df[FEATURE_COLUMNS] before ever touching
    # the model, so the fake model's input still needs these columns to
    # exist, even though _FakeModel ignores their values entirely.
    df = pd.DataFrame({col: 0.0 for col in FEATURE_COLUMNS}, index=range(len(labels)))
    df["defaulted"] = labels
    return df


def test_evaluate_model_perfect_separation():
    # 0.6 and 0.9 (the two actual defaulters) both rank above 0.1 and 0.4
    # (the two non-defaulters) -- every positive/negative pair is correctly
    # ordered, so AUC must be exactly 1.0.
    model = _FakeModel([0.1, 0.4, 0.6, 0.9])
    metrics = evaluate_model(model, _eval_test_df([0, 0, 1, 1]))

    assert metrics["auc"] == 1.0
    # brier = mean((pred - actual)^2) = (0.1^2 + 0.4^2 + 0.4^2 + 0.1^2) / 4
    assert np.isclose(metrics["brier"], 0.085)
    assert np.isclose(metrics["calibration_error"], 0.25)


def test_evaluate_model_inverted_predictions_score_near_zero_auc():
    # The two actual defaulters (rows 2, 3) get the LOWEST scores here --
    # every pair is incorrectly ordered, so AUC must be exactly 0.0. This
    # is the bug class a "just check AUC is a valid probability-like number"
    # test would miss entirely: confidently scoring the wrong class as risky
    # (e.g. accidentally using predict_proba[:, 0] instead of [:, 1]).
    model = _FakeModel([0.9, 0.6, 0.4, 0.1])
    metrics = evaluate_model(model, _eval_test_df([0, 0, 1, 1]))
    assert metrics["auc"] == 0.0


# ---------------------------------------------------------------------------
# train_model: can't hand-verify its output, but real properties still hold:
# it should be deterministic, it should recover an obvious signal when one
# exists, and it should NOT confidently "recover" a signal that isn't there.
# ---------------------------------------------------------------------------

def _synthetic_training_df(n, informative, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: 0.0 for col in FEATURE_COLUMNS}, index=range(n))
    df["dti"] = rng.uniform(0, 40, n)
    if informative:
        df["defaulted"] = (df["dti"] > 20).astype(int)
    else:
        # label independent of every feature, including dti -- there is
        # nothing in X for the model to legitimately learn.
        df["defaulted"] = rng.integers(0, 2, n)
    return df


def test_train_model_recovers_strong_signal():
    train_df = _synthetic_training_df(300, informative=True, seed=1)
    test_df = _synthetic_training_df(100, informative=True, seed=2)

    model = train_model(train_df)
    metrics = evaluate_model(model, test_df)

    assert metrics["auc"] > 0.9


def test_train_model_does_not_fabricate_signal_from_noise():
    train_df = _synthetic_training_df(300, informative=False, seed=1)
    test_df = _synthetic_training_df(100, informative=False, seed=2)

    model = train_model(train_df)
    metrics = evaluate_model(model, test_df)

    # Not tightly bounded around 0.5: a 300-tree model can overfit spurious
    # patterns in a modest, purely-noise training set, and some of that
    # generalizes by chance to a same-distribution test set. The bound here
    # exists to catch a real bug (e.g. label leakage) that would push a
    # genuinely unrelated dataset's AUC much higher, not to assert "exactly
    # chance-level."
    assert metrics["auc"] < 0.75


def test_train_model_is_deterministic():
    train_df = _synthetic_training_df(200, informative=True, seed=1)
    test_df = _synthetic_training_df(50, informative=True, seed=2)
    X_test = test_df[FEATURE_COLUMNS]

    model_a = train_model(train_df)
    model_b = train_model(train_df)

    preds_a = model_a.predict_proba(X_test)
    preds_b = model_b.predict_proba(X_test)
    assert np.array_equal(preds_a, preds_b)


# ---------------------------------------------------------------------------
# fit_calibrator: can't hand-verify the fitted coefficients, but the property
# that matters holds regardless of exact values -- correcting deliberately
# overconfident raw scores should land closer to the true generating
# probability than the raw scores were.
# ---------------------------------------------------------------------------

def test_fit_calibrator_corrects_overconfident_raw_scores():
    rng = np.random.default_rng(0)
    n = 2000
    true_prob = rng.uniform(0.1, 0.9, n)
    labels = (rng.uniform(0, 1, n) < true_prob).astype(int)
    # exaggerate true_prob toward the extremes -- the same overconfidence
    # boosting produces by chasing a residual that's only zero at 0/1.
    raw_probs = np.clip(true_prob ** 0.3, 0.001, 0.999)

    calib_df = pd.DataFrame({col: 0.0 for col in FEATURE_COLUMNS}, index=range(n))
    calib_df["defaulted"] = labels

    calibrator = fit_calibrator(_FakeModel(raw_probs), calib_df)
    calibrated_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    raw_error = np.mean(np.abs(raw_probs - true_prob))
    calibrated_error = np.mean(np.abs(calibrated_probs - true_prob))
    assert calibrated_error < raw_error

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_logistic import (
    check_multicollinearity, fit_logistic, summarize_coefficients, disparate_impact_check,
)
from features import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# check_multicollinearity: pure function of X, no model dependency -- exact-
# value testable directly, same pattern as test_features.py.
# ---------------------------------------------------------------------------

def test_check_multicollinearity_orthogonal_columns_have_vif_one():
    # Independently-drawn columns are (to floating-point precision)
    # orthogonal: regressing one on the others gives R^2 ~ 0, so
    # VIF = 1 / (1 - R^2) ~ 1.0 exactly, by definition -- not a fitting
    # quirk, a direct consequence of what VIF means.
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(size=5000),
        "b": rng.normal(size=5000),
        "c": rng.normal(size=5000),
    })
    vif = check_multicollinearity(X)
    assert np.allclose(vif["VIF"], 1.0, atol=0.05)


def test_check_multicollinearity_flags_near_perfect_collinearity():
    # "b" is (almost) a linear function of "a" -- tiny noise added only to
    # keep the design matrix non-singular. R^2 -> 1, so VIF should be huge.
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    X = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(scale=1e-6, size=5000)})
    vif = check_multicollinearity(X)
    assert (vif["VIF"] > 1000).all()


# ---------------------------------------------------------------------------
# fit_logistic: deterministic (concave log-likelihood, unique MLE) but not
# hand-verifiable for real feature data -- no closed form, and its signature
# always uses the full FEATURE_COLUMNS set, so there's no way to isolate an
# intercept-only closed-form case here either. Property-based only.
# ---------------------------------------------------------------------------

def _synthetic_df(n, informative, seed):
    rng = np.random.default_rng(seed)
    # Every column gets real noise, not a constant -- multiple all-zero
    # columns plus an intercept makes the design matrix singular (their
    # coefficients become mathematically unidentifiable).
    df = pd.DataFrame({col: rng.normal(0, 1, n) for col in FEATURE_COLUMNS}, index=range(n))
    df["dti"] = rng.uniform(0, 40, n)
    if informative:
        # Probabilistic, not a hard deterministic threshold -- a threshold
        # would perfectly separate the classes, which breaks logistic
        # regression's MLE entirely (log-likelihood diverges to infinity
        # rather than converging). Real pathology, not just a fixture issue.
        prob_default = 1 / (1 + np.exp(-(df["dti"] - 20) / 3))
        df["defaulted"] = rng.binomial(1, prob_default)
    else:
        df["defaulted"] = rng.integers(0, 2, n)
    return df


def test_fit_logistic_is_deterministic():
    df = _synthetic_df(300, informative=True, seed=1)
    model_a = fit_logistic(df)
    model_b = fit_logistic(df)
    assert np.array_equal(model_a.params.values, model_b.params.values)


def test_fit_logistic_recovers_sign_of_strong_signal():
    # dti > 20 deterministically causes default here, so the fitted dti
    # coefficient must be positive (higher dti -> higher log-odds of
    # default) and statistically significant.
    df = _synthetic_df(500, informative=True, seed=1)
    model = fit_logistic(df)
    assert model.params["dti"] > 0
    assert model.pvalues["dti"] < 0.05


# ---------------------------------------------------------------------------
# summarize_coefficients: only touches model.params / .pvalues / .conf_int()
# -- a fake with hand-picked values makes the transformation exactly
# checkable without ever fitting a real model.
# ---------------------------------------------------------------------------

class _FakeFitResult:
    def __init__(self, params, pvalues, conf_int):
        self.params = pd.Series(params)
        self.pvalues = pd.Series(pvalues)
        self._conf_int = pd.DataFrame(conf_int, index=list(params.keys()))

    def conf_int(self):
        return self._conf_int


def test_summarize_coefficients_math_and_const_dropped():
    model = _FakeFitResult(
        params={"const": -1.0, "dti": 0.5, "log_income": -0.2},
        pvalues={"const": 0.0, "dti": 0.01, "log_income": 0.40},
        conf_int={0: [-1.5, 0.3, -0.6], 1: [-0.5, 0.7, 0.2]},
    )
    summary = summarize_coefficients(model)

    assert "const" not in summary.index
    assert np.isclose(summary.loc["dti", "odds_ratio"], np.exp(0.5))
    assert bool(summary.loc["dti", "significant_at_5pct"]) is True
    assert bool(summary.loc["log_income", "significant_at_5pct"]) is False


# ---------------------------------------------------------------------------
# disparate_impact_check: only touches model.predict() -- fake it, then the
# groupby/spread math on a small hand-built df is exactly hand-verifiable.
# ---------------------------------------------------------------------------

class _FakePredictModel:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, X):
        return self.predictions


def test_disparate_impact_check_group_means_and_spread():
    df = pd.DataFrame({col: 0.0 for col in FEATURE_COLUMNS}, index=range(4))
    df["region"] = ["A", "A", "B", "B"]
    # A's predictions: 0.10, 0.20 -> mean 0.15
    # B's predictions: 0.50, 0.70 -> mean 0.60
    model = _FakePredictModel([0.10, 0.20, 0.50, 0.70])

    by_group = disparate_impact_check(df, model, group_col="region")

    assert np.isclose(by_group["A"], 0.15)
    assert np.isclose(by_group["B"], 0.60)
    assert np.isclose(by_group.max() - by_group.min(), 0.45)

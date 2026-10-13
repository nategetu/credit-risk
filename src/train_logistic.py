"""
Phase 1: Inference, not just prediction.

The point of this script is NOT to get the best accuracy. It's to be able to
say, with statistical backing: "a $10k increase in DTI-adjusted risk moves
default odds by X%, and we're confident about that because the CI doesn't
cross zero." This is the muscle analysts usually don't build, because BI
tools don't ask you to interpret a coefficient's uncertainty.

Run this file directly: `python src/train_logistic.py`
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ingest import load_raw
from validate import clean_invalid_rows
from features import engineer_features, FEATURE_COLUMNS


def check_multicollinearity(X: pd.DataFrame) -> pd.DataFrame:
    """VIF > 5-10 signals a feature is redundant given the others."""
    X_const = sm.add_constant(X)
    vif = pd.DataFrame({
        "feature": X_const.columns,
        "VIF": [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
    })
    return vif[vif["feature"] != "const"].sort_values("VIF", ascending=False)


def fit_logistic(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["defaulted"].astype(float)
    X_const = sm.add_constant(X)

    model = sm.Logit(y, X_const).fit(disp=0)
    return model


def summarize_coefficients(model) -> pd.DataFrame:
    """
    The actually useful output of Phase 1: for each feature, the odds-ratio
    interpretation and whether we're confident the effect is nonzero.
    """
    conf = model.conf_int()
    conf.columns = ["ci_low", "ci_high"]
    summary = pd.DataFrame({
        "coef": model.params,
        "p_value": model.pvalues,
        "ci_low": conf["ci_low"],
        "ci_high": conf["ci_high"],
    })
    summary["odds_ratio"] = np.exp(summary["coef"])
    summary["significant_at_5pct"] = summary["p_value"] < 0.05
    return summary.drop(index="const").sort_values("p_value")


def disparate_impact_check(df: pd.DataFrame, model, group_col: str = "region") -> pd.DataFrame:
    """
    Real consumer lending models get audited for whether predicted risk (and
    resulting approval/pricing decisions) differ across protected-class
    proxies. `region` in the synthetic data is intentionally NOT causally
    related to true default risk -- if this check flags it anyway, that's
    worth understanding (proxy correlation with income/DTI, or model
    picking up spurious signal), which is exactly the kind of finding real
    fair-lending review has to chase down.
    """
    X = df[FEATURE_COLUMNS].astype(float)
    X_const = sm.add_constant(X, has_constant="add")
    df = df.copy()
    df["predicted_prob"] = model.predict(X_const)

    by_group = df.groupby(group_col)["predicted_prob"].mean().sort_values()
    spread = by_group.max() - by_group.min()
    print(f"\nMean predicted default probability by {group_col}:")
    print(by_group.round(4))
    print(f"Max spread across groups: {spread:.4f}")
    if spread > 0.03:
        print("⚠ Spread exceeds 3pp — worth investigating whether this proxy "
              "correlates with a legitimate risk factor or is spurious.")
    return by_group


if __name__ == "__main__":
    df = clean_invalid_rows(load_raw())
    df = engineer_features(df)

    model = fit_logistic(df)

    print("=" * 70)
    print("LOGISTIC REGRESSION SUMMARY (statistical inference)")
    print("=" * 70)
    print(summarize_coefficients(model).round(4).to_string())

    print("\n" + "=" * 70)
    print("MULTICOLLINEARITY CHECK (VIF)")
    print("=" * 70)
    print(check_multicollinearity(df[FEATURE_COLUMNS].astype(float)).round(2).to_string(index=False))

    print("\n" + "=" * 70)
    print("DISPARATE IMPACT CHECK")
    print("=" * 70)
    disparate_impact_check(df, model)

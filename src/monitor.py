"""
Population stability index (PSI): a covariate-shift monitor.

Detects whether the population currently being scored looks like the
population the model was trained on -- without needing outcome labels,
which for a loan can take 36-60 months to materialize (see the maturity
cutoff in ingest.py). This is a leading indicator, not a substitute for
eventually checking real accuracy once labels exist -- it can flag a
population shift that turns out not to matter, and it's blind to concept
drift (the same population, but the true feature-to-outcome relationship
changing), since it only ever looks at feature distributions, never
outcomes.

Split into a fit step and a score step because the reference (training)
side of PSI is fixed forever once training is done -- the bin edges and
expected_pct are both fully determined by the reference population alone
(qcut constructs the edges so each bin holds exactly 1/bins of reference,
by definition). fit_reference_distribution() is meant to run once per
trained model and get persisted (see credit_risk_dag.py's train_task,
which saves it to models/psi_reference.json); psi_from_reference() then
reruns cheaply against any new comparison population without ever
touching the training data again.

Run this file directly for a demonstration against real drift already
found in the Lending Club data (early vs. late vintages):
`CREDIT_RISK_RAW_DATA=data/raw/accepted_2007_to_2018Q4.csv.gz python src/monitor.py`
"""
import numpy as np
import pandas as pd

# Standard credit-risk-industry conventions for interpreting a PSI score.
PSI_NO_SHIFT = 0.10
PSI_MODERATE_SHIFT = 0.25

# Floor applied to zero-count bins so ln(actual/expected) stays finite -- a
# bin genuinely emptying out is a real, large drift signal, not an error
# to suppress.
EPSILON = 1e-4

MONITORED_COLUMNS = [
    "dti", "log_income", "loan_amount", "credit_history_months",
    "open_accounts", "delinquencies_2yr", "loan_to_income",
    "fico_score", "revol_util", "inquiries_6m", "total_accounts",
]


def fit_reference_distribution(reference: pd.Series, bins: int = 10) -> dict:
    """
    The frozen half of PSI: bin edges and expected_pct, both determined
    entirely by the reference (training) population. Persist the return
    value (it's JSON-serializable) and reuse it forever -- reference never
    needs to be touched again after this call.
    """
    edges = pd.qcut(reference, q=bins, duplicates="drop", retbins=True)[1]
    edges[0], edges[-1] = -np.inf, np.inf  # catch comparison values outside reference's observed range

    expected_pct = pd.cut(reference, bins=edges).value_counts(sort=False) / len(reference)
    expected_pct = expected_pct.clip(lower=EPSILON)

    return {"edges": edges.tolist(), "expected_pct": expected_pct.tolist()}


def psi_from_reference(comparison: pd.Series, reference_stats: dict) -> float:
    """Score a new comparison population against an already-fit reference."""
    edges = reference_stats["edges"]
    expected_pct = np.array(reference_stats["expected_pct"])

    actual_pct = pd.cut(comparison, bins=edges).value_counts(sort=False) / len(comparison)
    actual_pct = actual_pct.clip(lower=EPSILON).to_numpy()

    return float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())


def population_stability_index(reference: pd.Series, comparison: pd.Series, bins: int = 10) -> float:
    """Convenience one-shot version (fit + score) for exploratory use, e.g.
    the demo below. Production usage should call the two steps separately
    so the reference side only ever gets computed once -- see module docstring."""
    return psi_from_reference(comparison, fit_reference_distribution(reference, bins=bins))


def interpret(psi: float) -> str:
    if psi < PSI_NO_SHIFT:
        return "no significant shift"
    if psi < PSI_MODERATE_SHIFT:
        return "moderate shift -- worth investigating"
    return "significant shift -- population has meaningfully changed"


if __name__ == "__main__":
    from ingest import load_raw
    from validate import clean_invalid_rows
    from features import engineer_features

    df = clean_invalid_rows(load_raw())
    df = engineer_features(df)
    df["year"] = df["origination_date"].dt.year

    reference = df[df["year"] <= 2009]
    comparison = df[df["year"] >= 2014]

    print(f"Reference (training-like): {len(reference):,} loans, {reference['year'].min()}-{reference['year'].max()}")
    print(f"Comparison (later vintage): {len(comparison):,} loans, {comparison['year'].min()}-{comparison['year'].max()}")
    print()

    for col in MONITORED_COLUMNS:
        reference_stats = fit_reference_distribution(reference[col])
        psi = psi_from_reference(comparison[col], reference_stats)
        print(f"{col:25s} PSI = {psi:.4f}  ({interpret(psi)})")

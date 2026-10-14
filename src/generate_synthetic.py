"""
Generates a synthetic consumer loan dataset shaped like Lending Club /
Home Credit data, with a KNOWN underlying default-risk relationship baked in.

Why synthetic data at all: this environment has no network access to Kaggle,
and more importantly, having ground-truth known coefficients lets you sanity
check your Phase 1 inference work ("did my regression recover something close
to the true relationship I baked in?") before you ever touch real data.

Once you've downloaded a real dataset, point src/ingest.py at it instead and
treat this as a first-pass integration test fixture.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_LOANS = 20_000
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "loans_synthetic.csv"


def generate(n=N_LOANS, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    origination_date = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 5, size=n), unit="D"
    )

    annual_income = rng.lognormal(mean=10.8, sigma=0.45, size=n).round(2)
    loan_amount = rng.uniform(1000, 40000, size=n).round(2)
    credit_history_months = rng.integers(6, 360, size=n)
    open_accounts = rng.integers(1, 25, size=n)
    delinquencies_2yr = rng.poisson(0.35, size=n)
    dti = np.clip(rng.normal(loc=18, scale=9, size=n), 0, 60).round(2)
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "medical", "other"],
        size=n, p=[0.4, 0.25, 0.15, 0.1, 0.1],
    )
    # region is a geography-derived proxy field, kept deliberately uncorrelated
    # with true default risk, so you can test your fairness/disparate-impact
    # check in train_logistic.py against a variable that SHOULDN'T predict risk.
    region = rng.choice(["A", "B", "C", "D"], size=n)

    fico_score = np.clip(rng.normal(695, 30, size=n), 620, 850).round()
    term_months = rng.choice([36, 60], size=n, p=[0.75, 0.25])
    revol_util = np.clip(rng.normal(55, 24, size=n), 0, 150).round(1)
    revol_bal = (annual_income * rng.uniform(0, 0.4, size=n)).round(2)
    inquiries_6m = rng.poisson(0.7, size=n)
    total_accounts = open_accounts + rng.integers(0, 30, size=n)
    emp_length_years = rng.integers(0, 11, size=n).astype(float)
    emp_length_years[rng.random(n) < 0.05] = np.nan  # ~5% missing, like LC
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n, p=[0.40, 0.50, 0.10])

    # ---- true underlying default-risk process (log-odds), used to LABEL data ----
    log_odds = (
        -3.2
        + 0.55 * (dti / 10)
        - 0.35 * (np.log(annual_income) - 10.8)
        + 0.40 * delinquencies_2yr
        - 0.15 * (credit_history_months / 60)
        + 0.10 * (loan_amount / 10000)
        - 0.25 * ((fico_score - 695) / 30)
        + 0.35 * (term_months == 60)
        + 0.10 * ((revol_util - 55) / 24)
        + 0.12 * inquiries_6m
        + 0.15 * (home_ownership == "RENT")
        + rng.normal(0, 0.6, size=n)  # irreducible noise
    )
    default_prob = 1 / (1 + np.exp(-log_odds))
    defaulted = rng.binomial(1, default_prob)

    df = pd.DataFrame({
        "loan_id": np.arange(1, n + 1),
        "origination_date": origination_date,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_history_months": credit_history_months,
        "open_accounts": open_accounts,
        "delinquencies_2yr": delinquencies_2yr,
        "dti": dti,
        "loan_purpose": loan_purpose,
        "region": region,
        "defaulted": defaulted,
        "fico_score": fico_score,
        "term_months": term_months,
        "revol_util": revol_util,
        "revol_bal": revol_bal,
        "inquiries_6m": inquiries_6m,
        "total_accounts": total_accounts,
        "emp_length_years": emp_length_years,
        "home_ownership": home_ownership,
    })

    # sprinkle in some realistic data quality issues on purpose, so Phase 3's
    # validation step has something to actually catch
    dirty_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[dirty_idx[: len(dirty_idx) // 2], "annual_income"] = -1  # bad values
    df.loc[dirty_idx[len(dirty_idx) // 2:], "dti"] = 150  # impossible DTI

    return df.sort_values("origination_date").reset_index(drop=True)


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} synthetic loans to {OUT_PATH}")
    print(f"Default rate: {df['defaulted'].mean():.2%}")

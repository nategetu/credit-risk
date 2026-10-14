"""
Ingest step: load raw loan data and standardize column names/types.

Everything downstream (validate, features, train) depends only on the
standardized STANDARD_COLUMNS names, so this is the ONLY file that should
need to change when you switch data sources. It sniffs the raw file's header
to decide which source-specific transform to run, rather than hardcoding a
filename -- add a new `_looks_like_x` / `_load_x` pair here for any future
data source.
"""
import os
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

STANDARD_COLUMNS = [
    "loan_id", "origination_date", "annual_income", "loan_amount",
    "credit_history_months", "open_accounts", "delinquencies_2yr",
    "dti", "loan_purpose", "region", "defaulted",
    # Credit-bureau and application fields added after the first model
    # (AUC 0.60 on the original 11 features). All are known at application
    # time. Deliberately NOT included: Lending Club's own `grade`,
    # `sub_grade`, and `int_rate` -- those are LC's risk model output, so
    # using them would mostly mean re-learning LC's pricing.
    "fico_score", "term_months", "revol_util", "revol_bal", "inquiries_6m",
    "total_accounts", "emp_length_years", "home_ownership",
]

# Identity mapping: the synthetic generator already writes STANDARD_COLUMNS.
RAW_COLUMNS = {c: c for c in STANDARD_COLUMNS}

# Lending Club "accepted loans" schema (kaggle.com/datasets/wordsforthewise/lending-club).
LC_LOAN_STATUS_TO_DEFAULTED = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    # Everything else (Current, Late, In Grace Period, ...) has no resolved
    # outcome yet -- including it would either fabricate a label or leak
    # information, so those rows are dropped, not mapped.
}
LC_USECOLS = [
    "id", "issue_d", "annual_inc", "loan_amnt", "earliest_cr_line",
    "open_acc", "delinq_2yrs", "dti", "purpose", "zip_code", "loan_status",
    "term", "fico_range_low", "fico_range_high", "revol_util", "revol_bal",
    "inq_last_6mths", "total_acc", "emp_length", "home_ownership",
    # Not used: mort_acc. It's 100% missing for every 2007-2011 vintage, so
    # under a time-based split the training set would be mostly NaN while
    # the test set is fully populated -- a train/test mismatch, not signal.
]
LC_SAMPLE_SIZE = 200_000
LC_SAMPLE_SEED = 42


def _looks_like_lending_club(header: list[str]) -> bool:
    return "loan_amnt" in header and "issue_d" in header and "loan_status" in header


def _load_lending_club(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=LC_USECOLS, low_memory=False)
    df["origination_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

    # Snapshot date must come from the FULL dataset (every loan, including
    # still-active ones), not from the terminal-status subset below -- the
    # max origination date among already-resolved loans is itself censored
    # by the same maturity effect we're trying to correct for.
    snapshot_period = df["origination_date"].dt.to_period("M").max()

    df = df[df["loan_status"].isin(LC_LOAN_STATUS_TO_DEFAULTED)].copy()
    df["defaulted"] = df["loan_status"].map(LC_LOAN_STATUS_TO_DEFAULTED)

    earliest_cr_line = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_history_months"] = (
        (df["origination_date"].dt.year - earliest_cr_line.dt.year) * 12
        + (df["origination_date"].dt.month - earliest_cr_line.dt.month)
    )

    # Only keep loans old enough that their full term has elapsed by the
    # snapshot date. Without this, a "resolved" loan from the last few
    # months of the snapshot is disproportionately either an early payoff
    # or a very-fast default -- the typical loan that takes 2-3 years to
    # play out simply isn't resolved yet and is silently excluded. That
    # censoring bias would otherwise concentrate in the most recent
    # vintages, exactly where the time-based train/test split puts the test
    # set.
    term_months = df["term"].str.strip().str.split().str[0].astype(int)
    maturity_period = df["origination_date"].dt.to_period("M") + term_months
    df = df[maturity_period <= snapshot_period]
    df["term_months"] = term_months.loc[df.index]

    # LC reports FICO as a 4-5 point range at origination; the midpoint is
    # the standard single-number summary.
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # "10+ years" -> 10, "< 1 year" -> 0, missing stays NaN (features.py
    # turns that into an explicit missing flag rather than guessing).
    df["emp_length_years"] = df["emp_length"].str.extract(r"(\d+)", expand=False).astype(float)
    df.loc[df["emp_length"].str.startswith("<", na=False), "emp_length_years"] = 0.0

    # zip_code arrives pre-anonymized to a 3-digit prefix (e.g. "190xx"). The
    # full zip3 is too sparse (~900 groups) for a readable disparate-impact
    # groupby, so bucket to its first digit -- ~10 groups, close to a US
    # Census division, and comparable cardinality to the synthetic A/B/C/D.
    df["region"] = df["zip_code"].str[0]

    df = df.rename(columns={
        "id": "loan_id",
        "annual_inc": "annual_income",
        "loan_amnt": "loan_amount",
        "open_acc": "open_accounts",
        "delinq_2yrs": "delinquencies_2yr",
        "purpose": "loan_purpose",
        "inq_last_6mths": "inquiries_6m",
        "total_acc": "total_accounts",
    })

    df = df.dropna(subset=[
        "annual_income", "dti", "credit_history_months", "open_accounts",
        "delinquencies_2yr", "region", "revol_util",
    ])

    if len(df) > LC_SAMPLE_SIZE:
        frac = LC_SAMPLE_SIZE / len(df)
        df = (
            df.groupby(df["origination_date"].dt.year, group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=LC_SAMPLE_SEED))
        )

    return df[STANDARD_COLUMNS].sort_values("origination_date").reset_index(drop=True)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        env_path = os.environ.get("CREDIT_RISK_RAW_DATA")
        if env_path:
            path = Path(env_path)
        else:
            candidates = sorted(DATA_DIR.glob("raw/*.csv"))
            if not candidates:
                raise FileNotFoundError(
                    "No CSV found in data/raw/. Run `python src/generate_synthetic.py` "
                    "first, or set CREDIT_RISK_RAW_DATA to point at a real dataset file."
                )
            path = candidates[0]

    header = list(pd.read_csv(path, nrows=0).columns)

    if _looks_like_lending_club(header):
        return _load_lending_club(path)

    df = pd.read_csv(path)
    df = df.rename(columns=RAW_COLUMNS)

    missing = set(STANDARD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Ingested data is missing expected columns: {missing}")

    df["origination_date"] = pd.to_datetime(df["origination_date"])
    return df[STANDARD_COLUMNS]


if __name__ == "__main__":
    df = load_raw()
    print(df.head())
    print(f"\nLoaded {len(df):,} rows from {DATA_DIR / 'raw'}")

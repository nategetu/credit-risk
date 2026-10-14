import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validate import validate_loans, clean_invalid_rows, ValidationError


def _valid_df(n=5):
    return pd.DataFrame({
        "loan_id": range(1, n + 1),
        "annual_income": [50000.0] * n,
        "loan_amount": [10000.0] * n,
        "dti": [20.0] * n,
        "fico_score": [700.0] * n,
        "revol_util": [50.0] * n,
        "term_months": [36] * n,
        "defaulted": [0, 1] * (n // 2) + [0] * (n % 2),
    })


def test_valid_data_passes():
    df = _valid_df()
    validate_loans(df)  # should not raise


def test_negative_income_fails():
    df = _valid_df()
    df.loc[0, "annual_income"] = -500
    with pytest.raises(ValidationError, match="negative annual_income"):
        validate_loans(df)


def test_impossible_dti_fails():
    df = _valid_df()
    df.loc[0, "dti"] = 150
    with pytest.raises(ValidationError, match="dti > 100"):
        validate_loans(df)


def test_duplicate_loan_id_fails():
    df = _valid_df()
    df.loc[1, "loan_id"] = df.loc[0, "loan_id"]
    with pytest.raises(ValidationError, match="duplicate loan_id"):
        validate_loans(df)


def test_clean_invalid_rows_removes_bad_data():
    df = _valid_df(n=10)
    df.loc[0, "annual_income"] = -1
    df.loc[1, "dti"] = 200
    cleaned = clean_invalid_rows(df)
    assert len(cleaned) == 8
    validate_loans(cleaned)  # should pass after cleaning


def test_impossible_fico_fails():
    df = _valid_df()
    df.loc[0, "fico_score"] = 900
    with pytest.raises(ValidationError, match="fico_score outside 300-850"):
        validate_loans(df)


def test_unknown_term_fails():
    df = _valid_df()
    df.loc[0, "term_months"] = 48
    with pytest.raises(ValidationError, match="term other than 36 or 60"):
        validate_loans(df)

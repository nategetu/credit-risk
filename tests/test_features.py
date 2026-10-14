import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import engineer_features, get_feature_matrix, FEATURE_COLUMNS


def _sample_df():
    return pd.DataFrame({
        "annual_income": [50000.0, 100000.0],
        "loan_amount": [10000.0, 5000.0],
        "credit_history_months": [24, 120],
        "open_accounts": [3, 8],
        "delinquencies_2yr": [0, 1],
        "dti": [15.0, 25.0],
        "loan_purpose": ["debt_consolidation", "medical"],
        "region": ["A", "B"],
        "fico_score": [700.0, 660.0],
        "term_months": [36, 60],
        "revol_util": [40.0, 85.0],
        "revol_bal": [5000.0, 20000.0],
        "inquiries_6m": [0, 2],
        "total_accounts": [10, 20],
        "emp_length_years": [5.0, np.nan],
        "home_ownership": ["RENT", "OWN"],
    })


def test_log_income_computed_correctly():
    df = engineer_features(_sample_df())
    assert np.isclose(df["log_income"].iloc[0], np.log(50000.0))


def test_loan_to_income_ratio():
    df = engineer_features(_sample_df())
    assert np.isclose(df["loan_to_income"].iloc[0], 10000.0 / 50000.0)


def test_purpose_dummies_created():
    df = engineer_features(_sample_df())
    assert df["purpose_debt_consolidation"].tolist() == [1, 0]
    assert df["purpose_medical"].tolist() == [0, 1]


def test_feature_matrix_has_all_expected_columns():
    X = get_feature_matrix(_sample_df())
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (2, len(FEATURE_COLUMNS))


def test_zero_income_does_not_produce_nan():
    df = _sample_df()
    df.loc[0, "annual_income"] = 0
    result = engineer_features(df)
    assert not result["log_income"].isnull().any()


def test_missing_emp_length_is_filled_and_flagged():
    df = engineer_features(_sample_df())
    assert df["emp_length_years"].tolist() == [5.0, 0.0]
    assert df["emp_length_missing"].tolist() == [0, 1]
    assert not get_feature_matrix(_sample_df()).isnull().any().any()


def test_term_and_home_ownership_flags():
    df = engineer_features(_sample_df())
    assert df["term_60"].tolist() == [0, 1]
    assert df["home_rent"].tolist() == [1, 0]
    assert df["home_mortgage"].tolist() == [0, 0]


def test_single_row_with_none_emp_length_is_numeric():
    # Mirrors an API request that omits the optional field.
    df = _sample_df().iloc[[0]].copy()
    df["emp_length_years"] = [None]
    X = get_feature_matrix(df)
    assert X.dtypes.map(lambda d: d.kind in "biuf").all()
    assert X["emp_length_missing"].iloc[0] == 1

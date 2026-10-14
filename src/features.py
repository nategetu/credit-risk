"""
Feature engineering. Kept separate from modeling so both train_logistic.py
and train_gbm.py consume the exact same feature set — a common bug source
is two models silently drifting apart on feature definitions.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "log_income", "loan_amount", "credit_history_months", "open_accounts",
    "delinquencies_2yr", "dti", "loan_to_income",
    "purpose_debt_consolidation", "purpose_credit_card",
    "purpose_home_improvement", "purpose_medical",
    "fico_score", "term_60", "revol_util", "revol_bal_to_income",
    "inquiries_6m", "total_accounts", "emp_length_years",
    "emp_length_missing", "home_rent", "home_mortgage",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_income"] = np.log(df["annual_income"].clip(lower=1))
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"].clip(lower=1)

    purpose_dummies = pd.get_dummies(df["loan_purpose"], prefix="purpose")
    for col in ["purpose_debt_consolidation", "purpose_credit_card",
                "purpose_home_improvement", "purpose_medical"]:
        if col in purpose_dummies.columns:
            df[col] = purpose_dummies[col].astype(int)
        else:
            df[col] = 0

    df["term_60"] = (df["term_months"] == 60).astype(int)
    df["revol_bal_to_income"] = df["revol_bal"] / df["annual_income"].clip(lower=1)

    # Missing employment length is filled with 0 AND flagged, so the
    # logistic regression (which can't take NaN) can still use the rows,
    # and both models can learn whether "missing" itself carries risk.
    # to_numeric matters for single-row API requests: an omitted value
    # arrives as None, which pandas stores as an object column that
    # LightGBM refuses.
    emp = pd.to_numeric(df["emp_length_years"], errors="coerce")
    df["emp_length_missing"] = emp.isna().astype(int)
    df["emp_length_years"] = emp.fillna(0).astype(float)

    # OWN (and LC's handful of OTHER/NONE/ANY) is the reference category.
    df["home_rent"] = (df["home_ownership"] == "RENT").astype(int)
    df["home_mortgage"] = (df["home_ownership"] == "MORTGAGE").astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df)
    return df[FEATURE_COLUMNS]


if __name__ == "__main__":
    from ingest import load_raw
    from validate import clean_invalid_rows

    df = clean_invalid_rows(load_raw())
    X = get_feature_matrix(df)
    print(X.head())
    print(f"\nFeature matrix shape: {X.shape}")

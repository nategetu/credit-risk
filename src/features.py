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

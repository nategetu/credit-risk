"""
Phase 3 validation: fail LOUDLY on bad data, before it ever reaches a model.

This is deliberately dependency-free (no Great Expectations) so you can see
exactly what a validation framework does under the hood. Once this feels
natural, swapping in Great Expectations or Pandera is a good next step —
they give you the same idea with much less boilerplate at scale.
"""
import pandas as pd


class ValidationError(Exception):
    """Raised when the dataset fails a hard data-quality check."""


def validate_loans(df: pd.DataFrame) -> pd.DataFrame:
    errors = []

    if df["annual_income"].lt(0).any():
        n = int(df["annual_income"].lt(0).sum())
        errors.append(f"{n} rows have negative annual_income")

    if df["dti"].gt(100).any():
        n = int(df["dti"].gt(100).sum())
        errors.append(f"{n} rows have dti > 100 (impossible)")

    if df["dti"].lt(0).any():
        n = int(df["dti"].lt(0).sum())
        errors.append(f"{n} rows have negative dti")

    if df["loan_amount"].le(0).any():
        n = int(df["loan_amount"].le(0).sum())
        errors.append(f"{n} rows have non-positive loan_amount")

    if not df["fico_score"].between(300, 850).all():
        n = int((~df["fico_score"].between(300, 850)).sum())
        errors.append(f"{n} rows have fico_score outside 300-850 (impossible)")

    if df["revol_util"].lt(0).any():
        n = int(df["revol_util"].lt(0).sum())
        errors.append(f"{n} rows have negative revol_util")

    if not df["term_months"].isin([36, 60]).all():
        n = int((~df["term_months"].isin([36, 60])).sum())
        errors.append(f"{n} rows have a term other than 36 or 60 months")

    if df["defaulted"].isin([0, 1]).all() is False:
        errors.append("defaulted column has values outside {0, 1}")

    if df["loan_id"].duplicated().any():
        n = int(df["loan_id"].duplicated().sum())
        errors.append(f"{n} duplicate loan_id values found")

    null_counts = df[["annual_income", "dti", "defaulted"]].isnull().sum()
    if null_counts.any():
        errors.append(f"Null values found in required columns: {null_counts[null_counts > 0].to_dict()}")

    if errors:
        raise ValidationError(
            "Data validation failed with the following issues:\n  - " + "\n  - ".join(errors)
        )

    return df


def clean_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    A real pipeline has to decide: reject the whole batch, or quarantine bad
    rows and proceed? This function demonstrates the "quarantine" pattern —
    use it explicitly (don't silently call this before validate_loans, or
    you'll never notice upstream data quality regressing).
    """
    before = len(df)
    df = df[df["annual_income"] >= 0]
    df = df[df["dti"].between(0, 100)]
    df = df[df["loan_amount"] > 0]
    df = df[df["fico_score"].between(300, 850)]
    df = df[df["revol_util"] >= 0]
    df = df[df["term_months"].isin([36, 60])]
    after = len(df)
    if after < before:
        print(f"Quarantined {before - after} invalid rows ({(before - after) / before:.2%} of data)")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from ingest import load_raw

    df = load_raw()
    try:
        validate_loans(df)
        print("Validation passed.")
    except ValidationError as e:
        print(f"Validation failed as expected on raw synthetic data:\n{e}")
        print("\nRunning clean_invalid_rows() and re-validating...")
        df_clean = clean_invalid_rows(df)
        validate_loans(df_clean)
        print("Validation passed after quarantine.")

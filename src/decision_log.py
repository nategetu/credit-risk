"""
Durable record of every /predict decision.

Two distinct reasons this exists, not one:
  1. Compliance -- a denial has to come with a retrievable record and
     principal reasons, mirroring adverse-action-notice requirements
     (see api.py's docstring). A decision the service can't produce a
     record of is a compliance gap, not just a missing log line -- so
     the write happens *before* the response is returned, and a failed
     write fails the request. This is deliberate, not an oversight
     (see MODEL_CARD.md section 10).
  2. Monitoring -- this table is also the "comparison" population a
     future Airflow task reads to compute PSI against the model's
     training-time reference (see monitor.py). The engineered feature
     values are stored exactly as computed at decision time, not
     re-derived later -- if engineer_features() ever changes, that
     shouldn't retroactively alter what a past decision is compared
     against.

Uses SQLAlchemy Core rather than raw sqlite3 so the *interface* is
already cloud-portable: pointing this at a managed Postgres instance in
a real deployment is a CREDIT_RISK_DB_URL change, not a rewrite.
"""
import os
from pathlib import Path

from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    String, Float, DateTime, JSON,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_DB_URL = f"sqlite:///{DATA_DIR / 'decisions.db'}"
DB_URL = os.environ.get("CREDIT_RISK_DB_URL", DEFAULT_DB_URL)

engine = create_engine(DB_URL)
metadata = MetaData()

decisions = Table(
    "decisions",
    metadata,
    Column("application_id", String, primary_key=True),
    Column("timestamp", DateTime, nullable=False),
    # Raw applicant-submitted fields -- the application actually on file.
    Column("annual_income", Float, nullable=False),
    Column("loan_amount", Float, nullable=False),
    Column("credit_history_months", Float, nullable=False),
    Column("open_accounts", Float, nullable=False),
    Column("delinquencies_2yr", Float, nullable=False),
    Column("dti", Float, nullable=False),
    Column("loan_purpose", String, nullable=False),
    Column("fico_score", Float, nullable=False),
    Column("term_months", Float, nullable=False),
    Column("revol_util", Float, nullable=False),
    Column("revol_bal", Float, nullable=False),
    Column("inquiries_6m", Float, nullable=False),
    Column("total_accounts", Float, nullable=False),
    Column("emp_length_years", Float, nullable=True),  # optional on the application
    Column("home_ownership", String, nullable=False),
    # Engineered features exactly as fed to the model (FEATURE_COLUMNS in
    # features.py) -- the population PSI will compare against training.
    Column("log_income", Float, nullable=False),
    Column("loan_to_income", Float, nullable=False),
    Column("purpose_debt_consolidation", Float, nullable=False),
    Column("purpose_credit_card", Float, nullable=False),
    Column("purpose_home_improvement", Float, nullable=False),
    Column("purpose_medical", Float, nullable=False),
    Column("term_60", Float, nullable=False),
    Column("revol_bal_to_income", Float, nullable=False),
    Column("emp_length_missing", Float, nullable=False),
    Column("home_rent", Float, nullable=False),
    Column("home_mortgage", Float, nullable=False),
    # Decision + explanation.
    Column("default_probability", Float, nullable=False),
    Column("decision", String, nullable=False),
    Column("decision_threshold", Float, nullable=False),
    Column("base_log_odds", Float, nullable=False),
    Column("predicted_log_odds", Float, nullable=False),
    Column("principal_reasons", JSON, nullable=False),
    Column("feature_contributions", JSON, nullable=False),
)


def init_db() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    metadata.create_all(engine)


def log_decision(record: dict) -> None:
    """Insert one decision row. Raises on failure -- callers must not
    swallow this; a decision without a durable record shouldn't ship."""
    with engine.begin() as conn:
        conn.execute(decisions.insert().values(**record))

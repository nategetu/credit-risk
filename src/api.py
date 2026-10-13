"""
FastAPI service: score a single loan application and explain the score.

Mirrors real adverse-action-notice requirements -- a denial (or a worse
price) has to come with the *principal reasons*, not just a number. See
README.md for the full design discussion this implements.

Run: uvicorn src.api:app --reload --app-dir src
"""
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

import decision_log
from features import engineer_features, FEATURE_COLUMNS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "gbm_model.txt"

# How many risk-increasing features to surface as "principal reasons" --
# an arbitrary but standard choice (mirrors classic credit-score reason-code
# conventions); in a regulated shop this number is a compliance call, not
# an engineering one.
TOP_N_REASONS = 4

# The approve/deny cutoff is a business policy decision, not a modeling one
# (see README) -- it depends on the relative dollar cost of the two ways
# this can go wrong, which this service has no way to know on its own.
# Deliberately configured via env vars rather than hardcoded, and deliberately
# NOT settable per-request: an applicant shouldn't be able to supply their own
# approval bar. Defaults to equal costs (a 0.5 threshold) if unset, which is
# almost certainly not the right number for real use -- it's a neutral
# fallback, not a recommendation.
COST_FALSE_DENY = float(os.environ.get("CREDIT_RISK_COST_FALSE_DENY", 1.0))
COST_FALSE_APPROVE = float(os.environ.get("CREDIT_RISK_COST_FALSE_APPROVE", 1.0))

# Loaded once at server startup (see `lifespan` below), not per-request.
ml_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    decision_log.init_db()
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    ml_state["booster"] = booster
    # tree_path_dependent perturbation (the default with no `data=` arg)
    # uses statistics already stored in the trees themselves, so no
    # background dataset needs to be loaded into the server process.
    ml_state["explainer"] = shap.TreeExplainer(booster)
    # Cost-minimizing cutoff: deny when P(default) > C_deny / (C_deny + C_approve).
    ml_state["decision_threshold"] = COST_FALSE_DENY / (COST_FALSE_DENY + COST_FALSE_APPROVE)
    yield
    ml_state.clear()


app = FastAPI(title="Credit Risk Scoring API", lifespan=lifespan)


class LoanApplication(BaseModel):
    annual_income: float = Field(..., ge=0)
    loan_amount: float = Field(..., gt=0)
    credit_history_months: float = Field(..., ge=0)
    open_accounts: int = Field(..., ge=0)
    delinquencies_2yr: int = Field(..., ge=0)
    dti: float = Field(..., ge=0, le=100)
    loan_purpose: str


class FeatureContribution(BaseModel):
    feature: str
    log_odds_contribution: float


class PredictResponse(BaseModel):
    application_id: str
    timestamp: datetime
    default_probability: float
    decision: Literal["approve", "deny"]
    decision_threshold: float
    base_log_odds: float
    predicted_log_odds: float
    applicant_values: LoanApplication
    principal_reasons: list[FeatureContribution]
    feature_contributions: list[FeatureContribution]


@app.post("/predict", response_model=PredictResponse)
def predict(application: LoanApplication) -> PredictResponse:
    booster = ml_state["booster"]
    explainer = ml_state["explainer"]
    decision_threshold = ml_state["decision_threshold"]

    raw_row = pd.DataFrame([application.model_dump()])
    engineered = engineer_features(raw_row)
    X = engineered[FEATURE_COLUMNS]

    default_probability = float(booster.predict(X)[0])
    predicted_log_odds = float(booster.predict(X, raw_score=True)[0])

    explanation = explainer(X)
    base_log_odds = float(explanation.base_values[0])
    contributions = [
        FeatureContribution(feature=feat, log_odds_contribution=float(val))
        for feat, val in zip(FEATURE_COLUMNS, explanation.values[0])
    ]
    contributions.sort(key=lambda c: c.log_odds_contribution, reverse=True)
    principal_reasons = [c for c in contributions if c.log_odds_contribution > 0][:TOP_N_REASONS]
    decision = "deny" if default_probability > decision_threshold else "approve"

    application_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)
    engineered_row = engineered.iloc[0]

    # Durable record before the response goes out -- a decision this
    # service can't produce a record of shouldn't ship (see decision_log.py).
    decision_log.log_decision({
        "application_id": application_id,
        "timestamp": timestamp,
        "annual_income": application.annual_income,
        "loan_amount": application.loan_amount,
        "credit_history_months": application.credit_history_months,
        "open_accounts": application.open_accounts,
        "delinquencies_2yr": application.delinquencies_2yr,
        "dti": application.dti,
        "loan_purpose": application.loan_purpose,
        "log_income": float(engineered_row["log_income"]),
        "loan_to_income": float(engineered_row["loan_to_income"]),
        "purpose_debt_consolidation": float(engineered_row["purpose_debt_consolidation"]),
        "purpose_credit_card": float(engineered_row["purpose_credit_card"]),
        "purpose_home_improvement": float(engineered_row["purpose_home_improvement"]),
        "purpose_medical": float(engineered_row["purpose_medical"]),
        "default_probability": default_probability,
        "decision": decision,
        "decision_threshold": decision_threshold,
        "base_log_odds": base_log_odds,
        "predicted_log_odds": predicted_log_odds,
        "principal_reasons": [r.model_dump() for r in principal_reasons],
        "feature_contributions": [c.model_dump() for c in contributions],
    })

    return PredictResponse(
        application_id=application_id,
        timestamp=timestamp,
        default_probability=default_probability,
        decision=decision,
        decision_threshold=decision_threshold,
        base_log_odds=base_log_odds,
        predicted_log_odds=predicted_log_odds,
        applicant_values=application,
        principal_reasons=principal_reasons,
        feature_contributions=contributions,
    )

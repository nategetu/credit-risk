# Model Card: Consumer Credit Default Model (GBM)

This is the documentation a model risk validator would expect to find before
reviewing this model — what it does, what it's for, what was decided and why,
and what isn't done yet. Structured around SR 11-7 model risk management
expectations: intended use, data, methodology, performance, key decisions,
monitoring, and known limitations.

## 1. Model overview

- **What's served**: `src/train_gbm.py` trains a LightGBM classifier
  (`models/gbm_model.txt`), predicting probability of default on a consumer
  loan. This is the model `src/api.py`'s `/predict` endpoint actually calls.
- **A second model exists but is not served**: `src/train_logistic.py`
  trains a logistic regression for statistical *inference* (coefficient
  significance, confidence intervals) — it answers "which factors are
  associated with default and how confident are we," not "what should this
  specific applicant be scored." It is not wired into the API and was never
  intended to be; regularizing or otherwise touching it for predictive
  performance would break the classical-inference guarantees it exists for.
- **Features** (`src/features.py`, 21 total):
  - *Application*: `log_income`, `loan_amount`, `loan_to_income`, `dti`,
    `term_60`, one-hot loan purpose (`debt_consolidation`, `credit_card`,
    `home_improvement`, `medical`), `emp_length_years` +
    `emp_length_missing`, and home ownership (`home_rent`,
    `home_mortgage`; OWN is the reference).
  - *Credit bureau*: `fico_score` (midpoint of LC's reported range),
    `credit_history_months`, `open_accounts`, `total_accounts`,
    `delinquencies_2yr`, `inquiries_6m`, `revol_util`,
    `revol_bal_to_income`.
  - *Deliberately excluded*: LC's `grade`, `sub_grade`, and `int_rate`
    (LC's own risk model output — using them would mostly re-learn LC's
    pricing), public records and income verification (no measurable lift
    in a leave-one-out comparison), and `mort_acc` (100% missing for
    2007–2011 vintages, so the time-based train set would be mostly NaN
    while the test set is fully populated).
  - All are point-in-time
  attributes of the applicant/application, captured once, at the moment of
  application — none are ongoing or externally-sourced (e.g. no macro
  indicators).

## 2. Intended use / scope

- **In scope**: scoring probability of default for **unsecured consumer
  personal loans**, for the purposes of an approve/deny decision, using
  Lending-Club-shaped applicant data.
- **Out of scope — do not use for**:
  - Any other loan product (mortgage, auto, secured lending). The model has
    never seen a secured-loan feature (LTV, collateral value) and default
    behavior for secured/long-tenor products is not comparable to unsecured
    personal loans.
  - **Pricing / interest-rate setting.** The model was trained and validated
    to predict default probability, not to set a price. Using its output for
    a purpose it wasn't validated for is misuse even if the score itself is
    accurate.
  - Any population outside typical Lending Club personal-loan applicants
    (see Data section — training data ends 2015).

## 3. Data

- **Source**: Lending Club accepted-loans data (`accepted_2007_to_2018Q4.csv.gz`),
  ingested via `src/ingest.py`.
- **Target**: `loan_status` resolved outcomes only (`Fully Paid`, `Charged
  Off`, `Default`) — loans still `Current`/`Late` are dropped, since their
  eventual outcome is unknown.
- **Maturity/censoring cutoff**: only loans whose full term (36 or 60 months)
  had already elapsed by the dataset's snapshot date are included. Without
  this, the most recent vintages would be biased toward early payoffs and
  fast defaults (the only outcomes that could have resolved yet), landing
  disproportionately in the test set under a time-based split. **This caps
  the usable date range at 2007–2015.** Any application scored today is
  being scored by a model trained on applicant behavior up to a decade old.
- **Sampling**: 200k rows, stratified by origination year.

## 4. Methodology

- **Split**: chronological three-way split (`time_based_split` in
  `train_gbm.py`) — train 70% / calibration 15% / test 15%, ordered by
  `origination_date`, no shuffling. A random split would leak future loan
  behavior into training a model meant to score past-only information.
- **Model**: LightGBM, `learning_rate=0.05`, `min_child_samples=30`,
  `num_leaves=31`, `n_estimators=300` (see `train_gbm.py` for full params).

## 5. Performance (current, from `reports/pipeline_metrics.json`)

| Metric | Raw | Calibrated |
|---|---|---|
| AUC | 0.6639 | 0.6639 |
| Brier score | 0.1221 | 0.1226 |
| Calibration error | **0.0141** | 0.0197 |

Adding the credit-bureau features raised test AUC from 0.612 (original
11 features) to 0.664. FICO alone accounts for the largest single gain
(+0.04 on its own); inquiries, revolving utilization, employment length,
and term each add smaller increments.

## 6. Calibration decision — what was found, and why raw ships

- **Both raw and calibrated metrics are computed and saved on every training
  run** (`gbm_metrics.json`, `pipeline_metrics.json`). The pipeline does
  **not** auto-select between them — this is deliberate: which one ships is
  a model-validation judgment call for a human, not something a script
  should silently decide. (Auto-selecting the better-looking metric on each retrain would
  reproduce the same "winner's curse" selection bias as picking the best of
  several CV folds instead of averaging them, and would remove human
  accountability for a decision that affects real approve/deny outcomes.)
- **Finding on real data**: calibration (Platt scaling, fit on the held-out
  calibration slice) made calibration error *worse* — 0.0141 raw vs. 0.0197
  calibrated on the current model (0.0073 vs. 0.0129 on the original
  11-feature model; the finding has held across both).
- **Why**: `learning_rate=0.05` + `min_child_samples=30` already constrain
  GBM's typical overconfidence mechanism (boosting keeps chasing a residual
  that's only zero at 0/1), so there was little systematic bias left for
  Platt scaling to correct. The correction only added estimation variance
  from its own, smaller (15%) calibration slice — a real example of
  bias-variance tradeoff applying even to a 2-parameter correction.
- **Decision**: `api.py` serves the **raw** GBM score. `pipeline.py`'s CI
  gates check raw metrics accordingly, since that's what's actually served.
- **What would trigger revisiting this**: `pipeline.py` already prints a
  runtime note if calibrated beats raw on any given run. If that holds up
  **across multiple retrains**, not just one run (a single win could still
  be noise from the small calibration slice), wiring the calibrator into
  `api.py`'s `/predict` should be reconsidered.

## 7. Decision threshold / business policy assumptions

- `api.py` computes `decision_threshold = C_deny / (C_deny + C_approve)`
  from `CREDIT_RISK_COST_FALSE_DENY` / `CREDIT_RISK_COST_FALSE_APPROVE` env
  vars, defaulting to **1.0 / 1.0 (equal cost)** if unset.
- **This default is a neutral placeholder, not a validated business
  assumption.** The relative real-dollar cost of wrongly denying a
  creditworthy applicant vs. wrongly approving one who defaults is a lending
  policy decision this service has no way to know on its own, and nobody has
  supplied real costs yet. Any deployment of this model needs an actual
  cost ratio, sourced from business/finance, before the decision threshold
  means anything.

## 8. Human oversight / decisioning

- **No case-level human review exists.** `api.py`'s `/predict` computes
  `decision` (approve/deny) and returns it directly — there is no step where
  a person reviews an individual application before the decision reaches the
  applicant. This is a fully automated decision at the individual-case
  level, which (since no one is
  positioned to catch a structurally biased or wrong call before it affects
  a specific person) puts it in the *higher*, not lower, validation-scrutiny
  tier.
- **Model-level oversight does exist** — the raw-vs-calibrated ship decision
  (Section 6) and the cost-ratio configuration (Section 7) are both human
  decisions, made once at the model/policy level, not per-case.

## 9. Fairness

- `train_logistic.py`'s `disparate_impact_check(df, model, group_col=
  "region")` compares predicted risk across regional groups. It is a
  standalone function, run manually via `train_logistic.py`'s `__main__` —
  **it is not wired into `pipeline.py`'s CI gates or into `api.py`.** No
  fairness threshold currently blocks a bad model from shipping the way the
  AUC/calibration gates do.

## 10. Ongoing monitoring

- **Live decision logging** — every `/predict` call is durably recorded in
  `src/decision_log.py` (SQLite locally via `CREDIT_RISK_DB_URL`, Postgres/
  cloud-portable by swapping that URL — see the module docstring) *before*
  the response is returned; a failed write fails the request. This exists
  for two distinct reasons, not one: (1) compliance — a denial needs a
  retrievable record with reasons, mirroring adverse-action-notice
  requirements, and (2) it's the future source of the "comparison"
  population for PSI (below). Engineered feature values are stored exactly
  as computed at decision time, not re-derived later, so a future change to
  `engineer_features()` can't retroactively alter what a past decision gets
  compared against.
- **Population drift (PSI)** — `src/monitor.py`'s `population_stability_index()`
  is implemented and validated against real drift (comparing 2007-09 vs.
  2014-15 vintages; `dti` PSI = 0.56, well past the "significant shift"
  threshold of 0.25). Live requests are now logged (above), but **PSI is
  still not actually computed against them anywhere** — no scheduled job
  reads the `decisions` table yet. Next step: an Airflow task that reads
  the table, compares it against the model's *training-time* population
  (fixed at whatever `train_gbm.py` used to fit the currently-served model —
  deliberately not a sliding "recent" window, which would hide slow drift
  the same way comparing only adjacent months would), and reports PSI per
  `monitored_columns`.
  - **Known blind spot, by design, not a bug**: `dti` arrives pre-computed
    from Lending Club — the raw debt/income figures behind it were never
    captured, so if they drifted in a way that canceled out in the ratio,
    nothing here would catch it. `loan_to_income` doesn't have this problem,
    since both `loan_amount` and `log_income` are tracked independently
    alongside it.
- **Concept drift / backtesting** — not implemented. No mechanism exists yet
  to compare predicted default probability against realized outcomes by
  vintage/risk band once loans season. This is the monitoring gap that would
  catch the model's predictions silently going wrong even when input
  features show no drift at all (e.g. a macro shock that changes the
  feature→outcome relationship without changing the feature distributions).
  The `decisions` table has what this needs on the prediction side; it has
  no way yet to join back to an eventual loan outcome.
- **Fast leading indicators (roll rate / cure rate)** — not implemented.
  Full backtesting requires waiting out a loan's 36-60 month term; a
  roll-rate-based early warning (aggregate delinquency-transition rates by
  vintage, checked within weeks) does not yet exist in this repo.

## 11. Known limitations (consolidated)

1. Training data ends 2015 — model has never seen post-2015 economic
   conditions or applicant behavior.
2. Decision threshold cost ratio is an unvalidated placeholder (Section 7).
3. No case-level human review of individual decisions (Section 8).
4. No fairness gate in CI (Section 9).
5. No live drift monitoring or backtesting wired into production (Section 10).
6. AUC (0.66) is moderate — meaningful rank-ordering signal, but below
   what's achievable with LC's own grade/interest rate (excluded on
   purpose, see Section 1) or richer bureau data (full tradeline history,
   payment-level behavior) that LC's public file doesn't include.

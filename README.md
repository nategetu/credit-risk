# Consumer Credit Risk Pipeline

A learning project that builds a consumer loan default prediction system the way
a regulated fintech risk team would: with statistical inference, calibrated
modeling, data validation, versioning, and CI/CD gates — not a one-off notebook.

## Why this exists

Data analyst work builds strong SQL/EDA/reporting skills but usually skips:
inferential statistics with correctly interpreted uncertainty, predictive
modeling with proper validation, pipeline engineering, and CI/CD. This project
hits all four using a consumer lending dataset, where the rigor isn't
academic — it's what real credit risk models require.

## Dataset

This pipeline is built against the [Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
(`accepted_2007_to_2018Q4.csv.gz`) from Kaggle, with `make synthetic-data`
available as a fallback so the pipeline runs end-to-end without any external
dependency (and is what CI uses).

To use the real dataset:

1. `kaggle datasets download -d wordsforthewise/lending-club -f accepted_2007_to_2018Q4.csv.gz -p data/raw`
   (requires a Kaggle account + API token at `~/.kaggle/access_token` or
   `~/.kaggle/kaggle.json`). We only need the *accepted* loans file — the
   *rejected* one has no outcome label, since rejected applications were
   never originated.
2. Point the pipeline at it: `CREDIT_RISK_RAW_DATA=data/raw/accepted_2007_to_2018Q4.csv.gz make pipeline`
   (or `python src/train_logistic.py`, etc., with the same env var). `src/ingest.py`
   sniffs the file's header and automatically runs the Lending Club-specific
   transform — no code changes needed for this dataset. Adding a *different*
   real dataset (e.g. Home Credit) means adding a new `_looks_like_x` /
   `_load_x` pair there.

The Lending Club transform in `ingest.py` makes a few non-obvious calls
worth knowing about if you extend it:

- **Target definition**: `loan_status` has 7 values; only `Fully Paid`,
  `Charged Off`, and `Default` are resolved outcomes. Loans still `Current`
  or `Late` are dropped rather than labeled, since their eventual outcome
  is unknown.
- **Maturity/censoring cutoff**: a loan only gets included if its full term
  (36 or 60 months) has elapsed by the dataset's snapshot date. Otherwise
  the most recent vintages are systematically biased toward early payoffs
  and fast defaults (the only ones that could have already resolved),
  which lands right in the test set under a time-based split and quietly
  wrecks calibration. This cuts the usable date range to 2007-2015.
- **`region`** is the first digit of the zip3 prefix Lending Club already
  ships (`"190xx"` → `"1"`) — roughly a US Census division, ~10 groups,
  comparable cardinality to the synthetic data's A/B/C/D.
- Data is sampled to 200k rows (stratified by origination year) for fast
  iteration; the full filtered dataset is several times larger.

## Project structure

```
consumer-credit-risk-pipeline/
├── data/
│   ├── raw/              # untouched source data (gitignored)
│   └── processed/        # validated, feature-engineered data (gitignored)
├── models/                # trained model artifacts (gitignored)
├── reports/                # evaluation reports, calibration plots
├── src/
│   ├── generate_synthetic.py   # fake data generator (Phase 0)
│   ├── ingest.py                # load + standardize raw data
│   ├── validate.py              # schema/range checks (Great Expectations style)
│   ├── features.py              # feature engineering
│   ├── train_logistic.py        # Phase 1: inference-focused logistic regression
│   ├── train_gbm.py             # Phase 2: gradient boosting model
│   ├── evaluate.py              # calibration, AUC, fairness slice metrics
│   └── pipeline.py              # orchestrates the full DAG end-to-end
├── tests/
│   ├── test_validate.py
│   └── test_features.py
├── .github/workflows/ci.yml     # Phase 4: CI gates
├── requirements.txt
└── Makefile
```

## The four phases (do them in order)

### Phase 1 — Inference (`src/train_logistic.py`)
Run a logistic regression and actually read the output: coefficient signs,
confidence intervals, p-values, VIF for multicollinearity. The goal is to be
able to say *why* the model thinks someone is risky, not just *that* it
flagged them. This also runs a basic disparate-impact check across a
demographic-proxy column (zip-code-derived region in the synthetic data) —
a real regulatory concern in lending, and a good habit to build now.

### Phase 2 — Modeling (`src/train_gbm.py`)
A LightGBM/XGBoost model with a **time-based** train/test split (loans
"originated" earlier train, later ones test — no shuffling, no leakage).
Includes hyperparameter search and, critically, **calibration** checking:
a model that says "15% default risk" should be right about 15% of the time,
which matters more than raw AUC for a model that prices risk.

### Phase 3 — Pipeline (`src/pipeline.py`, `src/validate.py`)
Ties ingest → validate → feature-engineer → train → evaluate into one script
with explicit, fail-loud steps. Bad upstream data (negative income, DTI over
100%) should crash the pipeline, not silently corrupt a model.

### Phase 4 — CI/CD (`.github/workflows/ci.yml`)
GitHub Actions runs unit tests on every push, then runs the pipeline against
synthetic data and fails the build if:
- data validation fails
- test-set AUC drops below a threshold
- calibration error exceeds a threshold

This is the gate that would, in a real org, block a bad model from shipping.

## Getting started

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# generate fake data so you can run everything immediately
python src/generate_synthetic.py

# run the full pipeline
python src/pipeline.py

# run tests (what CI runs)
pytest tests/ -v
```

## Stretch goals (once the four phases work)

- [x] Wire in the real Lending Club dataset (see Dataset section above).
- [x] Wrap the trained model in a FastAPI endpoint (`src/api.py`) returning a
  default probability, an approve/deny decision, and SHAP-based principal
  reasons — mirrors real **adverse action notice** requirements. Run with
  `uvicorn api:app --reload --app-dir src` (docs at `/docs`). The approve/deny
  cutoff is a cost-based threshold, configurable via `CREDIT_RISK_COST_FALSE_DENY`
  / `CREDIT_RISK_COST_FALSE_APPROVE` env vars (deliberately a business-policy
  input, not a hardcoded constant):

  ```
  deny when P(default) > C_false_deny / (C_false_deny + C_false_approve)
  ```

  the expected-cost-minimizing cutoff given the dollar cost of each mistake
  (denying a good applicant vs. approving one who defaults). Equivalent to
  the expected-profit breakeven point `R / (R + L)` for per-loan revenue `R`
  and default loss `L` — the same threshold either way you derive it. Real
  lenders generalize this to *many* risk buckets with their own price rather
  than one global approve/deny line (this is literally what Lending Club's
  own `grade`/`sub_grade` columns are), and layer on real-world complications
  this simple version ignores: adverse selection from raising rates on risky
  buckets, revenue lost to early payoffs, correlated portfolio-level default
  risk in a downturn, and fair-lending constraints that can override a purely
  profit-maximizing cutoff.
- [x] Add a drift monitor (`src/monitor.py`) comparing feature distributions via
  population stability index. Detects covariate shift (the applicant population
  changing) without needing outcome labels, which for a loan take 36-60 months
  to materialize — a leading indicator, not a substitute for eventually checking
  real accuracy. Deliberately scoped as a standalone function proven against real
  drift already in the Lending Club data (early vs. late vintages), not wired into
  `api.py` with live request logging — that's persistence/orchestration scope,
  not a statistics one. Run: `CREDIT_RISK_RAW_DATA=data/raw/accepted_2007_to_2018Q4.csv.gz python src/monitor.py`.
- [x] Swap `src/pipeline.py`'s ad-hoc orchestration for Airflow
  (`airflow_home/dags/credit_risk_dag.py`) — real per-task tracking, retries,
  and dependency semantics instead of one script that's all-or-nothing.

  **Setup** (two separate venvs on purpose — Airflow's pinned dependencies
  would otherwise risk conflicting with pandas/lightgbm/etc.):
  ```bash
  python3 -m venv venv_airflow
  source venv_airflow/bin/activate
  pip install "apache-airflow==3.3.0" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.3.0/constraints-3.13.txt"
  export AIRFLOW_HOME=$(pwd)/airflow_home
  airflow standalone   # bundles scheduler + webserver + a SQLite metadata DB;
                        # local/dev only, not for production
  # UI at localhost:8080; admin password printed to the console / saved to
  # airflow_home/simple_auth_manager_passwords.json.generated
  # New DAGs are paused by default: airflow dags unpause credit_risk_pipeline
  ```

  Each of the five tasks runs via `@task.external_python(python=".../venv/bin/python")`
  — the DAG definition itself runs in Airflow's venv, but each task's actual
  work (which needs pandas/lightgbm/shap/etc.) runs in *this project's*
  existing venv, in a separate subprocess. That has one real, non-obvious
  consequence: only the function body (plus explicit arguments) crosses that
  boundary — module-level constants and top-of-file imports do **not**, since
  the task never actually runs inside this file's process. Every import, and
  every constant a task needs, has to live inside the task function itself.

  Intermediate output between tasks is persisted to `data/processed/*.csv`
  (XCom carries the file *path*, not the DataFrame — XCom is for small
  values). This is what makes the core payoff real and checkable, not just
  theoretical: clearing a single downstream task (e.g. `evaluate_task`) and
  re-running it reuses the four upstream tasks' already-persisted output
  untouched, rather than re-ingesting/re-validating/re-training from scratch
  — proven directly against a real run during this build, not just asserted.
- [x] More unit tests for `train_gbm.py` and `train_logistic.py`
  (`tests/test_train_gbm.py`, `tests/test_train_logistic.py`). Testing ML
  code needs a different approach than testing pure functions like
  `features.py`: deterministic parts (`time_based_split`, `check_multicollinearity`)
  get exact hand-computed assertions same as before; functions that only
  depend on a model's `.predict()`/`.params` (`evaluate_model`,
  `summarize_coefficients`, `disparate_impact_check`) get a fake stand-in
  model with hand-picked outputs, which turns their downstream math exactly
  hand-verifiable too; actual model fitting (`train_model`, `fit_logistic`)
  gets property-based tests instead (deterministic given a fixed seed,
  recovers a known strong synthetic signal, doesn't fabricate signal from
  pure noise) since there's no way to hand-verify a trained model's exact
  output values.

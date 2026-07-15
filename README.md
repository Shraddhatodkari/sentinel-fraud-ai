# Sentinel — Agentic Cross-Silo Financial Crime Investigation Platform

Financial institutions investigate fraud, money laundering (AML), and
sanctions violations using **separate systems, run by separate teams**.
This means the same customer can be independently flagged by the Fraud
team, the AML team, and Sanctions screening — with no one connecting
the dots, because each team only sees its own alerts.

**Sentinel** solves two layered problems:
1. **Alert fatigue** — most individual fraud alerts are false positives,
   burying analysts in manual review (the original problem).
2. **Siloed investigations** — even when an alert is genuine, the
   institution often misses that the same account or beneficiary is
   ALSO flagged in AML or Sanctions systems, because those systems don't
   talk to each other.

Sentinel investigates each alert like a human analyst would, then links
it across silos: it checks whether the account or beneficiary involved
is also flagged elsewhere in the institution, and routes the case to
every team that needs to know — not just the one that happened to catch
it first.

## Problem this solves

- Industry-wide, false-positive rates on fraud alerts commonly run 90%+.
- Fraud, AML, and Sanctions teams typically operate on separate systems
  with no shared case view — a well-documented operational gap at most
  institutions, not a hypothetical one.
- Coordinated fraud (rings sharing devices/IPs/beneficiaries) is often
  invisible to per-transaction, per-team scoring.
- A customer flagged by two teams independently should be treated very
  differently than one flagged by only one — but today, that connection
  is usually missed entirely.

## Architecture

```
TransactionInput
      │
      ▼
┌─────────────────────┐
│  Investigator Agent  │  gathers risk scores (XGBoost + Isolation Forest),
│                       │  SHAP explanations, and network findings (NetworkX)
└──────────┬───────────┘
           │  InvestigationContext
           ▼
┌─────────────────────┐
│  Disposition Agent    │  (PydanticAI + LLM) decides AUTO_CLEAR vs ESCALATE,
│                       │  writes rationale + analyst summary
└──────────┬───────────┘
           │  DispositionResult
           ▼
┌─────────────────────┐
│   Case Linker          │  checks if this account/beneficiary is ALSO
│  (NetworkX graph)      │  flagged in AML alerts or Sanctions hits —
│                       │  the cross-silo connection teams normally miss
└──────────┬───────────┘
           │  CrossSiloFindings
           ▼
┌─────────────────────┐
│  Case Routing Agent    │  (PydanticAI + LLM) decides which team(s) —
│                       │  Fraud / AML / Sanctions — should own the case
└──────────┬───────────┘
           │  CaseRoutingResult
           ▼
┌─────────────────────┐
│   Audit Logger         │  immutable record in MongoDB Atlas
└──────────┬───────────┘
           │
           ▼
   LangSmith traces the full reasoning chain end-to-end
           │
           ▼
┌─────────────────────┐
│  Analyst Feedback     │  analyst confirms/overrides decision via
│  (feedback loop)      │  /alerts/feedback → tracked separately
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│  Drift Monitor         │  scheduled job (ml/monitor_drift.py) checks
│                       │  precision/recall against baseline, flags decay
└─────────────────────┘

Additionally, at any time: GET /alerts/{id}/similar-cases retrieves past
investigations with similar risk profiles from the audit log — "have we
seen something like this before, and what did we decide?"
```

## Tech Stack

Python · FastAPI · XGBoost · Isolation Forest · SHAP · PydanticAI ·
Multi-Agent Orchestration · NetworkX (Case Linking Graph) · MongoDB Atlas ·
LangSmith · Docker · Analyst Feedback Loop · Drift Monitoring

## Project structure

```
sentinel-fraud-ai/
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── config.py                # settings from .env
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── ml_models.py         # loads XGBoost/Isolation Forest
│   ├── agents/
│   │   ├── investigator_agent.py
│   │   ├── disposition_agent.py
│   │   ├── case_routing_agent.py    # cross-silo team routing
│   │   └── prompts.py
│   ├── services/
│   │   ├── network_analysis.py  # NetworkX shared-entity checks (fraud only)
│   │   ├── case_linking.py      # NetworkX cross-silo graph (fraud+AML+sanctions)
│   │   ├── case_retrieval.py    # similar past case lookup
│   │   ├── shap_explainer.py
│   │   ├── audit_logger.py      # MongoDB writes
│   │   ├── feedback_loop.py     # analyst override tracking
│   │   └── tracing.py           # LangSmith setup
│   ├── db/
│   │   └── mongo_client.py
│   └── routers/
│       └── alerts.py            # API + feedback + case routing endpoints
├── ml/
│   ├── train_xgboost.py
│   ├── train_isolation_forest.py
│   ├── evaluate.py
│   └── monitor_drift.py         # weekly drift check
├── data/
│   ├── raw/                     # transactions.csv, aml_alerts.csv, sanctions_hits.csv
│   └── processed/                # trained models, metrics.json
├── tests/
├── scripts/
│   ├── seed_synthetic_alerts.py
│   └── seed_aml_sanctions_signals.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup — step by step

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/sentinel-fraud-ai.git
cd sentinel-fraud-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in:
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` — for the Disposition Agent
- `MONGODB_URI` — free cluster at https://www.mongodb.com/cloud/atlas
- `LANGCHAIN_API_KEY` — free account at https://smith.langchain.com

### 3. Generate data and train models

```bash
python scripts/seed_synthetic_alerts.py         # generates data/raw/transactions.csv
python scripts/seed_aml_sanctions_signals.py    # generates AML alerts + sanctions hits
python -m ml.train_xgboost                      # trains + saves XGBoost model
python -m ml.train_isolation_forest             # trains + saves Isolation Forest
python -m ml.evaluate                           # prints + saves benchmark metrics
```

> Note: these are run with `-m` (as modules) rather than as plain scripts,
> since `train_isolation_forest.py` and `evaluate.py` import shared
> constants from `train_xgboost.py`. Always run from the project root.

> To use a real dataset instead of synthetic data (recommended before
> publishing final numbers), download **IEEE-CIS Fraud Detection** or
> **PaySim** from Kaggle, and reshape it to match the columns used in
> `scripts/seed_synthetic_alerts.py`.

### 4. Run the API

```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

### 5. Test an alert

Fraud-only investigation:
```bash
curl -X POST http://localhost:8000/alerts/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_test_001",
    "account_id": 123,
    "beneficiary_id": 456,
    "device_id": 1,
    "ip_id": 10,
    "amount": 4500.0,
    "hour_of_day": 3,
    "is_new_beneficiary": 1,
    "account_age_days": 12,
    "prior_alerts_for_account": 2
  }'
```

Full cross-silo investigation (checks AML + Sanctions and routes to the
correct team(s)):
```bash
curl -X POST http://localhost:8000/alerts/investigate-case \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_test_002",
    "account_id": 2651,
    "beneficiary_id": 456,
    "device_id": 1,
    "ip_id": 10,
    "amount": 4500.0,
    "hour_of_day": 3,
    "is_new_beneficiary": 1,
    "account_age_days": 12,
    "prior_alerts_for_account": 2
  }'
```
> Use an `account_id` from your generated `aml_alerts.csv` (e.g. run
> `python -c "import pandas as pd; print(pd.read_csv('data/raw/aml_alerts.csv')['account_id'].iloc[0])"`)
> to see a real cross-silo finding trigger AML routing.

### 6. Run tests

```bash
pytest tests/ -v
```

## Benchmark results

Run `python ml/evaluate.py` and paste your results here, e.g.:

```json
{
  "model_auc": 0.94,
  "precision": 0.81,
  "recall": 0.77,
  "auto_clear_rate": 0.42,
  "auto_clear_correctness": 0.98,
  "simulated_analyst_review_reduction_pct": 42.0
}
```

## Run with Docker (one command, local Mongo included)

```bash
docker-compose up --build
```

This starts the API on `http://localhost:8000` plus a local MongoDB
instance — no Atlas account needed for local development. Swap
`MONGODB_URI` back to your Atlas connection string in `.env` when you
want audit logs to persist to the cloud instead.

## Analyst feedback loop

Every disposition decision can be confirmed or overridden by an analyst:

```bash
curl -X POST http://localhost:8000/alerts/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_test_001",
    "original_decision": "auto_clear",
    "analyst_decision": "escalate",
    "analyst_notes": "Found a pattern the model missed"
  }'

# Check the current override rate:
curl http://localhost:8000/alerts/feedback/override-rate
```

A rising override rate is the signal that auto-clear rules need
tightening — this is the mechanism that would feed periodic model
retraining in a production system.

## Drift monitoring

```bash
python -m ml.monitor_drift
```

Re-evaluates the model against recent labeled data and flags if
precision/recall has degraded beyond a 5-point threshold vs. the
original benchmark. In production this would run on a schedule (cron /
Airflow) and alert on-call if drift is detected.

## Dashboard (for demo, screenshots, and video)

A lightweight Streamlit dashboard is included so you can see and record
the system working end-to-end — no separate frontend framework needed.

```bash
pip install streamlit pyvis   # already in requirements.txt
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` with 5 pages:
- **Overview** — alert volume, auto-clear rate, benchmark metrics
- **Case Detail** — risk scores, fraud probability, cross-silo AML/Sanctions findings for one transaction
- **Relationship Graph** — interactive graph of shared devices/IPs/beneficiaries (pyvis)
- **Audit Log** — table of logged decisions (reads from MongoDB, or a local export — see below)
- **Investigation Report** — auto-generated plain-English case summary

To populate the Audit Log page without a live DB connection during a
demo session:
```bash
python scripts/export_audit_logs_demo.py
```

## Screenshots

See `docs/screenshots/` — includes a real relationship graph rendered
directly from this project's synthetic fraud-ring data (`relationship_graph.png`).
Capture the rest yourself once the dashboard is running (see the
step-by-step guide below).

## Roadmap / possible extensions

- Swap in-memory NetworkX graph for a persistent graph DB (Neo4j) refreshed
  incrementally as new transactions arrive
- Use analyst feedback to periodically retrain/recalibrate the model,
  closing the loop instead of just measuring override rate
- Automate drift monitoring on a real schedule with alerting (Slack/email)
- Deploy to a live URL (Render / Railway / Fly.io) for a public demo link

## License

MIT — see [LICENSE](LICENSE)

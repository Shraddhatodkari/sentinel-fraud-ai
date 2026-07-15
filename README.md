# Sentinel — Agentic Cross-Silo Financial Crime Investigation Platform

Financial institutions investigate fraud, money laundering (AML), and
sanctions violations using **separate systems, run by separate teams**.
This means the same customer can be independently flagged by the Fraud, AML, and Sanctions teams, with no unified view connecting related investigations across the organization.

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

## Problem Addressed

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

## Screenshots

### Swagger API Homepage

Interactive FastAPI documentation exposing investigation, cross-silo routing, analyst feedback, and case retrieval endpoints.

![Swagger API Homepage](screenshots/Swagger%20API%20Homepage.PNG)

---

### Fraud Investigation Response

Fraud investigation output showing model predictions, SHAP explanations, and AI-generated analyst rationale.

![Fraud Investigation Response](screenshots/Fraud%20Investigation%20Response.PNG)

---

### Cross-Silo Case Routing Response

Cross-silo investigation identifying linked AML and sanctions findings and routing the case to the appropriate investigation teams.

![Cross-Silo Case Routing Response](screenshots/Cross-Silo%20Case%20Routing%20Response.PNG)

---

### Multi-Agent Cross-Silo Investigation Result

End-to-end investigation workflow combining fraud analysis, case linking, and AI-driven routing decisions.

![Multi-Agent Cross-Silo Investigation Result](screenshots/Multi-Agent%20Cross-Silo%20Investigation%20Result.PNG)

---

### System Architecture

High-level architecture illustrating the multi-agent investigation workflow and supporting components.

![System Architecture](screenshots/Architecture%20Diagram.PNG)


## Tech Stack

Python · FastAPI · XGBoost · Isolation Forest · SHAP · PydanticAI ·
Multi-Agent Orchestration · NetworkX (Case Linking Graph) · MongoDB Atlas ·
LangSmith · Docker · Analyst Feedback Loop · Drift Monitoring

## Project Structure

```text
sentinel-fraud-ai/
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── config.py                # settings from .env
│   ├── models/
│   │   ├── schemas.py
│   │   └── ml_models.py
│   ├── agents/
│   │   ├── investigator_agent.py
│   │   ├── disposition_agent.py
│   │   ├── case_routing_agent.py
│   │   └── prompts.py
│   ├── services/
│   │   ├── network_analysis.py
│   │   ├── case_linking.py
│   │   ├── case_retrieval.py
│   │   ├── shap_explainer.py
│   │   ├── audit_logger.py
│   │   ├── feedback_loop.py
│   │   └── tracing.py
│   ├── db/
│   │   └── mongo_client.py
│   └── routers/
│       └── alerts.py
├── ml/
├── data/
├── dashboard/
│   └── app.py
├── docs/
├── screenshots/
├── scripts/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup — step by step

### 1. Clone and install

```bash
git clone https://github.com/Shraddhatodkari/sentinel-fraud-ai.git
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
- `GOOGLE_API_KEY` — Google Gemini API key
- `MONGODB_URI` — MongoDB Atlas connection string
- `LANGCHAIN_API_KEY` — LangSmith API key
  
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

## Benchmark Results

Run the evaluation script:

```bash
python -m ml.evaluate
```

Example output:

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
- **Audit Log** — table of investigation decisions stored in MongoDB Atlas
- **Investigation Report** — auto-generated plain-English case summary


## Roadmap / possible extensions

- Swap in-memory NetworkX graph for a persistent graph DB (Neo4j) refreshed
  incrementally as new transactions arrive
- Use analyst feedback to periodically retrain/recalibrate the model,
  closing the loop instead of just measuring override rate
- Automate drift monitoring on a real schedule with alerting (Slack/email)
- Deploy to a live URL (Render / Railway / Fly.io) for a public demo link

## License

MIT — see [LICENSE](LICENSE)

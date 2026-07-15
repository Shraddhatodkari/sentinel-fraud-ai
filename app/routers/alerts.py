"""
STEP 8 (+ STEP 12 extension). The main API endpoints — orchestrates the
full pipeline:

TransactionInput -> InvestigatorAgent -> DispositionAgent -> AuditLogger -> response
                                       -> CaseLinker -> RoutingAgent (cross-silo)
"""
import os

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.agents.case_routing_agent import decide_routing
from app.agents.disposition_agent import decide_disposition
from app.agents.investigator_agent import InvestigatorAgent
from app.db.mongo_client import get_db
from app.models.ml_models import get_model_registry
from app.models.schemas import CaseRoutingResult, DispositionResult, TransactionInput
from app.services.audit_logger import AuditLogger
from app.services.case_linking import CaseLinker
from app.services.case_retrieval import CaseRetriever
from app.services.feedback_loop import AnalystFeedback, FeedbackStore
from app.services.network_analysis import FraudNetworkAnalyzer
from app.services.shap_explainer import FraudExplainer

router = APIRouter(prefix="/alerts", tags=["alerts"])

# Loaded once at startup — see app/main.py
_investigator: InvestigatorAgent | None = None
_case_linker: CaseLinker | None = None


def get_investigator() -> InvestigatorAgent:
    global _investigator
    if _investigator is None:
        registry = get_model_registry()
        explainer = FraudExplainer(registry.xgb_model, registry.feature_columns)
        # Network analyzer needs historical transaction context to detect
        # shared entities — loaded from the same dataset used for training.
        history_df = pd.read_csv("data/raw/transactions.csv")
        network_analyzer = FraudNetworkAnalyzer(history_df)
        _investigator = InvestigatorAgent(registry, explainer, network_analyzer)
    return _investigator


def get_case_linker() -> CaseLinker:
    """
    Builds the cross-silo graph linking fraud transactions to AML alerts
    and sanctions hits. AML/sanctions files are optional — if you haven't
    run scripts/seed_aml_sanctions_signals.py yet, this still works with
    just the fraud transaction graph (no cross-silo findings will surface).
    """
    global _case_linker
    if _case_linker is None:
        history_df = pd.read_csv("data/raw/transactions.csv")
        aml_df = (
            pd.read_csv("data/raw/aml_alerts.csv")
            if os.path.exists("data/raw/aml_alerts.csv") else None
        )
        sanctions_df = (
            pd.read_csv("data/raw/sanctions_hits.csv")
            if os.path.exists("data/raw/sanctions_hits.csv") else None
        )
        _case_linker = CaseLinker(history_df, aml_df, sanctions_df)
    return _case_linker


@router.post("/investigate", response_model=DispositionResult)
async def investigate_alert(
    transaction: TransactionInput,
    investigator: InvestigatorAgent = Depends(get_investigator),
):
    """Fraud-only investigation: risk scores, SHAP, network analysis, disposition."""
    try:
        context = investigator.investigate(transaction)
        disposition = await decide_disposition(context)

        db = get_db()
        audit_logger = AuditLogger(db)
        await audit_logger.log_decision(context, disposition)

        return disposition
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/investigate-case", response_model=CaseRoutingResult)
async def investigate_case(
    transaction: TransactionInput,
    investigator: InvestigatorAgent = Depends(get_investigator),
    case_linker: CaseLinker = Depends(get_case_linker),
):
    """
    Full cross-silo investigation: runs the fraud disposition AND checks
    whether the same account/beneficiary is flagged in AML or Sanctions
    systems, then routes the case to the correct team(s). This is the
    endpoint that answers "who else in the bank should know about this?"
    """
    try:
        context = investigator.investigate(transaction)
        disposition = await decide_disposition(context)

        cross_silo_findings = case_linker.find_cross_silo_findings(
            account_id=transaction.account_id,
            beneficiary_id=transaction.beneficiary_id,
        )
        routing = await decide_routing(disposition, cross_silo_findings)

        db = get_db()
        audit_logger = AuditLogger(db)
        await audit_logger.log_decision(context, disposition)
        await audit_logger.log_routing(routing)

        return routing
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_feedback(feedback: AnalystFeedback):
    """
    Lets an analyst confirm or override a disposition decision. This is
    the feedback loop: overrides get tracked so we can measure how often
    the system's auto-clear/escalate calls need correcting, without ever
    mutating the original audit record.
    """
    db = get_db()
    store = FeedbackStore(db)
    record = await store.record_feedback(feedback)
    return record


@router.get("/{transaction_id}/similar-cases")
async def get_similar_cases(transaction_id: str, top_n: int = 3):
    """
    Retrieves past investigations with similar risk profiles from the
    audit log, along with what was decided — so an analyst can see
    "have we handled something like this before?" without manually
    searching case history.
    """
    db = get_db()
    current = await db["audit_logs"].find_one({"transaction_id": transaction_id})
    if not current:
        raise HTTPException(status_code=404, detail="Transaction not found in audit log")

    past_cases = await db["audit_logs"].find({}).to_list(length=1000)
    retriever = CaseRetriever(past_cases)
    return retriever.find_similar(current, top_n=top_n)


@router.get("/feedback/override-rate")
async def get_override_rate():
    """Returns the current analyst override rate — the key health metric
    for whether auto-clear rules need tightening. Check this on a
    regular cadence in production (this is what drift monitoring watches)."""
    db = get_db()
    store = FeedbackStore(db)
    return await store.get_override_rate()


@router.get("/health")
async def health_check():
    return {"status": "ok"}

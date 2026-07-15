"""
STEP 6. Writes an immutable record of every disposition decision —
auto-cleared or escalated — for regulatory/audit review. This is what
lets Sentinel claim "audit-ready" rather than just "explainable."
"""
from datetime import datetime, timezone

from app.models.schemas import DispositionResult, InvestigationContext


class AuditLogger:
    def __init__(self, db):
        self.collection = db["audit_logs"]

    async def log_decision(
        self,
        context: InvestigationContext,
        disposition: DispositionResult,
        agent_trace_id: str | None = None,
    ):
        record = {
            "transaction_id": context.transaction_id,
            "fraud_probability": context.risk.fraud_probability,
            "anomaly_score": context.risk.anomaly_score,
            "shap_factors": context.risk.top_shap_factors,
            "network_findings": [nf.model_dump() for nf in context.network_findings],
            "decision": disposition.decision.value,
            "confidence": disposition.confidence,
            "rationale": disposition.rationale,
            "summary_for_analyst": disposition.summary_for_analyst,
            "agent_trace_id": agent_trace_id,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        await self.collection.insert_one(record)
        return record

    async def log_routing(self, routing) -> dict:
        """
        Logs the cross-silo case routing decision separately from the
        fraud disposition — this is the record that shows WHY a case was
        (or wasn't) looped in to AML or Sanctions, for audit purposes.
        """
        record = {
            "transaction_id": routing.transaction_id,
            "teams_involved": routing.teams_involved,
            "rationale": routing.rationale,
            "cross_silo_findings": [f.model_dump() for f in routing.cross_silo_findings],
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        await self.collection.database["case_routing_logs"].insert_one(record)
        return record

"""
STEP 5A. The Investigator Agent.

Gathers ALL context around a flagged transaction — risk scores, SHAP
explanations, and network connections to other flagged transactions —
and packages it into a structured InvestigationContext. It does not
decide anything; it only investigates. The Disposition Agent decides.

This separation mirrors how a real fraud-ops team works: a junior
analyst gathers case facts, a senior analyst/reviewer makes the call.
"""
from app.models.schemas import InvestigationContext, RiskScore, TransactionInput
from app.models.ml_models import ModelRegistry
from app.services.shap_explainer import FraudExplainer
from app.services.network_analysis import FraudNetworkAnalyzer


class InvestigatorAgent:
    """
    Note: this agent is deterministic/rule-based for the context-gathering
    step (scores, SHAP, network lookups are computed directly — there's no
    ambiguity in "what are the facts"). The LLM-driven reasoning happens in
    the Disposition Agent, where judgment is actually required. This keeps
    the investigation step fast, cheap, and 100% reproducible for audit
    purposes, while still reserving "agentic" reasoning for the decision
    that needs it.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        explainer: FraudExplainer,
        network_analyzer: FraudNetworkAnalyzer,
    ):
        self.model_registry = model_registry
        self.explainer = explainer
        self.network_analyzer = network_analyzer

    def investigate(self, transaction: TransactionInput) -> InvestigationContext:
        txn_dict = transaction.model_dump()

        scores = self.model_registry.score(txn_dict)
        top_factors = self.explainer.top_factors(txn_dict)

        network_findings = self.network_analyzer.find_shared_entities(
            transaction_id=transaction.transaction_id,
            device_id=transaction.device_id,
            ip_id=transaction.ip_id,
            beneficiary_id=transaction.beneficiary_id,
        )

        return InvestigationContext(
            transaction_id=transaction.transaction_id,
            risk=RiskScore(
                fraud_probability=scores["fraud_probability"],
                anomaly_score=scores["anomaly_score"],
                top_shap_factors=top_factors,
            ),
            network_findings=network_findings,
            prior_alerts_for_account=transaction.prior_alerts_for_account,
        )

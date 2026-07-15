"""
STEP 5B. The Disposition Agent.

Takes the InvestigationContext produced by the Investigator Agent and
makes the actual auto-clear / escalate decision using an LLM via
PydanticAI, with a strictly typed output (DispositionResult) so the
result is always safe to log and act on programmatically.

This is where "judgment" actually lives — deciding whether the combination
of signals is trustworthy enough to skip human review.
"""
from pydantic_ai import Agent

from app.models.schemas import DispositionResult, InvestigationContext
from app.agents.prompts import DISPOSITION_SYSTEM_PROMPT

# Swap the model string for whichever provider/key you configured in .env.
# Examples: "openai:gpt-4o-mini", "anthropic:claude-sonnet-4-6"
disposition_agent = Agent(
    "google:gemini-3.5-flash",
    output_type=DispositionResult,
    system_prompt=DISPOSITION_SYSTEM_PROMPT,
)


async def decide_disposition(context: InvestigationContext) -> DispositionResult:
    prompt = f"""
Investigation context for transaction {context.transaction_id}:

- Fraud probability: {context.risk.fraud_probability:.3f}
- Anomaly score: {context.risk.anomaly_score:.3f}
- Top SHAP factors: {context.risk.top_shap_factors}
- Network findings: {[nf.model_dump() for nf in context.network_findings]}
- Prior alerts for this account: {context.prior_alerts_for_account}

Decide: AUTO_CLEAR or ESCALATE, with confidence, rationale, and (if
escalating) a summary_for_analyst.
"""
    result = await disposition_agent.run(prompt)
    disposition = result.output
    disposition.transaction_id = context.transaction_id
    return disposition

"""
STEP 12. The Case Routing Agent — the piece that actually solves the
"siloed teams" problem, not just the "too many false positives" problem.

Takes the fraud disposition PLUS cross-silo findings (AML alerts,
sanctions hits on the same account/beneficiary) and decides which
team(s) should own or be looped into the case.
"""

from pydantic_ai import Agent

from app.agents.prompts import ROUTING_SYSTEM_PROMPT
from app.models.schemas import (
    CaseRoutingResult,
    CrossSiloFindingRef,
    DispositionResult,
)
from app.services.case_linking import CrossSiloFinding

routing_agent = Agent(
    "google:gemini-3.5-flash",
    output_type=CaseRoutingResult,
    system_prompt=ROUTING_SYSTEM_PROMPT,
)


async def decide_routing(
    disposition: DispositionResult,
    cross_silo_findings: list[CrossSiloFinding],
) -> CaseRoutingResult:

    findings_text = (
        "\n".join(
            f"- [{f.source_team}] {f.detail}"
            for f in cross_silo_findings
        )
        if cross_silo_findings
        else "None — no matches found in AML or Sanctions systems."
    )

    prompt = f"""
Fraud disposition for transaction {disposition.transaction_id}:

- Decision: {disposition.decision.value}
- Rationale: {disposition.rationale}

Cross-silo findings:
{findings_text}

Decide which team(s) should own or be looped into this case, and why.
"""

    result = await routing_agent.run(prompt)

    routing = result.output
    routing.transaction_id = disposition.transaction_id
    routing.cross_silo_findings = [
        CrossSiloFindingRef(**f.model_dump())
        for f in cross_silo_findings
    ]

    return routing
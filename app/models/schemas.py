"""
Shared Pydantic models used across the API, agents, and services.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    transaction_id: str
    account_id: int
    beneficiary_id: int
    device_id: int
    ip_id: int
    amount: float
    hour_of_day: int = Field(ge=0, le=23)
    is_new_beneficiary: int = Field(ge=0, le=1)
    account_age_days: int
    prior_alerts_for_account: int


class RiskScore(BaseModel):
    fraud_probability: float
    anomaly_score: float
    top_shap_factors: list[str]


class NetworkFinding(BaseModel):
    shared_entity_type: str  # "device" | "ip" | "beneficiary"
    connected_transaction_ids: list[str]


class InvestigationContext(BaseModel):
    transaction_id: str
    risk: RiskScore
    network_findings: list[NetworkFinding]
    prior_alerts_for_account: int


class Disposition(str, Enum):
    AUTO_CLEAR = "auto_clear"
    ESCALATE = "escalate"


class DispositionResult(BaseModel):
    transaction_id: str
    decision: Disposition
    confidence: float
    rationale: str
    summary_for_analyst: Optional[str] = None


class CrossSiloFindingRef(BaseModel):
    source_team: str
    finding_type: str
    detail: str
    reference_id: str


class CaseRoutingResult(BaseModel):
    transaction_id: str
    teams_involved: list[str]  # e.g. ["Fraud", "AML", "Sanctions"]
    rationale: str
    cross_silo_findings: list[CrossSiloFindingRef] = []

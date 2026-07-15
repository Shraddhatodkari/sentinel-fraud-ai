"""
STEP 13. Similar Case Retrieval — efficient because it reuses data you
already have (the audit log), rather than requiring a new system.

Analysts constantly ask "have we seen something like this before, and
what did we decide?" Right now that knowledge only lives in individual
analysts' memory. This service searches PAST audit log entries for
similar past investigations by comparing feature vectors (risk scores,
SHAP factor patterns), and surfaces the past decision + outcome —
promoting more consistent decisions and faster investigations.

No vector database needed: at portfolio-project scale, a simple cosine
similarity over a small numeric feature vector, computed in Python, is
fast enough and honest about its own scale limits (noted below).
"""
import numpy as np
from pydantic import BaseModel


class SimilarCase(BaseModel):
    transaction_id: str
    similarity_score: float
    past_decision: str
    past_rationale: str


class CaseRetriever:
    """
    Expects audit log records shaped like the ones AuditLogger writes:
    {transaction_id, fraud_probability, anomaly_score, decision, rationale}
    """

    def __init__(self, past_cases: list[dict]):
        self.past_cases = past_cases

    @staticmethod
    def _vector(case: dict) -> np.ndarray:
        return np.array([
            case.get("fraud_probability", 0.0),
            case.get("anomaly_score", 0.0),
        ])

    def find_similar(self, current_case: dict, top_n: int = 3) -> list[SimilarCase]:
        if not self.past_cases:
            return []

        current_vec = self._vector(current_case)
        scored = []
        for case in self.past_cases:
            if case["transaction_id"] == current_case.get("transaction_id"):
                continue  # don't match a case against itself
            past_vec = self._vector(case)
            # cosine similarity
            denom = (np.linalg.norm(current_vec) * np.linalg.norm(past_vec))
            similarity = float(np.dot(current_vec, past_vec) / denom) if denom > 0 else 0.0
            scored.append((similarity, case))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SimilarCase(
                transaction_id=case["transaction_id"],
                similarity_score=round(sim, 4),
                past_decision=case.get("decision", "unknown"),
                past_rationale=case.get("rationale", ""),
            )
            for sim, case in scored[:top_n]
        ]

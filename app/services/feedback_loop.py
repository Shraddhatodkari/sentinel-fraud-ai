"""
STEP 9. Feedback loop — lets analysts confirm or override a disposition
decision. Overrides are stored separately from the original audit log so
we can measure how often the system's decisions get reversed, and by
which rule/factor, without ever mutating the original immutable record.
"""
from datetime import datetime, timezone

from pydantic import BaseModel

from app.models.schemas import Disposition


class AnalystFeedback(BaseModel):
    transaction_id: str
    original_decision: Disposition
    analyst_decision: Disposition  # what the analyst actually decided
    analyst_notes: str | None = None


class FeedbackStore:
    def __init__(self, db):
        self.collection = db["analyst_feedback"]

    async def record_feedback(self, feedback: AnalystFeedback):
        was_override = feedback.original_decision != feedback.analyst_decision
        record = {
            **feedback.model_dump(),
            "was_override": was_override,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        await self.collection.insert_one(record)
        return record

    async def get_override_rate(self) -> dict:
        """
        Returns how often the system's auto-clear decisions were overridden
        by an analyst — the key signal for whether auto-clear rules need
        tightening. This is what you'd check on a regular cadence in
        production, and what the drift monitor (see ml/monitor_drift.py)
        also flags automatically.
        """
        total = await self.collection.count_documents({})
        if total == 0:
            return {"total_feedback": 0, "override_rate": None}
        overrides = await self.collection.count_documents({"was_override": True})
        return {
            "total_feedback": total,
            "overrides": overrides,
            "override_rate": round(overrides / total, 4),
        }

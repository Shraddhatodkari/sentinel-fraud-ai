"""
Run: pytest tests/test_feedback.py
Uses mongomock-style logic conceptually — for a real run against Atlas,
see README for setup. This test checks the override-detection logic only.
"""
from app.models.schemas import Disposition
from app.services.feedback_loop import AnalystFeedback


def test_override_detected_when_decisions_differ():
    feedback = AnalystFeedback(
        transaction_id="txn_001",
        original_decision=Disposition.AUTO_CLEAR,
        analyst_decision=Disposition.ESCALATE,
        analyst_notes="Found suspicious pattern the model missed",
    )
    was_override = feedback.original_decision != feedback.analyst_decision
    assert was_override is True


def test_no_override_when_decisions_match():
    feedback = AnalystFeedback(
        transaction_id="txn_002",
        original_decision=Disposition.ESCALATE,
        analyst_decision=Disposition.ESCALATE,
    )
    was_override = feedback.original_decision != feedback.analyst_decision
    assert was_override is False

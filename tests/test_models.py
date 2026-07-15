"""
Run: pytest tests/test_models.py
Requires models to already be trained (see README Phase 2).
"""
import pytest

from app.models.ml_models import get_model_registry


@pytest.fixture(scope="module")
def registry():
    return get_model_registry()


def test_score_returns_probability_in_valid_range(registry):
    txn = {
        "amount": 500.0,
        "hour_of_day": 14,
        "is_new_beneficiary": 0,
        "account_age_days": 800,
        "prior_alerts_for_account": 0,
    }
    scores = registry.score(txn)
    assert 0.0 <= scores["fraud_probability"] <= 1.0
    assert 0.0 <= scores["anomaly_score"] <= 1.0


def test_high_risk_transaction_scores_higher_than_low_risk(registry):
    low_risk = {
        "amount": 50.0,
        "hour_of_day": 14,
        "is_new_beneficiary": 0,
        "account_age_days": 2000,
        "prior_alerts_for_account": 0,
    }
    high_risk = {
        "amount": 9000.0,
        "hour_of_day": 3,
        "is_new_beneficiary": 1,
        "account_age_days": 5,
        "prior_alerts_for_account": 3,
    }
    low_score = registry.score(low_risk)["fraud_probability"]
    high_score = registry.score(high_risk)["fraud_probability"]
    assert high_score > low_score

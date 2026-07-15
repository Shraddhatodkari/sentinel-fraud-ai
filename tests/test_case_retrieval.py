"""
Run: pytest tests/test_case_retrieval.py
"""
from app.services.case_retrieval import CaseRetriever


def test_finds_most_similar_case_first():
    past_cases = [
        {"transaction_id": "txn_A", "fraud_probability": 0.85, "anomaly_score": 0.7,
         "decision": "escalate", "rationale": "High risk pattern"},
        {"transaction_id": "txn_B", "fraud_probability": 0.05, "anomaly_score": 0.02,
         "decision": "auto_clear", "rationale": "Low risk"},
    ]
    retriever = CaseRetriever(past_cases)
    current = {"transaction_id": "txn_NEW", "fraud_probability": 0.80, "anomaly_score": 0.65}

    results = retriever.find_similar(current, top_n=1)
    assert len(results) == 1
    assert results[0].transaction_id == "txn_A"
    assert results[0].past_decision == "escalate"


def test_excludes_self_match():
    past_cases = [
        {"transaction_id": "txn_SAME", "fraud_probability": 0.5, "anomaly_score": 0.5,
         "decision": "escalate", "rationale": "test"},
    ]
    retriever = CaseRetriever(past_cases)
    current = {"transaction_id": "txn_SAME", "fraud_probability": 0.5, "anomaly_score": 0.5}

    results = retriever.find_similar(current)
    assert results == []


def test_empty_history_returns_empty_list():
    retriever = CaseRetriever([])
    results = retriever.find_similar({"transaction_id": "txn_X", "fraud_probability": 0.5, "anomaly_score": 0.5})
    assert results == []

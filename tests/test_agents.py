"""
Run: pytest tests/test_agents.py
"""
import pandas as pd

from app.services.network_analysis import FraudNetworkAnalyzer


def test_network_analyzer_detects_shared_device():
    df = pd.DataFrame([
        {"transaction_id": "txn_A", "device_id": 1, "ip_id": 10, "beneficiary_id": 100},
        {"transaction_id": "txn_B", "device_id": 1, "ip_id": 11, "beneficiary_id": 101},
        {"transaction_id": "txn_C", "device_id": 2, "ip_id": 12, "beneficiary_id": 102},
    ])
    analyzer = FraudNetworkAnalyzer(df)

    findings = analyzer.find_shared_entities(
        transaction_id="txn_A", device_id=1, ip_id=10, beneficiary_id=100
    )
    device_findings = [f for f in findings if f.shared_entity_type == "device"]
    assert len(device_findings) == 1
    assert "txn_B" in device_findings[0].connected_transaction_ids


def test_network_analyzer_finds_nothing_when_isolated():
    df = pd.DataFrame([
        {"transaction_id": "txn_A", "device_id": 1, "ip_id": 10, "beneficiary_id": 100},
        {"transaction_id": "txn_C", "device_id": 2, "ip_id": 12, "beneficiary_id": 102},
    ])
    analyzer = FraudNetworkAnalyzer(df)

    findings = analyzer.find_shared_entities(
        transaction_id="txn_C", device_id=2, ip_id=12, beneficiary_id=102
    )
    assert findings == []

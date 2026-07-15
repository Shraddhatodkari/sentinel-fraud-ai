"""
STEP 4. Detects coordinated fraud by checking whether a flagged
transaction shares a device, IP, or beneficiary with OTHER flagged
transactions — a signal that per-transaction scoring alone can't see.

This builds an in-memory graph from historical transaction data.
For a production system, this would be a persistent graph store
(e.g., Neo4j) refreshed incrementally — noted as a future improvement.
"""
import pandas as pd
import networkx as nx

from app.models.schemas import NetworkFinding


class FraudNetworkAnalyzer:
    def __init__(self, transactions_df: pd.DataFrame):
        self.graph = nx.Graph()
        self._build_graph(transactions_df)

    def _build_graph(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            txn = row["transaction_id"]
            self.graph.add_edge(txn, f"device_{row['device_id']}")
            self.graph.add_edge(txn, f"ip_{row['ip_id']}")
            self.graph.add_edge(txn, f"beneficiary_{row['beneficiary_id']}")

    def find_shared_entities(
        self, transaction_id: str, device_id: int, ip_id: int, beneficiary_id: int
    ) -> list[NetworkFinding]:
        findings = []
        entity_map = {
            "device": f"device_{device_id}",
            "ip": f"ip_{ip_id}",
            "beneficiary": f"beneficiary_{beneficiary_id}",
        }
        for entity_type, node in entity_map.items():
            if node not in self.graph:
                continue
            connected_txns = [
                n for n in self.graph.neighbors(node)
                if n.startswith("txn_") and n != transaction_id
            ]
            if connected_txns:
                findings.append(
                    NetworkFinding(
                        shared_entity_type=entity_type,
                        connected_transaction_ids=connected_txns[:10],  # cap for readability
                    )
                )
        return findings

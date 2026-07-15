"""
STEP 11. Case Linking Layer — the core of the "siloed teams miss
connections" problem.

Fraud, AML, and sanctions teams typically investigate the same
customers using separate systems. This builds ONE graph connecting:
- fraud alert transactions (accounts, devices, IPs, beneficiaries)
- AML alerts (flagged accounts)
- sanctions hits (flagged beneficiaries)

Given a single flagged transaction, it answers a question no single
team's system can answer alone: "Is this account or beneficiary ALSO
flagged somewhere else in the institution, under a different team?"
"""
import pandas as pd
import networkx as nx

from pydantic import BaseModel


class CrossSiloFinding(BaseModel):
    source_team: str          # "AML" | "Sanctions"
    finding_type: str          # e.g. "aml_alert" | "sanctions_hit"
    detail: str                # human-readable description
    reference_id: str          # aml_alert_id or sanctions_hit_id


class CaseLinker:
    def __init__(
        self,
        transactions_df: pd.DataFrame,
        aml_df: pd.DataFrame | None = None,
        sanctions_df: pd.DataFrame | None = None,
    ):
        self.graph = nx.Graph()
        self._index_transactions(transactions_df)
        if aml_df is not None:
            self._index_aml(aml_df)
        if sanctions_df is not None:
            self._index_sanctions(sanctions_df)

    def _index_transactions(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            acct = f"account_{row['account_id']}"
            bene = f"beneficiary_{row['beneficiary_id']}"
            self.graph.add_edge(acct, bene, relation="transacted_with")

    def _index_aml(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            acct = f"account_{row['account_id']}"
            alert_node = f"aml_alert_{row['aml_alert_id']}"
            self.graph.add_node(
                alert_node,
                type="aml_alert",
                typology=row["typology"],
                severity=row["severity"],
                reference_id=row["aml_alert_id"],
            )
            self.graph.add_edge(acct, alert_node, relation="flagged_by_aml")

    def _index_sanctions(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            bene = f"beneficiary_{row['beneficiary_id']}"
            hit_node = f"sanctions_hit_{row['sanctions_hit_id']}"
            self.graph.add_node(
                hit_node,
                type="sanctions_hit",
                match_confidence=row["match_confidence"],
                watchlist_source=row["watchlist_source"],
                reference_id=row["sanctions_hit_id"],
            )
            self.graph.add_edge(bene, hit_node, relation="matched_watchlist")

    def find_cross_silo_findings(
        self, account_id: int, beneficiary_id: int
    ) -> list[CrossSiloFinding]:
        """
        Checks whether the account or beneficiary on THIS fraud alert
        is also flagged in the AML or sanctions systems — the exact
        connection that gets missed when teams work in isolation.
        """
        findings: list[CrossSiloFinding] = []
        acct_node = f"account_{account_id}"
        bene_node = f"beneficiary_{beneficiary_id}"

        for node in (acct_node, bene_node):
            if node not in self.graph:
                continue
            for neighbor in self.graph.neighbors(node):
                data = self.graph.nodes.get(neighbor, {})
                if data.get("type") == "aml_alert":
                    findings.append(CrossSiloFinding(
                        source_team="AML",
                        finding_type="aml_alert",
                        detail=(
                            f"Account also flagged by AML for "
                            f"{data['typology']} (severity: {data['severity']})"
                        ),
                        reference_id=data["reference_id"],
                    ))
                elif data.get("type") == "sanctions_hit":
                    findings.append(CrossSiloFinding(
                        source_team="Sanctions",
                        finding_type="sanctions_hit",
                        detail=(
                            f"Beneficiary matched {data['watchlist_source']} "
                            f"watchlist (confidence: {data['match_confidence']})"
                        ),
                        reference_id=data["reference_id"],
                    ))
        return findings

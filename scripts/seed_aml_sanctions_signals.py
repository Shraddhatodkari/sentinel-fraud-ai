"""
Generates synthetic AML (Anti-Money Laundering) alerts and sanctions
screening hits, tied to the SAME account/beneficiary/device pool as
scripts/seed_synthetic_alerts.py.

This is what makes the "siloed teams miss connections" problem real and
testable: a customer can appear in the fraud system, the AML system, AND
a sanctions hit — each investigated independently, unless something
links them, which is exactly what the Case Linking layer does.

Run (after seed_synthetic_alerts.py):
    python scripts/seed_aml_sanctions_signals.py

Output:
    data/raw/aml_alerts.csv
    data/raw/sanctions_hits.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N_ACCOUNTS = 5000


def generate_aml_alerts(transactions_df: pd.DataFrame, n_alerts: int = 400) -> pd.DataFrame:
    """
    AML alerts flag accounts (not individual transactions) for patterns
    like structuring (many transactions just under reporting thresholds)
    or rapid fund movement through an account ("layering").
    """
    flagged_accounts = RNG.choice(
        transactions_df["account_id"].unique(), size=n_alerts, replace=False
    )
    typologies = RNG.choice(
        ["structuring", "layering", "rapid_movement", "shell_company_pattern"],
        size=n_alerts,
    )
    severity = RNG.choice(["low", "medium", "high"], size=n_alerts, p=[0.5, 0.35, 0.15])

    return pd.DataFrame({
        "aml_alert_id": [f"aml_{i:05d}" for i in range(n_alerts)],
        "account_id": flagged_accounts,
        "typology": typologies,
        "severity": severity,
        "team": "AML",
    })


def generate_sanctions_hits(transactions_df: pd.DataFrame, n_hits: int = 60) -> pd.DataFrame:
    """
    Sanctions hits flag beneficiaries matched (even partially) against a
    watchlist. In reality this comes from fuzzy name-matching against
    OFAC/UN/EU lists — simulated here as a direct flag on a subset of
    beneficiary_ids.
    """
    flagged_beneficiaries = RNG.choice(
        transactions_df["beneficiary_id"].unique(), size=n_hits, replace=False
    )
    match_confidence = np.round(RNG.uniform(0.6, 1.0, size=n_hits), 2)
    watchlist_source = RNG.choice(["OFAC", "UN", "EU"], size=n_hits)

    return pd.DataFrame({
        "sanctions_hit_id": [f"sanc_{i:05d}" for i in range(n_hits)],
        "beneficiary_id": flagged_beneficiaries,
        "match_confidence": match_confidence,
        "watchlist_source": watchlist_source,
        "team": "Sanctions",
    })


if __name__ == "__main__":
    txn_path = Path("data/raw/transactions.csv")
    if not txn_path.exists():
        raise FileNotFoundError(
            "Run `python scripts/seed_synthetic_alerts.py` first to generate transactions.csv"
        )
    transactions_df = pd.read_csv(txn_path)

    aml_df = generate_aml_alerts(transactions_df)
    sanctions_df = generate_sanctions_hits(transactions_df)

    aml_df.to_csv("data/raw/aml_alerts.csv", index=False)
    sanctions_df.to_csv("data/raw/sanctions_hits.csv", index=False)

    print(f"Generated {len(aml_df)} AML alerts -> data/raw/aml_alerts.csv")
    print(f"Generated {len(sanctions_df)} sanctions hits -> data/raw/sanctions_hits.csv")

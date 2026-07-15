"""
STEP 1 OF THE PROJECT.

Generates a synthetic transaction dataset that mimics real fraud data
(similar structure to public datasets like IEEE-CIS / PaySim), so you can
build and test the entire pipeline before (or instead of) plugging in a
real dataset.

Run:
    python scripts/seed_synthetic_alerts.py

Output:
    data/raw/transactions.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N_TRANSACTIONS = 20_000
N_DEVICES = 3_000
N_ACCOUNTS = 5_000
FRAUD_RATE = 0.03  # ~3% fraud, roughly realistic order of magnitude


def generate_transactions(n=N_TRANSACTIONS) -> pd.DataFrame:
    account_ids = RNG.integers(0, N_ACCOUNTS, size=n)
    beneficiary_ids = RNG.integers(0, N_ACCOUNTS, size=n)
    device_ids = RNG.integers(0, N_DEVICES, size=n)
    ip_pool = RNG.integers(0, N_DEVICES * 2, size=n)  # ips loosely tied to devices

    amount = np.round(RNG.lognormal(mean=4.0, sigma=1.2, size=n), 2)
    hour_of_day = RNG.integers(0, 24, size=n)
    is_new_beneficiary = RNG.integers(0, 2, size=n)
    account_age_days = RNG.integers(1, 3000, size=n)
    prior_alerts_for_account = RNG.poisson(0.3, size=n)

    # Label generation with some real signal baked in, so models
    # actually learn something meaningful:
    fraud_score = (
        0.00025 * amount
        + 0.15 * (hour_of_day < 5).astype(float)
        + 0.3 * is_new_beneficiary
        + 0.4 * (account_age_days < 30).astype(float)
        + 0.5 * (prior_alerts_for_account > 1).astype(float)
        + RNG.normal(0, 0.3, size=n)
    )
    threshold = np.quantile(fraud_score, 1 - FRAUD_RATE)
    is_fraud = (fraud_score >= threshold).astype(int)

    # Deliberately create a handful of "fraud rings": groups of
    # transactions sharing a device/IP/beneficiary, so the network
    # analysis layer has something real to detect.
    ring_size = 40
    n_rings = 6
    for _ in range(n_rings):
        idx = RNG.choice(n, size=ring_size, replace=False)
        shared_device = RNG.integers(0, N_DEVICES)
        shared_beneficiary = RNG.integers(0, N_ACCOUNTS)
        device_ids[idx] = shared_device
        beneficiary_ids[idx] = shared_beneficiary
        is_fraud[idx] = 1

    df = pd.DataFrame({
        "transaction_id": [f"txn_{i:06d}" for i in range(n)],
        "account_id": account_ids,
        "beneficiary_id": beneficiary_ids,
        "device_id": device_ids,
        "ip_id": ip_pool,
        "amount": amount,
        "hour_of_day": hour_of_day,
        "is_new_beneficiary": is_new_beneficiary,
        "account_age_days": account_age_days,
        "prior_alerts_for_account": prior_alerts_for_account,
        "is_fraud": is_fraud,
    })
    return df


if __name__ == "__main__":
    df = generate_transactions()
    out_path = Path("data/raw/transactions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} transactions -> {out_path}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3%}")

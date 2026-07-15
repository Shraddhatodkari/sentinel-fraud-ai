"""
STEP 2B. Trains the Isolation Forest anomaly detector — this catches
NOVEL fraud patterns that the supervised XGBoost model (trained on past
fraud) wouldn't recognize, since it never sees labels during training.

Run:
    python ml/train_isolation_forest.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.train_xgboost import FEATURE_COLUMNS


def train():
    df = pd.read_csv("data/processed/train_set.csv")
    X = df[FEATURE_COLUMNS]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,  # matches our synthetic fraud rate
        random_state=42,
    )
    model.fit(X)

    out_dir = Path("data/processed/models")
    joblib.dump(model, out_dir / "iso_forest_model.pkl")
    print(f"Isolation Forest trained and saved to {out_dir / 'iso_forest_model.pkl'}")


if __name__ == "__main__":
    train()

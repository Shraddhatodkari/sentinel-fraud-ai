"""
Loads trained models once at startup and exposes a single scoring function.
"""
import json

import joblib
import pandas as pd

from app.config import settings


class ModelRegistry:
    def __init__(self):
        self.xgb_model = joblib.load(settings.xgb_model_path)
        self.iso_forest = joblib.load(settings.iso_forest_model_path)
        with open(settings.feature_columns_path) as f:
            self.feature_columns = json.load(f)

    def score(self, transaction: dict) -> dict:
        """Returns fraud probability (XGBoost) and anomaly score (Isolation Forest)."""
        X = pd.DataFrame([transaction])[self.feature_columns]
        fraud_probability = float(self.xgb_model.predict_proba(X)[:, 1][0])
        # Isolation Forest: lower score_samples = more anomalous. We invert
        # and normalize roughly into a 0-1 "anomaly score" for readability.
        raw_anomaly = self.iso_forest.score_samples(X)[0]
        anomaly_score = float(max(0.0, min(1.0, 0.5 - raw_anomaly)))
        return {
            "fraud_probability": fraud_probability,
            "anomaly_score": anomaly_score,
        }


# Loaded once, reused across requests
model_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global model_registry
    if model_registry is None:
        model_registry = ModelRegistry()
    return model_registry

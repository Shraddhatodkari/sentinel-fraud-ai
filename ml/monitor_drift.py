"""
STEP 10. Drift monitoring.

Re-evaluates the model against a rolling window of recent labeled data
and flags if precision/recall has degraded beyond a threshold vs the
original benchmark — the kind of check a real production fraud system
runs on a schedule (e.g., weekly, via cron or an Airflow DAG).

Run:
    python -m ml.monitor_drift

In production this would run on a schedule and alert (Slack/email/PagerDuty)
if drift is detected — noted here as the natural next step.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from ml.train_xgboost import FEATURE_COLUMNS

DRIFT_THRESHOLD = 0.05  # flag if precision or recall drops more than 5 points


def load_baseline_metrics() -> dict:
    path = Path("data/processed/metrics.json")
    if not path.exists():
        raise FileNotFoundError(
            "No baseline metrics found. Run `python -m ml.evaluate` first."
        )
    with open(path) as f:
        return json.load(f)


def check_drift(recent_data_path: str = "data/raw/transactions.csv") -> dict:
    """
    In production, `recent_data_path` would point to the last N days of
    labeled outcomes (i.e., alerts where the true fraud/not-fraud label
    is now known, e.g., via chargeback data or confirmed investigations).
    For this project, we simulate it by re-scoring against the same
    labeled dataset the model was evaluated on.
    """
    baseline = load_baseline_metrics()

    df = pd.read_csv(recent_data_path)
    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    model = joblib.load("data/processed/models/xgb_model.pkl")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    current = {
        "auc": roc_auc_score(y, probs),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
    }

    precision_drift = baseline["precision"] - current["precision"]
    recall_drift = baseline["recall"] - current["recall"]

    drifted = precision_drift > DRIFT_THRESHOLD or recall_drift > DRIFT_THRESHOLD

    result = {
        "baseline_precision": baseline["precision"],
        "current_precision": round(current["precision"], 4),
        "precision_drift": round(precision_drift, 4),
        "baseline_recall": baseline["recall"],
        "current_recall": round(current["recall"], 4),
        "recall_drift": round(recall_drift, 4),
        "drift_detected": drifted,
    }

    if drifted:
        print("⚠️  DRIFT DETECTED — model performance has degraded beyond threshold.")
        print("    In production: trigger retraining pipeline + alert on-call.")
    else:
        print("✅ No significant drift detected.")

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    check_drift()

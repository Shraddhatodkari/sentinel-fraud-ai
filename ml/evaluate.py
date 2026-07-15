"""
STEP 2C. Evaluates the trained models on the held-out test set and
produces the benchmark numbers you'll quote on LinkedIn / in interviews.

Run:
    python ml/evaluate.py

Output:
    data/processed/metrics.json
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from ml.train_xgboost import FEATURE_COLUMNS

AUTO_CLEAR_THRESHOLD = 0.15  # predicted fraud probability below this -> auto-clear


def evaluate():
    test_df = pd.read_csv("data/processed/test_set.csv")
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["is_fraud"]

    model = joblib.load("data/processed/models/xgb_model.pkl")
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    # Simulate auto-clear: alerts scored as low risk get auto-cleared
    auto_cleared_mask = probs < AUTO_CLEAR_THRESHOLD
    auto_clear_rate = auto_cleared_mask.mean()
    # precision of auto-clear decision = how many auto-cleared were ACTUALLY not fraud
    auto_clear_correct = (y_test[auto_cleared_mask] == 0).mean() if auto_cleared_mask.sum() > 0 else None

    metrics = {
        "model_auc": round(float(auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "auto_clear_rate": round(float(auto_clear_rate), 4),
        "auto_clear_correctness": round(float(auto_clear_correct), 4) if auto_clear_correct else None,
        "simulated_analyst_review_reduction_pct": round(float(auto_clear_rate) * 100, 2),
    }

    out_path = Path("data/processed/metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved to {out_path}")
    print("\n>>> Use 'simulated_analyst_review_reduction_pct' and 'auto_clear_correctness'")
    print(">>> as the headline numbers in your LinkedIn description / resume.")


if __name__ == "__main__":
    evaluate()

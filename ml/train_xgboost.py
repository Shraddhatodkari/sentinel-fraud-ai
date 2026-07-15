"""
STEP 2A. Trains the XGBoost fraud classifier.

Run:
    python ml/train_xgboost.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "is_new_beneficiary",
    "account_age_days",
    "prior_alerts_for_account",
]
TARGET_COLUMN = "is_fraud"


def train():
    df = pd.read_csv("data/raw/transactions.csv")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train)

    out_dir = Path("data/processed/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "xgb_model.pkl")

    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    # Save train/test split for evaluate.py and isolation forest training
    X_test.assign(is_fraud=y_test).to_csv("data/processed/test_set.csv", index=False)
    X_train.assign(is_fraud=y_train).to_csv("data/processed/train_set.csv", index=False)

    print(f"XGBoost model trained and saved to {out_dir / 'xgb_model.pkl'}")


if __name__ == "__main__":
    train()

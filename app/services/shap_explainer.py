"""
STEP 3. Wraps SHAP to turn a raw XGBoost prediction into a ranked list of
human-readable "why was this flagged" factors — the core explainability
layer that the Investigator agent consumes.
"""
import pandas as pd
import shap


class FraudExplainer:
    def __init__(self, xgb_model, feature_columns: list[str]):
        self.explainer = shap.TreeExplainer(xgb_model)
        self.feature_columns = feature_columns

    def top_factors(self, transaction: dict, top_n: int = 3) -> list[str]:
        X = pd.DataFrame([transaction])[self.feature_columns]
        shap_values = self.explainer.shap_values(X)

        # shap_values shape: (1, n_features)
        contributions = list(zip(self.feature_columns, shap_values[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        factors = []
        for feature, value in contributions[:top_n]:
            direction = "increased" if value > 0 else "decreased"
            factors.append(
                f"{self._readable(feature)} {direction} risk "
                f"(value: {transaction.get(feature)})"
            )
        return factors

    @staticmethod
    def _readable(feature: str) -> str:
        mapping = {
            "amount": "Transaction amount",
            "hour_of_day": "Time of day",
            "is_new_beneficiary": "New beneficiary flag",
            "account_age_days": "Account age",
            "prior_alerts_for_account": "Prior alerts on account",
        }
        return mapping.get(feature, feature)

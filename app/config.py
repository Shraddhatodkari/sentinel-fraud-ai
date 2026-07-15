"""
Centralized app configuration, loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM keys (only one needs to be set depending on agent model string)
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "sentinel"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "sentinel-fraud-ai"

    # Business logic
    auto_clear_confidence_threshold: float = 0.85
    environment: str = "development"

    # Paths
    xgb_model_path: str = "data/processed/models/xgb_model.pkl"
    iso_forest_model_path: str = "data/processed/models/iso_forest_model.pkl"
    feature_columns_path: str = "data/processed/models/feature_columns.json"


settings = Settings()

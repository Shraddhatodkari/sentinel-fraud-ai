"""
STEP 7. Configures LangSmith tracing for the agent pipeline.

LangSmith hooks into env vars automatically (LANGCHAIN_TRACING_V2,
LANGCHAIN_API_KEY, LANGCHAIN_PROJECT) — this module just ensures they're
set from our settings object and gives us a place to add custom trace
tags if needed later.
"""
import os

from app.config import settings


def configure_tracing():
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        print(f"[tracing] LangSmith tracing enabled for project: {settings.langchain_project}")
    else:
        print("[tracing] LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true in .env to enable)")

"""
FastAPI application entrypoint.

Run:
    uvicorn app.main:app --reload
"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from app.routers import alerts
from app.services.tracing import configure_tracing

app = FastAPI(
    title="Sentinel — Agentic AI Analyst for Fraud Alert Triage",
    description=(
        "Investigates fraud alerts like a human analyst would: gathers "
        "context (risk scores, SHAP explanations, network connections), "
        "then auto-clears high-confidence low-risk alerts or escalates "
        "ambiguous/high-risk ones with a plain-English summary."
    ),
    version="0.1.0",
)

configure_tracing()
app.include_router(alerts.router)


@app.get("/")
async def root():
    return {
        "service": "sentinel-fraud-ai",
        "docs": "/docs",
        "status": "running",
    }
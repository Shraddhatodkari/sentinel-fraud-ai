"""
Run: pytest tests/test_api.py
Note: requires MongoDB and an LLM API key configured in .env to fully pass,
since the /investigate endpoint calls the Disposition Agent + audit log.
The /health check does not need either.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/alerts/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

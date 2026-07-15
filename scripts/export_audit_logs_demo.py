"""
Exports MongoDB audit logs to a local JSON file so the dashboard's Audit
Log page can render them without needing a live DB connection during a
demo/screenshot session.

Run (after you've processed a few /alerts/investigate calls):
    python scripts/export_audit_logs_demo.py
"""
import asyncio
import json
from pathlib import Path

from app.db.mongo_client import get_db


async def export_logs():
    db = get_db()
    logs = await db["audit_logs"].find({}).to_list(length=1000)
    for log in logs:
        log["_id"] = str(log["_id"])  # ObjectId isn't JSON-serializable

    out_path = Path("data/processed/audit_logs_demo.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(logs, f, indent=2, default=str)

    print(f"Exported {len(logs)} audit log records -> {out_path}")


if __name__ == "__main__":
    asyncio.run(export_logs())

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

# Unit tests should not require a live Postgres server just to construct TestClient(app).
os.environ.setdefault("APP_INIT_DB_ON_STARTUP", "0")


@pytest.fixture(autouse=True)
def _stub_run_store(monkeypatch: pytest.MonkeyPatch):
    runs: dict[str, dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []

    def _run_key(run_id: uuid.UUID | str) -> str:
        return str(run_id)

    def _insert_run(run_id, question, params, status, artifacts_path):
        runs[_run_key(run_id)] = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc),
            "question": question,
            "params": params or {},
            "status": status,
            "artifacts_path": artifacts_path,
        }

    def _get_run(run_id):
        row = runs.get(_run_key(run_id))
        if row is None:
            return None
        return dict(row)

    def _list_runs(limit: int, offset: int):
        rows = list(runs.values())
        rows.sort(key=lambda row: row.get("created_at") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return [dict(row) for row in rows[offset: offset + limit]]

    def _insert_audit(run_id, actor, action, payload):
        audits.append(
            {
                "run_id": None if run_id is None else _run_key(run_id),
                "actor": actor,
                "action": action,
                "payload": payload,
            }
        )

    def _update_run_status(run_id, status):
        key = _run_key(run_id)
        row = runs.get(key)
        if row is None:
            return False
        row["status"] = status
        return True

    monkeypatch.setattr("app.routes.runs.insert_run", _insert_run)
    monkeypatch.setattr("app.routes.runs.get_run", _get_run)
    monkeypatch.setattr("app.routes.runs.list_runs", _list_runs)
    monkeypatch.setattr("app.routes.runs.insert_audit", _insert_audit)

    monkeypatch.setattr("app.routes.tools.insert_run", _insert_run)
    monkeypatch.setattr("app.routes.tools.get_run", _get_run)
    monkeypatch.setattr("app.routes.tools.insert_audit", _insert_audit)
    monkeypatch.setattr("app.routes.tools.update_run_status", _update_run_status)

    monkeypatch.setattr("app.agent.runner.insert_run", _insert_run)
    monkeypatch.setattr("app.agent.runner.get_run", _get_run)
    monkeypatch.setattr("app.agent.runner.insert_audit", _insert_audit)
    monkeypatch.setattr("app.agent.runner.update_run_status", _update_run_status)

    return {"runs": runs, "audits": audits}

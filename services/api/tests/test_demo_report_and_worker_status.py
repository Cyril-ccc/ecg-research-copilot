from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.ecg_task_queue import get_queue_dirs as get_ecg_queue_dirs
from app.core.report_task_queue import get_queue_dirs as get_report_queue_dirs
from app.main import app


def test_worker_status_returns_queue_counts_and_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setattr("app.routes.tools.ARTIFACTS_DIR", tmp_path)

    ecg_dirs = get_ecg_queue_dirs(tmp_path)
    report_dirs = get_report_queue_dirs(tmp_path)

    (ecg_dirs["pending"] / "a.json").write_text("{}", encoding="utf-8")
    (ecg_dirs["running"] / "b.json").write_text("{}", encoding="utf-8")
    (report_dirs["pending"] / "c.json").write_text("{}", encoding="utf-8")

    heartbeat = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": "idle",
        "processed_jobs": 3,
    }
    (ecg_dirs["base"] / "worker_heartbeat.json").write_text(
        json.dumps(heartbeat),
        encoding="utf-8",
    )

    client = TestClient(app)
    resp = client.get("/tools/worker_status")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["ok"] is True
    assert body["ecg_worker"]["online"] is True
    assert body["ecg_worker"]["state"] == "idle"
    assert body["queues"]["ecg_features"]["pending"] == 1
    assert body["queues"]["ecg_features"]["running"] == 1
    assert body["queues"]["ecg_features"]["backlog"] == 2
    assert body["queues"]["report"]["pending"] == 1
    assert body["queues"]["report"]["backlog"] == 1


def test_demo_report_enqueues_and_succeeds(tmp_path, monkeypatch):
    run_id = str(uuid.uuid4())
    created_runs: set[str] = set()
    status_updates: list[str] = []
    audit_actions: list[str] = []

    def _fake_get_run(run_uuid):
        rid = str(run_uuid)
        if rid in created_runs:
            return {"run_id": rid, "status": "CREATED"}
        return None

    def _fake_insert_run(run_uuid, question, params, status, artifacts_path):
        _ = (question, params, status, artifacts_path)
        created_runs.add(str(run_uuid))

    def _fake_insert_audit(_run_id, _actor, action, _payload):
        audit_actions.append(action)

    def _fake_update_run_status(_run_uuid, status):
        status_updates.append(status)
        return True

    def _fake_run_demo_report_pipeline(*, run_id: str, sample_n: int, question: str, config: dict):
        run_dir = Path(tmp_path) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.md").write_text("# demo report", encoding="utf-8")
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"sample_n": sample_n, "question": question, "config": config}),
            encoding="utf-8",
        )
        return {
            "run_id": run_id,
            "sample_n": sample_n,
            "report_path": str(run_dir / "report.md"),
            "metadata_path": str(run_dir / "run_metadata.json"),
        }

    monkeypatch.setattr("app.routes.tools.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("app.routes.tools.get_run", _fake_get_run)
    monkeypatch.setattr("app.routes.tools.insert_run", _fake_insert_run)
    monkeypatch.setattr("app.routes.tools.insert_audit", _fake_insert_audit)
    monkeypatch.setattr("app.routes.tools.update_run_status", _fake_update_run_status)
    monkeypatch.setattr("app.routes.tools._run_demo_report_pipeline", _fake_run_demo_report_pipeline)

    client = TestClient(app)
    resp = client.post(
        "/tools/demo_report",
        json={
            "run_id": run_id,
            "sample_n": 8,
            "question": "pytest demo report",
            "config": {},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["run_id"] == run_id
    assert body["queue_status"] == "QUEUED"

    run_dir = Path(tmp_path) / run_id
    result_path = run_dir / "demo_report_task_result.json"
    assert result_path.exists()
    result = json.loads(result_path.read_text("utf-8"))
    assert result["status"] == "DEMO_REPORT_SUCCEEDED"

    assert run_id in created_runs
    assert status_updates == ["DEMO_REPORT_RUNNING", "DEMO_REPORT_SUCCEEDED"]
    assert "demo_report_request" in audit_actions

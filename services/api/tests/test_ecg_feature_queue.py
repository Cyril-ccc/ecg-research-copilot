import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_extract_ecg_features_enqueues_job(tmp_path, monkeypatch):
    run_id = str(uuid.uuid4())
    captured_audits: list[tuple[str, dict]] = []

    def _fake_get_run(_run_uuid):
        return {"run_id": run_id, "status": "CREATED"}

    def _fake_insert_audit(_run_id, _actor, action, payload):
        captured_audits.append((action, payload))

    monkeypatch.setattr("app.routes.tools.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("app.routes.tools.get_run", _fake_get_run)
    monkeypatch.setattr("app.routes.tools.insert_audit", _fake_insert_audit)

    client = TestClient(app)
    resp = client.post(
        "/tools/extract_ecg_features",
        json={
            "run_id": run_id,
            "record_ids": ["r1", "r2", "r1"],
            "params": {"feature_version": "v1.0"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["run_id"] == run_id
    assert body["queue_status"] == "QUEUED"

    pending_dir = Path(tmp_path) / "_queue" / "ecg_features" / "pending"
    queue_files = sorted(pending_dir.glob("*.json"))
    assert len(queue_files) == 1

    payload = json.loads(queue_files[0].read_text("utf-8"))
    assert payload["run_id"] == run_id
    assert payload["record_ids"] == ["r1", "r2"]
    assert payload["params"]["feature_version"] == "v1.0"

    assert any(action == "extract_ecg_features_enqueue" for action, _ in captured_audits)

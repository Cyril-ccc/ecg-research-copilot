from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_generate_report_enqueues_and_succeeds(tmp_path, monkeypatch):
    run_id = str(uuid.uuid4())
    captured_audits: list[tuple[str, dict]] = []
    status_updates: list[str] = []

    def _fake_get_run(_run_uuid):
        return {"run_id": run_id, "status": "CREATED"}

    def _fake_insert_audit(_run_id, _actor, action, payload):
        captured_audits.append((action, payload))

    def _fake_update_run_status(_run_uuid, status):
        status_updates.append(status)
        return True

    def _fake_run_generate_report_pipeline(*, run_id: str, config: dict):
        run_dir = Path(tmp_path) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report_path = run_dir / "report.md"
        meta_path = run_dir / "run_metadata.json"
        report_path.write_text("# report", encoding="utf-8")
        meta_path.write_text(json.dumps({"ok": True, "config": config}), encoding="utf-8")
        return report_path, meta_path, {"ok": True}

    monkeypatch.setattr("app.routes.tools.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("app.routes.tools.get_run", _fake_get_run)
    monkeypatch.setattr("app.routes.tools.insert_audit", _fake_insert_audit)
    monkeypatch.setattr("app.routes.tools.update_run_status", _fake_update_run_status)
    monkeypatch.setattr("app.routes.tools._run_generate_report_pipeline", _fake_run_generate_report_pipeline)

    client = TestClient(app)
    resp = client.post(
        "/tools/generate_report",
        json={
            "run_id": run_id,
            "config": {
                "question": "test report",
                "params": {"compare_features": ["mean_hr"]},
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["run_id"] == run_id
    assert body["queue_status"] == "QUEUED"

    run_dir = Path(tmp_path) / run_id
    assert (run_dir / "report.md").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "report_task_result.json").exists()

    done_dir = Path(tmp_path) / "_queue" / "report" / "done"
    assert len(list(done_dir.glob("*.json"))) == 1
    assert status_updates == ["REPORT_RUNNING", "REPORT_SUCCEEDED"]
    assert any(action == "generate_report_request" for action, _ in captured_audits)


def test_generate_report_marks_failed_when_pipeline_errors(tmp_path, monkeypatch):
    run_id = str(uuid.uuid4())
    status_updates: list[str] = []

    def _fake_get_run(_run_uuid):
        return {"run_id": run_id, "status": "CREATED"}

    def _fake_update_run_status(_run_uuid, status):
        status_updates.append(status)
        return True

    def _fake_run_generate_report_pipeline(*, run_id: str, config: dict):
        raise RuntimeError(f"boom {run_id} {config}")

    monkeypatch.setattr("app.routes.tools.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("app.routes.tools.get_run", _fake_get_run)
    monkeypatch.setattr("app.routes.tools.insert_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.routes.tools.update_run_status", _fake_update_run_status)
    monkeypatch.setattr("app.routes.tools._run_generate_report_pipeline", _fake_run_generate_report_pipeline)

    client = TestClient(app)
    resp = client.post(
        "/tools/generate_report",
        json={
            "run_id": run_id,
            "config": {"question": "will fail"},
        },
    )
    assert resp.status_code == 200, resp.text

    run_dir = Path(tmp_path) / run_id
    result_path = run_dir / "report_task_result.json"
    assert result_path.exists()
    result = json.loads(result_path.read_text("utf-8"))
    assert result["status"] == "REPORT_FAILED"

    failed_dir = Path(tmp_path) / "_queue" / "report" / "failed"
    assert len(list(failed_dir.glob("*.json"))) == 1
    assert status_updates == ["REPORT_RUNNING", "REPORT_FAILED"]

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.runner import AgentRunResult
from app.main import app


def test_agent_ask_route_returns_result(tmp_path: Path, monkeypatch):
    run_id = "33333333-3333-3333-3333-333333333333"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    plan_path = run_dir / "plan.json"
    trace_path = run_dir / "agent_trace.json"
    final_answer_path = run_dir / "final_answer.md"
    facts_path = run_dir / "final_answer_facts.json"

    plan_path.write_text("{}", encoding="utf-8")
    trace_path.write_text("{}", encoding="utf-8")
    final_answer_path.write_text("# answer", encoding="utf-8")
    facts_path.write_text("{}", encoding="utf-8")

    class _FakeRunner:
        def run_question(self, **kwargs):  # noqa: ANN003
            _ = kwargs
            return AgentRunResult(
                run_id=run_id,
                status="SUCCEEDED",
                plan_path=plan_path,
                trace_path=trace_path,
                final_answer_path=final_answer_path,
                facts_path=facts_path,
                error=None,
                steps=[{"tool": "build_cohort", "status": "succeeded"}],
            )

    monkeypatch.setattr("app.routes.agent.AgentRunner", lambda: _FakeRunner())

    client = TestClient(app)
    resp = client.post("/agent/ask", json={"question": "test question"})
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["ok"] is True
    assert body["run_id"] == run_id
    assert body["status"] == "SUCCEEDED"
    assert body["step_count"] == 1
    assert body["final_answer_path"].endswith("final_answer.md")


def test_agent_ask_route_returns_403_on_policy_reject(tmp_path: Path, monkeypatch):
    run_id = "44444444-4444-4444-4444-444444444444"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    plan_path = run_dir / "plan.json"
    trace_path = run_dir / "agent_trace.json"
    plan_path.write_text("{}", encoding="utf-8")
    trace_path.write_text("{}", encoding="utf-8")

    class _FakeRunner:
        def run_question(self, **kwargs):  # noqa: ANN003
            _ = kwargs
            return AgentRunResult(
                run_id=run_id,
                status="REJECTED",
                plan_path=plan_path,
                trace_path=trace_path,
                final_answer_path=None,
                facts_path=None,
                error="request rejected: patient-level export is not allowed",
                steps=[],
            )

    monkeypatch.setattr("app.routes.agent.AgentRunner", lambda: _FakeRunner())

    client = TestClient(app)
    resp = client.post("/agent/ask", json={"question": "print subject_id"})
    assert resp.status_code == 403, resp.text
    detail = resp.json()["detail"]
    assert detail["run_id"] == run_id
    assert "rejected" in detail["message"]

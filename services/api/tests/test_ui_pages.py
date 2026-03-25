from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_ui_agent_page_available():
    client = TestClient(app)
    resp = client.get("/ui/agent")
    assert resp.status_code == 200, resp.text
    assert "Agent 问答页" in resp.text

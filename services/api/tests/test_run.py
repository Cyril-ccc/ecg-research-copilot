import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

def test_create_run(tmp_path, monkeypatch):
    # 把 ARTIFACTS_DIR 指到临时目录，避免污染本地 storage
    # 直接拦截并替换 runs.py 里已经导入的 ARTIFACTS_DIR 变量！
    monkeypatch.setattr("app.routes.runs.ARTIFACTS_DIR", tmp_path)

    client = TestClient(app)
    r = client.post("/runs", json={})
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    p = Path(tmp_path) / run_id / "params.json"
    assert p.exists()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["run_id"] == run_id

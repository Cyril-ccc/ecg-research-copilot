"""
Cohort smoke tests — DB is mocked so no real Postgres needed.

Strategy:
- Patch `app.routes.tools.get_data_conn` to return a context manager
  that yields a fake connection/cursor with pre-baked rows.
- Verify:
    1. build_cohort returns ok=True with expected row count
    2. cohort.parquet was written to the artifacts dir
    3. cohort_summary.json was written and numbers are self-consistent
    4. GET /runs/{run_id}/artifacts lists both files
"""
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

# ── fake DB rows ──────────────────────────────────────────────────────────────
FAKE_ROWS = [
    {"subject_id": 1, "hadm_id": 100, "index_time": "2180-01-01 08:00:00", "cohort_label": "electrolyte_hyperkalemia"},
    {"subject_id": 2, "hadm_id": 200, "index_time": "2180-03-15 12:00:00", "cohort_label": "electrolyte_hyperkalemia"},
    {"subject_id": 3, "hadm_id": 300, "index_time": "2181-06-20 06:30:00", "cohort_label": "electrolyte_hyperkalemia"},
    {"subject_id": 1, "hadm_id": 101, "index_time": "2182-09-05 14:00:00", "cohort_label": "electrolyte_hyperkalemia"},
]


@contextmanager
def _fake_data_conn():
    """Context manager that mimics psycopg connection with a fake cursor."""
    cur = MagicMock()
    cur.fetchall.return_value = [
        MagicMock(**{"__iter__": lambda s: iter(row.items()), "keys": lambda: row.keys(), **row})
        for row in FAKE_ROWS
    ]
    # Make dict(r) work: rows should behave like mappings
    cur.fetchall.return_value = FAKE_ROWS  # tools.py does [dict(r) for r in rows]

    txn = MagicMock()
    txn.__enter__ = MagicMock(return_value=txn)
    txn.__exit__ = MagicMock(return_value=False)

    cur_ctx = MagicMock()
    cur_ctx.__enter__ = MagicMock(return_value=cur)
    cur_ctx.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.transaction.return_value = txn
    conn.cursor.return_value = cur_ctx

    yield conn


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient with artifacts redirected to tmp_path and DB mocked."""
    monkeypatch.setattr("app.routes.runs.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("app.routes.tools.ARTIFACTS_DIR", tmp_path)
    # Mock the DB connection used by build_cohort
    monkeypatch.setattr("app.routes.tools.get_data_conn", _fake_data_conn)
    return TestClient(app)


# ── helpers ───────────────────────────────────────────────────────────────────
def _create_run(client: TestClient) -> str:
    r = client.post("/runs", json={"question": "smoke test"})
    assert r.status_code == 200
    return r.json()["run_id"]


def _build_cohort(client: TestClient, run_id: str) -> dict:
    r = client.post("/tools/build_cohort", json={
        "template_name": "electrolyte_hyperkalemia",
        "params": {"k_threshold": 5.5, "label_keyword": "potassium"},
        "run_id": run_id,
        "limit": 50,
    })
    assert r.status_code == 200, r.text
    return r.json()


# ── tests ─────────────────────────────────────────────────────────────────────

def test_build_cohort_returns_ok(client, tmp_path):
    run_id = _create_run(client)
    data = _build_cohort(client, run_id)

    assert data["ok"] is True
    assert data["template_name"] == "electrolyte_hyperkalemia"
    assert data["row_count"] == len(FAKE_ROWS)


def test_cohort_parquet_artifact_created(client, tmp_path):
    run_id = _create_run(client)
    _build_cohort(client, run_id)

    parquet_path = tmp_path / run_id / "cohort.parquet"
    assert parquet_path.exists(), "cohort.parquet not found in artifacts dir"
    assert parquet_path.stat().st_size > 0


def test_cohort_summary_json_created(client, tmp_path):
    run_id = _create_run(client)
    _build_cohort(client, run_id)

    summary_path = tmp_path / run_id / "cohort_summary.json"
    assert summary_path.exists(), "cohort_summary.json not found"

    summary = json.loads(summary_path.read_text("utf-8"))

    # Basic sanity
    assert "total_rows" in summary
    assert "distinct_subjects" in summary
    assert summary["total_rows"] == len(FAKE_ROWS)
    assert summary["distinct_subjects"] <= summary["total_rows"]


def test_summary_distinct_subjects_correct(client, tmp_path):
    run_id = _create_run(client)
    _build_cohort(client, run_id)

    summary = json.loads(
        (tmp_path / run_id / "cohort_summary.json").read_text("utf-8")
    )
    # FAKE_ROWS has subject_ids [1,2,3,1] → 3 distinct
    assert summary["distinct_subjects"] == 3


def test_missing_rates_populated(client, tmp_path):
    run_id = _create_run(client)
    _build_cohort(client, run_id)

    summary = json.loads(
        (tmp_path / run_id / "cohort_summary.json").read_text("utf-8")
    )
    assert "missing_rates" in summary
    # All FAKE_ROWS are complete — no missing values
    for col, rate in summary["missing_rates"].items():
        assert rate == 0.0, f"column {col} unexpectedly has missing rate {rate}"


def test_artifacts_endpoint_lists_files(client, tmp_path):
    run_id = _create_run(client)
    _build_cohort(client, run_id)

    r = client.get(f"/runs/{run_id}/artifacts")
    assert r.status_code == 200

    names = {item["name"] for item in r.json()}
    assert "cohort.parquet" in names
    assert "cohort_summary.json" in names


def test_summary_endpoint_returns_json(client, tmp_path):
    run_id = _create_run(client)
    _build_cohort(client, run_id)

    r = client.get(f"/runs/{run_id}/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total_rows"] == len(FAKE_ROWS)

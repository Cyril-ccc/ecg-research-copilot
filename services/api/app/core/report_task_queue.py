from __future__ import annotations

import json
from pathlib import Path
from typing import Any

QUEUE_NAME = "report"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def get_queue_dirs(artifacts_root: Path) -> dict[str, Path]:
    base = artifacts_root / "_queue" / QUEUE_NAME
    pending = base / "pending"
    running = base / "running"
    done = base / "done"
    failed = base / "failed"
    for p in (pending, running, done, failed):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "pending": pending,
        "running": running,
        "done": done,
        "failed": failed,
    }


def enqueue_report_job(artifacts_root: Path, payload: dict[str, Any]) -> Path:
    dirs = get_queue_dirs(artifacts_root)
    job_id = str(payload["job_id"])
    job_path = dirs["pending"] / f"{job_id}.json"
    _atomic_write_json(job_path, payload)
    return job_path


def move_report_job(job_path: Path, *, artifacts_root: Path, state: str) -> Path:
    dirs = get_queue_dirs(artifacts_root)
    if state not in ("running", "done", "failed"):
        raise ValueError(f"Unsupported state: {state}")
    target = dirs[state] / job_path.name
    job_path.replace(target)
    return target


def load_job_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


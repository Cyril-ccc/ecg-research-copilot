import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.config import ARTIFACTS_DIR
from app.db.models import get_run, insert_audit, insert_run, list_runs

logger = logging.getLogger("api.runs")
router = APIRouter()


class RunCreate(BaseModel):
    question: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class RunOut(BaseModel):
    run_id: str
    created_at: str | None = None
    question: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    status: str
    artifacts_path: str | None = None


class ArtifactMeta(BaseModel):
    name: str
    size_bytes: int


@router.post("/runs", response_model=RunOut)
def create_run(body: RunCreate):
    run_id = uuid.uuid4()
    status = "CREATED"
 
    # 1) artifacts 落盘
    run_dir: Path = ARTIFACTS_DIR / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": str(run_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": body.question,
        "params": body.params,
        "status": status,
        "artifacts_path": str(run_dir),
    }
    (run_dir / "params.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 2) 写 DB + audit
    insert_run(run_id, body.question, body.params, status, str(run_dir))
    insert_audit(run_id, "local", "CREATE_RUN", {"question": body.question, "params": body.params})

    logger.info("run_created", extra={"run_id": str(run_id)})
    return RunOut(**meta)


@router.get("/runs/{run_id}", response_model=RunOut)
def read_run(run_id: uuid.UUID):
    row = get_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="run not found")

    insert_audit(run_id, "local", "GET_RUN", {})
    logger.info("run_fetched", extra={"run_id": str(run_id)})

    return RunOut(
        run_id=str(row["run_id"]),
        created_at=row["created_at"].isoformat() if row.get("created_at") else None,
        question=row.get("question"),
        params=row.get("params") or {},
        status=row.get("status"),
        artifacts_path=row.get("artifacts_path"),
    )


@router.get("/runs", response_model=list[RunOut])
def read_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    rows = list_runs(limit=limit, offset=offset)
    insert_audit(None, "local", "LIST_RUNS", {"limit": limit, "offset": offset})
    logger.info("runs_listed", extra={"run_id": "-"})

    out: list[RunOut] = []
    for row in rows:
        out.append(
            RunOut(
                run_id=str(row["run_id"]),
                created_at=row["created_at"].isoformat() if row.get("created_at") else None,
                question=row.get("question"),
                params=row.get("params") or {},
                status=row.get("status"),
                artifacts_path=row.get("artifacts_path"),
            )
        )
    return out


@router.get("/runs/{run_id}/artifacts", response_model=list[ArtifactMeta])
def list_run_artifacts(run_id: uuid.UUID):
    row = get_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="run not found")

    run_dir: Path = ARTIFACTS_DIR / str(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        return []

    insert_audit(run_id, "local", "LIST_ARTIFACTS", {})
    logger.info("artifacts_listed", extra={"run_id": str(run_id)})
    
    out: list[ArtifactMeta] = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(run_dir)).replace("\\", "/")
            out.append(ArtifactMeta(name=rel, size_bytes=p.stat().st_size))

    return out


@router.get("/runs/{run_id}/summary")
def get_run_summary(run_id: uuid.UUID):
    """Return the contents of cohort_summary.json for a given run."""
    row = get_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="run not found")

    summary_path = ARTIFACTS_DIR / str(run_id) / "cohort_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="cohort_summary.json not found — run build_cohort first")

    logger.info("summary_fetched", extra={"run_id": str(run_id)})
    return json.loads(summary_path.read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/artifact/{artifact_path:path}")
def get_run_artifact(run_id: uuid.UUID, artifact_path: str):
    row = get_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="run not found")

    run_dir = (ARTIFACTS_DIR / str(run_id)).resolve()
    target = (run_dir / artifact_path).resolve()
    if not str(target).startswith(str(run_dir)):
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")

    insert_audit(run_id, "local", "GET_ARTIFACT", {"path": artifact_path})
    logger.info("artifact_fetched", extra={"run_id": str(run_id)})
    return FileResponse(path=target, filename=target.name)

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core.ecg_task_queue import (
    enqueue_ecg_feature_job,
    get_queue_dirs as get_ecg_queue_dirs,
)
from app.core.report_task_queue import (
    enqueue_report_job,
    get_queue_dirs as get_report_queue_dirs,
    load_job_payload as load_report_job_payload,
    move_report_job,
)
from app.core.config import (
    ARTIFACTS_DIR,
    DATA_SCHEMA,
    DEMO_DATA_DIR,
    DEMO_MANIFEST_PATH,
)
from app.core.schema_whitelist import load_whitelist
from app.core.sql_safety import SqlPolicy, validate_and_rewrite_sql
from app.db.models import get_run, insert_audit, insert_run, update_run_status
from app.db.session import get_data_conn
from app.core.cohort_templates import TEMPLATES

logger = logging.getLogger("api.tools")
router = APIRouter(prefix="/tools")

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunSqlRequest(BaseModel):
    sql: str
    limit: int = Field(default=1000, ge=1, le=5000)
    run_id: str | None = None  # 可选：把这次 SQL 调用归到某个 run 下

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # 允许 uuid 字符串；不合法就拒绝，避免写脏数据
        uuid.UUID(v)
        return v


class RunSqlResponse(BaseModel):
    ok: bool
    sql_sha256: str
    limited_sql: str | None = None
    row_count: int = 0
    rows: list[dict[str, Any]] = Field(default_factory=list)
    rejected_reason: str | None = None


class BuildCohortRequest(BaseModel):
    template_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    limit: int = Field(default=1000, ge=1, le=5000)  # 限制返回 rows 的数量（不等同于模板内部 limit）


class BuildCohortResponse(BaseModel):
    ok: bool
    template_name: str
    cohort_id: str | None = None
    cohort_table: str | None = None
    sql: str
    row_count: int
    rows: list[dict[str, Any]] = Field(default_factory=list)


class ExtractEcgFeaturesRequest(BaseModel):
    run_id: str
    record_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str) -> str:
        uuid.UUID(v)
        return v

    @field_validator("record_ids")
    @classmethod
    def validate_record_ids(cls, values: list[str]) -> list[str]:
        out = [str(v).strip() for v in values if str(v).strip()]
        if not out:
            raise ValueError("record_ids cannot be empty")
        return list(dict.fromkeys(out))


class ExtractEcgFeaturesResponse(BaseModel):
    ok: bool
    run_id: str
    job_id: str
    queue_status: str
    queued_at: str


class GenerateReportRequest(BaseModel):
    run_id: str
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str) -> str:
        uuid.UUID(v)
        return v


class GenerateReportResponse(BaseModel):
    ok: bool
    run_id: str
    job_id: str
    queue_status: str
    queued_at: str


class DemoReportRequest(BaseModel):
    run_id: str | None = None
    sample_n: int = Field(default=10, ge=1, le=200)
    question: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, v: str | None) -> str | None:
        if v is None or not str(v).strip():
            return None
        uuid.UUID(str(v).strip())
        return str(v).strip()


class DemoReportResponse(BaseModel):
    ok: bool
    run_id: str
    job_id: str
    queue_status: str
    queued_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def _run_generate_report_pipeline(*, run_id: str, config: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    try:
        from pipelines.generate_report import generate_report
    except Exception as exc:  # pragma: no cover - import path depends on runtime packaging
        raise RuntimeError(f"generate_report pipeline unavailable: {exc}") from exc

    question = config.get("question") if isinstance(config.get("question"), str) else None
    qc_parquet = config.get("qc_parquet")
    qc_override = Path(str(qc_parquet)).resolve() if qc_parquet else None

    params_json_arg: str | None = None
    if isinstance(config.get("params"), dict):
        params_json_arg = json.dumps(config["params"], ensure_ascii=True)
    elif isinstance(config.get("params_json"), str):
        params_json_arg = config["params_json"]

    return generate_report(
        run_id=run_id,
        artifacts_root=ARTIFACTS_DIR,
        question_arg=question,
        params_json_arg=params_json_arg,
        qc_path_override=qc_override,
    )


def _process_generate_report_job(job_path: Path, *, actor: str) -> None:
    running_path = job_path
    payload: dict[str, Any] = {}
    run_id = "unknown"
    job_id = job_path.stem
    run_uuid: uuid.UUID | None = None
    run_dir: Path | None = None

    try:
        payload = load_report_job_payload(job_path)
        run_id = str(payload.get("run_id", "")).strip()
        job_id = str(payload.get("job_id", job_path.stem))
        run_uuid = uuid.UUID(run_id)
        run_dir = ARTIFACTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        running_path = move_report_job(job_path, artifacts_root=ARTIFACTS_DIR, state="running")
        update_run_status(run_uuid, "REPORT_RUNNING")
        insert_audit(
            run_uuid,
            actor,
            "REPORT_RUNNING",
            {
                "job_id": job_id,
                "started_at": _utc_now_iso(),
            },
        )

        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        report_path, metadata_path, _meta = _run_generate_report_pipeline(run_id=run_id, config=config)
        _atomic_write_json(
            run_dir / "report_task_result.json",
            {
                "job_id": job_id,
                "run_id": run_id,
                "status": "REPORT_SUCCEEDED",
                "finished_at": _utc_now_iso(),
                "report_path": str(report_path),
                "metadata_path": str(metadata_path),
            },
        )

        update_run_status(run_uuid, "REPORT_SUCCEEDED")
        insert_audit(
            run_uuid,
            actor,
            "REPORT_SUCCEEDED",
            {
                "job_id": job_id,
                "report_path": str(report_path),
                "metadata_path": str(metadata_path),
            },
        )
        move_report_job(running_path, artifacts_root=ARTIFACTS_DIR, state="done")
    except Exception as exc:
        logger.exception("generate_report_job_failed run_id=%s job_id=%s err=%s", run_id, job_id, exc)
        try:
            if run_uuid is not None:
                update_run_status(run_uuid, "REPORT_FAILED")
                insert_audit(
                    run_uuid,
                    actor,
                    "REPORT_FAILED",
                    {
                        "job_id": job_id,
                        "error": str(exc),
                    },
                )
        except Exception:
            logger.exception("generate_report_failed_to_update_run_status run_id=%s", run_id)

        if run_dir is not None:
            _atomic_write_json(
                run_dir / "report_task_result.json",
                {
                    "job_id": job_id,
                    "run_id": run_id,
                    "status": "REPORT_FAILED",
                    "finished_at": _utc_now_iso(),
                    "error": str(exc),
                },
            )
        try:
            move_report_job(running_path, artifacts_root=ARTIFACTS_DIR, state="failed")
        except Exception:
            logger.exception("generate_report_failed_to_move_queue_file path=%s", running_path)


def _ensure_demo_run(*, run_id: str, question: str, params: dict[str, Any]) -> uuid.UUID:
    run_uuid = uuid.UUID(run_id)
    if get_run(run_uuid):
        return run_uuid

    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    created_at = _utc_now_iso()
    meta = {
        "run_id": run_id,
        "created_at": created_at,
        "question": question,
        "params": params,
        "status": "CREATED",
        "artifacts_path": str(run_dir),
    }
    (run_dir / "params.json").write_text(
        json.dumps(meta, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    insert_run(run_uuid, question, params, "CREATED", str(run_dir))
    return run_uuid


def _run_demo_report_pipeline(
    *,
    run_id: str,
    sample_n: int,
    question: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        from pipelines.demo_report import run_demo_report
    except Exception as exc:  # pragma: no cover - runtime import path may vary
        raise RuntimeError(f"demo_report pipeline unavailable: {exc}") from exc

    artifacts_root = Path(str(config.get("artifacts_root", ARTIFACTS_DIR))).resolve()
    data_dir = Path(str(config.get("data_dir", DEMO_DATA_DIR))).resolve()
    global_manifest = Path(str(config.get("global_manifest", DEMO_MANIFEST_PATH))).resolve()

    return run_demo_report(
        run_id=run_id,
        artifacts_root=artifacts_root,
        data_dir=data_dir,
        global_manifest_path=global_manifest,
        sample_n=sample_n,
        question=question,
    )


def _process_demo_report_job(payload: dict[str, Any], *, actor: str) -> None:
    run_id = str(payload.get("run_id", "")).strip()
    job_id = str(payload.get("job_id", "unknown"))
    sample_n = int(payload.get("sample_n", 10))
    question = str(payload.get("question", "")).strip() or "Demo report run"
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    run_dir = ARTIFACTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_uuid = uuid.UUID(run_id)

    _atomic_write_json(
        run_dir / "demo_report_task_result.json",
        {
            "job_id": job_id,
            "run_id": run_id,
            "status": "DEMO_REPORT_RUNNING",
            "started_at": _utc_now_iso(),
            "sample_n": sample_n,
        },
    )
    update_run_status(run_uuid, "DEMO_REPORT_RUNNING")
    insert_audit(
        run_uuid,
        actor,
        "DEMO_REPORT_RUNNING",
        {"job_id": job_id, "sample_n": sample_n},
    )

    try:
        summary = _run_demo_report_pipeline(
            run_id=run_id,
            sample_n=sample_n,
            question=question,
            config=config,
        )
        _atomic_write_json(
            run_dir / "demo_report_task_result.json",
            {
                "job_id": job_id,
                "run_id": run_id,
                "status": "DEMO_REPORT_SUCCEEDED",
                "finished_at": _utc_now_iso(),
                "sample_n": sample_n,
                "report_path": summary.get("report_path"),
                "metadata_path": summary.get("metadata_path"),
                "summary_path": str(run_dir / "demo_report_summary.json"),
            },
        )
        update_run_status(run_uuid, "DEMO_REPORT_SUCCEEDED")
        insert_audit(
            run_uuid,
            actor,
            "DEMO_REPORT_SUCCEEDED",
            {
                "job_id": job_id,
                "report_path": summary.get("report_path"),
                "metadata_path": summary.get("metadata_path"),
            },
        )
    except Exception as exc:
        logger.exception("demo_report_job_failed run_id=%s job_id=%s err=%s", run_id, job_id, exc)
        _atomic_write_json(
            run_dir / "demo_report_task_result.json",
            {
                "job_id": job_id,
                "run_id": run_id,
                "status": "DEMO_REPORT_FAILED",
                "finished_at": _utc_now_iso(),
                "error": str(exc),
            },
        )
        update_run_status(run_uuid, "DEMO_REPORT_FAILED")
        insert_audit(
            run_uuid,
            actor,
            "DEMO_REPORT_FAILED",
            {"job_id": job_id, "error": str(exc)},
        )


def _queue_counts(dirs: dict[str, Path]) -> dict[str, int]:
    pending_n = len(list(dirs["pending"].glob("*.json")))
    running_n = len(list(dirs["running"].glob("*.json")))
    done_n = len(list(dirs["done"].glob("*.json")))
    failed_n = len(list(dirs["failed"].glob("*.json")))
    return {
        "pending": pending_n,
        "running": running_n,
        "done": done_n,
        "failed": failed_n,
        "backlog": pending_n + running_n,
    }


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@router.post("/extract_ecg_features", response_model=ExtractEcgFeaturesResponse)
def extract_ecg_features(body: ExtractEcgFeaturesRequest, x_actor: str | None = Header(default=None)):
    actor = x_actor or "local"
    run_uuid = uuid.UUID(body.run_id)
    if not get_run(run_uuid):
        raise HTTPException(status_code=404, detail="run not found")

    job_id = uuid.uuid4().hex
    queued_at = _utc_now_iso()
    payload = {
        "job_id": job_id,
        "run_id": body.run_id,
        "record_ids": body.record_ids,
        "params": body.params,
        "queued_at": queued_at,
    }
    job_path = enqueue_ecg_feature_job(ARTIFACTS_DIR, payload)

    insert_audit(
        run_uuid,
        actor,
        "extract_ecg_features_enqueue",
        {
            "job_id": job_id,
            "record_count": len(body.record_ids),
            "params": body.params,
            "queue_path": str(job_path),
        },
    )
    logger.info("extract_ecg_features_enqueued", extra={"run_id": body.run_id, "job_id": job_id})
    return ExtractEcgFeaturesResponse(
        ok=True,
        run_id=body.run_id,
        job_id=job_id,
        queue_status="QUEUED",
        queued_at=queued_at,
    )




@router.post("/generate_report", response_model=GenerateReportResponse)
def generate_report(
    body: GenerateReportRequest,
    background_tasks: BackgroundTasks,
    x_actor: str | None = Header(default=None),
):
    actor = x_actor or "local"
    run_uuid = uuid.UUID(body.run_id)
    if not get_run(run_uuid):
        raise HTTPException(status_code=404, detail="run not found")

    job_id = uuid.uuid4().hex
    queued_at = _utc_now_iso()
    payload = {
        "job_id": job_id,
        "run_id": body.run_id,
        "config": body.config,
        "queued_at": queued_at,
        "actor": actor,
    }
    job_path = enqueue_report_job(ARTIFACTS_DIR, payload)

    insert_audit(
        run_uuid,
        actor,
        "generate_report_request",
        {
            "job_id": job_id,
            "config": body.config,
            "queue_path": str(job_path),
        },
    )
    background_tasks.add_task(_process_generate_report_job, job_path, actor=actor)
    logger.info("generate_report_enqueued", extra={"run_id": body.run_id, "job_id": job_id})
    return GenerateReportResponse(
        ok=True,
        run_id=body.run_id,
        job_id=job_id,
        queue_status="QUEUED",
        queued_at=queued_at,
    )


@router.post("/demo_report", response_model=DemoReportResponse)
def demo_report(
    body: DemoReportRequest,
    background_tasks: BackgroundTasks,
    x_actor: str | None = Header(default=None),
):
    actor = x_actor or "local"
    run_id = body.run_id or str(uuid.uuid4())
    question = body.question or "One-click demo report"
    run_uuid = _ensure_demo_run(
        run_id=run_id,
        question=question,
        params={"sample_n": body.sample_n, "source": "demo_report_api"},
    )

    job_id = uuid.uuid4().hex
    queued_at = _utc_now_iso()
    payload = {
        "job_id": job_id,
        "run_id": run_id,
        "sample_n": body.sample_n,
        "question": question,
        "config": body.config,
        "queued_at": queued_at,
    }
    insert_audit(
        run_uuid,
        actor,
        "demo_report_request",
        {
            "job_id": job_id,
            "sample_n": body.sample_n,
            "config": body.config,
        },
    )
    background_tasks.add_task(_process_demo_report_job, payload, actor=actor)
    logger.info("demo_report_enqueued", extra={"run_id": run_id, "job_id": job_id})
    return DemoReportResponse(
        ok=True,
        run_id=run_id,
        job_id=job_id,
        queue_status="QUEUED",
        queued_at=queued_at,
    )


@router.get("/worker_status")
def worker_status():
    ecg_dirs = get_ecg_queue_dirs(ARTIFACTS_DIR)
    report_dirs = get_report_queue_dirs(ARTIFACTS_DIR)

    ecg_counts = _queue_counts(ecg_dirs)
    report_counts = _queue_counts(report_dirs)

    heartbeat_path = ecg_dirs["base"] / "worker_heartbeat.json"
    heartbeat: dict[str, Any] = {}
    if heartbeat_path.exists():
        try:
            heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        except Exception:
            heartbeat = {}

    last_seen = heartbeat.get("updated_at")
    parsed = _parse_iso_datetime(last_seen if isinstance(last_seen, str) else None)
    age_seconds: float | None = None
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()
    online = bool(age_seconds is not None and age_seconds <= 15.0)

    return {
        "ok": True,
        "ecg_worker": {
            "online": online,
            "state": heartbeat.get("state", "unknown"),
            "last_seen": last_seen,
            "age_seconds": age_seconds,
            "heartbeat_path": str(heartbeat_path),
        },
        "queues": {
            "ecg_features": ecg_counts,
            "report": report_counts,
        },
    }


@router.post("/run_sql", response_model=RunSqlResponse)
def run_sql(body: RunSqlRequest, x_actor: str | None = Header(default=None)):
    actor = x_actor or "local"
    raw_sql = body.sql
    sql_hash = sha256_hex(raw_sql)

    audit_run_id = uuid.UUID(body.run_id) if body.run_id else None

    # 1) 读取 whitelist，并转成 {schema: {table: set(cols)}} 结构
    wl_raw = load_whitelist()
    allow = wl_raw.get("allow", {}) or {}
    wl: dict[str, dict[str, set[str]]] = {}
    for schema, tables in allow.items():
        wl[schema] = {}
        if not isinstance(tables, dict):
            continue
        for table, cols in tables.items():
            if isinstance(cols, list):
                wl[schema][table] = set(cols)
            else:
                wl[schema][table] = set()

    # 2) AST 校验 + 重写（强制 LIMIT / 表字段白名单 / 禁止危险语句）
    policy = SqlPolicy(
        max_limit=body.limit,          # 你也可以改成固定上限，比如 min(body.limit, 1000)
        allow_schema="public",
        whitelist=wl,
        require_qualified_columns=True,  # MVP：要求列带前缀，且禁止 *
    )

    try:
        limited_sql, meta = validate_and_rewrite_sql(raw_sql, policy)
        allowed = True
        reason = "ok"
    except Exception as e:
        allowed = False
        reason = str(e)
        limited_sql = None
        meta = {}

    # 3) 写审计：无论允许/拒绝都写
    insert_audit(
        audit_run_id,
        actor,
        "run_sql_request",
        {
            "sql_sha256": sql_hash,
            "allowed": allowed,
            "reason": reason,
            "limit": body.limit,
            "sql_preview": raw_sql[:200],
            "tables_used": meta.get("tables_used", []),
        },
    )

    if not allowed:
        logger.warning("run_sql_rejected", extra={"run_id": body.run_id or "-"})
        raise HTTPException(status_code=400, detail=f"unsafe sql: {reason}")

    # 4) 再写一条审计：实际执行的 SQL（只保存 preview）
    insert_audit(
        audit_run_id,
        actor,
        "run_sql_execute",
        {
            "sql_sha256": sql_hash,
            "limited_sql_preview": (limited_sql or "")[:200],
            "limit": body.limit,
            "tables_used": meta.get("tables_used", []),
        },
    )

    # 5) 执行：强制只读连接 + statement_timeout + search_path
    #    这些都是“硬护栏”，就算校验漏了也更安全
    out_rows: list[dict] = []
    with get_data_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET LOCAL transaction_read_only = on;")
                cur.execute("SET LOCAL statement_timeout = '8s';")
                cur.execute("SET LOCAL search_path = public;")
                cur.execute(limited_sql)
                rows = cur.fetchall()
                out_rows = [dict(r) for r in rows]

    logger.info("run_sql_ok", extra={"run_id": body.run_id or "-"})
    return RunSqlResponse(
        ok=True,
        sql_sha256=sql_hash,
        limited_sql=limited_sql,
        row_count=len(out_rows),
        rows=out_rows,
    )





@router.post("/build_cohort", response_model=BuildCohortResponse)
def build_cohort(body: BuildCohortRequest, x_actor: str | None = Header(default=None)):
    actor = x_actor or "local"
    audit_run_id = uuid.UUID(body.run_id) if body.run_id else None
    cohort_id: str | None = None

    if body.template_name not in TEMPLATES:
        insert_audit(audit_run_id, actor, "build_cohort_request", {
            "allowed": False,
            "reason": "unknown template",
            "template_name": body.template_name,
            "params": body.params,
        })
        raise HTTPException(status_code=400, detail="unknown template_name")

    # 1) 生成 SQL（模板化，不依赖 LLM）
    try:
        result = TEMPLATES[body.template_name](body.params)
        sql = result.sql
    except Exception as e:
        insert_audit(audit_run_id, actor, "build_cohort_request", {
            "allowed": False,
            "reason": str(e),
            "template_name": body.template_name,
            "params": body.params,
        })
        raise HTTPException(status_code=400, detail=f"invalid params: {e}")

    # 2) 审计：SQL 生成（很重要）
    insert_audit(audit_run_id, actor, "build_cohort_request", {
        "allowed": True,
        "template_name": body.template_name,
        "params": result.params,
        "sql": sql,
        "sql_preview": sql[:300],
        "sql_sha256": sha256_hex(sql),
    })

    # 3) 执行：走安全 SQL 笼子（复用 validate_and_rewrite_sql）
    #    这里为了简单：直接用你现在的 run_sql 校验器/只读连接
    wl_raw = load_whitelist()
    allow = wl_raw.get("allow", {}) or {}
    wl: dict[str, dict[str, set[str]]] = {}
    for schema, tables in allow.items():
        wl[schema] = {}
        if not isinstance(tables, dict):
            continue
        for table, cols in tables.items():
            wl[schema][table] = set(cols) if isinstance(cols, list) else set()

    policy = SqlPolicy(
        max_limit=body.limit,
        allow_schema=DATA_SCHEMA,
        whitelist=wl,
        require_qualified_columns=True,
    )

    try:
        limited_sql, meta = validate_and_rewrite_sql(sql, policy)
    except Exception as e:
        insert_audit(audit_run_id, actor, "build_cohort_execute", {
            "allowed": False,
            "template_name": body.template_name,
            "reason": str(e),
            "sql_sha256": sha256_hex(sql),
        })
        raise HTTPException(status_code=400, detail=f"unsafe cohort sql: {e}")

    with get_data_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET LOCAL transaction_read_only = on;")
                cur.execute("SET LOCAL statement_timeout = '120s';")
                cur.execute(f"SET LOCAL search_path = {DATA_SCHEMA}, public;")
                cur.execute(limited_sql)
                rows = [dict(r) for r in cur.fetchall()]

    cohort_id = uuid.uuid4().hex

    if body.run_id and rows:
        run_dir = ARTIFACTS_DIR / body.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(rows)
        # Compute summary
        summary = {
            "distinct_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
            "total_rows": len(df),
        }
        
        if "gender" in df.columns:
            summary["gender_distribution"] = df["gender"].value_counts().to_dict()
        if "age" in df.columns:
            summary["age_distribution"] = {
                "min": float(df["age"].min()),
                "max": float(df["age"].max()),
                "mean": float(df["age"].mean())
            }
            
        if "index_time" in df.columns:
            summary["index_time_range"] = {
                "min": str(df["index_time"].min()),
                "max": str(df["index_time"].max()),
            }

        # Key variables missing rate
        missing_rates = (df.isnull().sum() / len(df)).to_dict()
        summary["missing_rates"] = {k: float(v) for k, v in missing_rates.items()}
        
        (run_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        df.to_parquet(run_dir / "cohort.parquet", index=False)

    # 4) 审计：执行结果
    insert_audit(audit_run_id, actor, "build_cohort_execute", {
        "allowed": True,
        "template_name": body.template_name,
        "cohort_id": cohort_id,
        "row_count": len(rows),
        "tables_used": meta.get("tables_used", []),
        "validated_sql": limited_sql,
        "sql_sha256": sha256_hex(sql),
    })

    return BuildCohortResponse(
        ok=True,
        template_name=body.template_name,
        cohort_id=cohort_id,
        cohort_table=None,
        sql=sql,
        row_count=len(rows),
        rows=rows,
    )





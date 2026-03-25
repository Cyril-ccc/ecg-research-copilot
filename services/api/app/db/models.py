import uuid
from typing import Any

from psycopg.types.json import Json

from app.db.base import CREATE_TABLES_SQL
from app.db.session import get_conn


def init_db() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLES_SQL)
        conn.commit()


def insert_run(run_id: uuid.UUID, question: str | None, params: dict[str, Any], status: str, artifacts_path: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (run_id, question, params, status, artifacts_path)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (run_id, question, Json(params), status, artifacts_path),
            )
        conn.commit()


def get_run(run_id: uuid.UUID) -> dict[str, Any] | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def list_runs(limit: int, offset: int) -> list[dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM runs
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def insert_audit(run_id: uuid.UUID | None, actor: str, action: str, payload: dict[str, Any]):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_logs (run_id, actor, action, payload)
                VALUES (%s, %s, %s, %s)
                """,
                (run_id, actor, action, Json(payload)),
            )
        conn.commit()


def update_run_status(run_id: uuid.UUID, status: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                SET status = %s
                WHERE run_id = %s
                """,
                (status, run_id),
            )
            updated = cur.rowcount > 0
        conn.commit()
    return updated

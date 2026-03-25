"""Build storage/ecg_manifest.parquet from mimiciv.record_list + WFDB headers.

Designed for both demo and full MIMIC-IV-ECG datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import wfdb
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_URL = os.getenv(
    "DATA_DATABASE_URL",
    os.getenv("DATABASE_URL", "postgresql://ecg:ecg@localhost:5432/ecg"),
)
DEFAULT_SCHEMA = os.getenv("DATA_SCHEMA", "mimiciv")
DEFAULT_TABLE = "record_list"
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data" / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "storage" / "ecg_manifest.parquet"
DEFAULT_BATCH_SIZE = 2000
DEFAULT_LOG_EVERY = 5000
MANIFEST_FIELDS = [
    "record_id",
    "subject_id",
    "study_id",
    "file_name",
    "source",
    "fs",
    "n_samples",
    "n_leads",
    "ecg_time",
    "path",
]
MANIFEST_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("subject_id", pa.string()),
        ("study_id", pa.string()),
        ("file_name", pa.string()),
        ("source", pa.string()),
        ("fs", pa.int32()),
        ("n_samples", pa.int32()),
        ("n_leads", pa.int32()),
        ("ecg_time", pa.string()),
        ("path", pa.string()),
    ]
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ECG manifest parquet from record_list + WFDB headers"
    )
    parser.add_argument("--database-url", default=DEFAULT_DB_URL)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--record-list-table", default=DEFAULT_TABLE)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--where-sql", default="")
    parser.add_argument("--source", default="mimic_ecg")
    parser.add_argument("--strict-files", action="store_true")
    parser.add_argument("--strict-headers", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _validate_identifier(value: str, *, name: str) -> str:
    txt = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", txt):
        raise ValueError(f"invalid {name}: {value}")
    return txt


def _quote_ident(value: str) -> str:
    return f'"{_validate_identifier(value, name="sql_identifier")}"'


def _normalize_db_url(database_url: str) -> str:
    txt = str(database_url).strip()
    if txt.startswith("postgresql+psycopg://"):
        return "postgresql://" + txt.removeprefix("postgresql+psycopg://")
    if txt.startswith("postgres+psycopg://"):
        return "postgresql://" + txt.removeprefix("postgres+psycopg://")
    return txt


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    txt = str(value).strip()
    return txt or None


def _as_time_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    txt = str(value).strip()
    return txt or None


def _table_columns(conn: psycopg.Connection, *, schema: str, table_name: str) -> set[str]:
    sql = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %(schema)s
      AND table_name = %(table_name)s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, {"schema": schema, "table_name": table_name})
        return {str(r["column_name"]) for r in cur.fetchall()}


def _build_select_sql(
    *,
    schema: str,
    table_name: str,
    columns: list[str],
    where_sql: str,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    select_cols = ", ".join(_quote_ident(c) for c in columns)
    sql = f"SELECT {select_cols} FROM {_quote_ident(schema)}.{_quote_ident(table_name)}"
    if str(where_sql).strip():
        sql += f" WHERE ({where_sql.strip()})"
    params: dict[str, Any] = {}
    if int(limit) > 0:
        sql += " LIMIT %(limit)s"
        params["limit"] = int(limit)
    return sql, params


def _resolve_rel_path(row: dict[str, Any]) -> str | None:
    rel_path = _as_text(row.get("path"))
    if rel_path:
        return rel_path

    subject_id = _as_text(row.get("subject_id"))
    study_id = _as_text(row.get("study_id"))
    file_name = _as_text(row.get("file_name"))
    if subject_id and study_id and file_name:
        return f"files/p{subject_id}/s{study_id}/{file_name}"
    if subject_id and study_id:
        return f"files/p{subject_id}/s{study_id}/{study_id}"
    return None


def _build_record_id(row: dict[str, Any], rel_path: str) -> str | None:
    study_id = _as_text(row.get("study_id"))
    file_name = _as_text(row.get("file_name"))
    if study_id and file_name:
        return f"mimic_ecg_{study_id}_{file_name}"
    if study_id:
        return f"mimic_ecg_{study_id}"

    tail = _as_text(Path(rel_path).name)
    if tail:
        return f"mimic_ecg_{tail}"
    return None


def _read_header(data_dir: Path, rel_path: str) -> tuple[int, int, int]:
    header = wfdb.rdheader(str(data_dir / rel_path))
    return int(header.fs), int(header.sig_len), int(header.n_sig)


class _ParquetBufferWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.writer: pq.ParquetWriter | None = None
        self.rows_written = 0

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
        if self.writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                str(self.output_path),
                MANIFEST_SCHEMA,
                compression="snappy",
            )
        self.writer.write_table(table)
        self.rows_written += int(table.num_rows)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def build_manifest(
    *,
    database_url: str,
    schema: str,
    table_name: str,
    data_dir: Path,
    output_path: Path,
    summary_path: Path,
    where_sql: str,
    limit: int,
    batch_size: int,
    log_every: int,
    source: str,
    strict_files: bool,
    strict_headers: bool,
) -> dict[str, Any]:
    logger = logging.getLogger("build_manifest")
    if int(batch_size) < 1:
        raise ValueError("--batch-size must be >= 1")

    if not data_dir.exists():
        raise FileNotFoundError(f"data-dir not found: {data_dir}")

    db_url = _normalize_db_url(database_url)
    t0 = time.perf_counter()
    stats: dict[str, Any] = {
        "schema": schema,
        "table_name": table_name,
        "data_dir": str(data_dir),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "limit": int(limit),
        "where_sql": str(where_sql).strip(),
        "source": source,
        "strict_files": bool(strict_files),
        "strict_headers": bool(strict_headers),
        "rows_selected_db": 0,
        "rows_written": 0,
        "rows_skipped_missing_path": 0,
        "rows_skipped_missing_header_file": 0,
        "rows_skipped_header_error": 0,
        "rows_skipped_missing_record_id": 0,
        "rows_skipped_duplicate_record_id": 0,
        "first_errors": [],
    }
    seen_record_ids: set[str] = set()

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        table_cols = _table_columns(conn, schema=schema, table_name=table_name)
        if not table_cols:
            raise RuntimeError(f"table not found or empty columns: {schema}.{table_name}")

        required = {"subject_id", "study_id"}
        missing_required = sorted(required - table_cols)
        if missing_required:
            raise RuntimeError(
                f"{schema}.{table_name} missing required columns: {missing_required}"
            )

        path_capable = "path" in table_cols
        file_capable = "file_name" in table_cols
        if not path_capable and not file_capable:
            raise RuntimeError(
                f"{schema}.{table_name} must contain at least one of: path, file_name"
            )

        select_cols = [c for c in ("subject_id", "study_id", "file_name", "ecg_time", "path") if c in table_cols]
        sql, params = _build_select_sql(
            schema=schema,
            table_name=table_name,
            columns=select_cols,
            where_sql=where_sql,
            limit=limit,
        )
        logger.info("Querying %s.%s (%s columns)", schema, table_name, len(select_cols))

        writer = _ParquetBufferWriter(output_path=output_path)
        out_buffer: list[dict[str, Any]] = []

        try:
            with conn.cursor(name="ecg_manifest_cursor", row_factory=dict_row) as cur:
                cur.itersize = int(batch_size)
                cur.execute(sql, params)

                while True:
                    rows = cur.fetchmany(int(batch_size))
                    if not rows:
                        break

                    for row in rows:
                        stats["rows_selected_db"] += 1
                        row_obj = dict(row)

                        rel_path = _resolve_rel_path(row_obj)
                        if not rel_path:
                            stats["rows_skipped_missing_path"] += 1
                            continue

                        header_path = data_dir / f"{rel_path}.hea"
                        if strict_files and not header_path.exists():
                            msg = f"missing header file: {header_path}"
                            if len(stats["first_errors"]) < 20:
                                stats["first_errors"].append(msg)
                            raise FileNotFoundError(msg)
                        if not header_path.exists():
                            stats["rows_skipped_missing_header_file"] += 1
                            continue

                        try:
                            fs, n_samples, n_leads = _read_header(data_dir, rel_path)
                        except Exception as exc:
                            stats["rows_skipped_header_error"] += 1
                            msg = f"header error path={rel_path}: {exc}"
                            if len(stats["first_errors"]) < 20:
                                stats["first_errors"].append(msg)
                            if strict_headers:
                                raise RuntimeError(msg) from exc
                            continue

                        record_id = _build_record_id(row_obj, rel_path)
                        if not record_id:
                            stats["rows_skipped_missing_record_id"] += 1
                            continue
                        if record_id in seen_record_ids:
                            stats["rows_skipped_duplicate_record_id"] += 1
                            continue
                        seen_record_ids.add(record_id)

                        out_buffer.append(
                            {
                                "record_id": record_id,
                                "subject_id": _as_text(row_obj.get("subject_id")),
                                "study_id": _as_text(row_obj.get("study_id")),
                                "file_name": _as_text(row_obj.get("file_name")),
                                "source": source,
                                "fs": int(fs),
                                "n_samples": int(n_samples),
                                "n_leads": int(n_leads),
                                "ecg_time": _as_time_text(row_obj.get("ecg_time")),
                                "path": rel_path,
                            }
                        )

                        if len(out_buffer) >= int(batch_size):
                            writer.write_rows(out_buffer)
                            out_buffer = []

                        if int(log_every) > 0 and stats["rows_selected_db"] % int(log_every) == 0:
                            logger.info(
                                "progress rows=%s written=%s skipped_missing_path=%s skipped_header=%s",
                                stats["rows_selected_db"],
                                writer.rows_written,
                                stats["rows_skipped_missing_path"],
                                stats["rows_skipped_header_error"] + stats["rows_skipped_missing_header_file"],
                            )

            writer.write_rows(out_buffer)
        finally:
            writer.close()

    if writer.rows_written < 1:
        raise RuntimeError("no manifest rows written; check table data-dir/path/header files")

    elapsed = float(time.perf_counter() - t0)
    stats["rows_written"] = int(writer.rows_written)
    stats["duration_sec"] = round(elapsed, 3)
    stats["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    schema = _validate_identifier(args.schema, name="schema")
    table_name = _validate_identifier(args.record_list_table, name="record_list_table")
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output_path).resolve()
    summary_path = (
        Path(args.summary_path).resolve()
        if str(args.summary_path).strip()
        else output_path.with_suffix(".summary.json")
    )

    try:
        summary = build_manifest(
            database_url=str(args.database_url),
            schema=schema,
            table_name=table_name,
            data_dir=data_dir,
            output_path=output_path,
            summary_path=summary_path,
            where_sql=str(args.where_sql),
            limit=int(args.limit),
            batch_size=int(args.batch_size),
            log_every=int(args.log_every),
            source=str(args.source).strip() or "mimic_ecg",
            strict_files=bool(args.strict_files),
            strict_headers=bool(args.strict_headers),
        )
    except Exception as exc:
        logging.getLogger("build_manifest").error("failed: %s", exc)
        return 1

    logging.getLogger("build_manifest").info(
        "done rows_written=%s output=%s summary=%s",
        summary["rows_written"],
        output_path,
        summary_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



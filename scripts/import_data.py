"""Import MIMIC-IV clinical + ECG index data into Postgres.

Supports demo and full datasets via chunked CSV loading.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLINICAL_ROOT = PROJECT_ROOT / "data" / "mimic-iv-clinical-database-demo-2.2"
DEFAULT_ECG_ROOT = (
    PROJECT_ROOT / "data" / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
)
DEFAULT_DB_URL = os.getenv(
    "IMPORT_DATABASE_URL",
    "postgresql+psycopg://ecg:ecg@localhost:5432/ecg",
)
DEFAULT_SCHEMA = os.getenv("DATA_SCHEMA", "mimiciv")
DEFAULT_CHUNKSIZE = 200_000
NUMERIC_SQL_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "decimal",
    "numeric",
    "real",
    "double precision",
}

HOSP_TASKS: dict[str, str] = {
    "admissions.csv.gz": "admissions",
    "diagnoses_icd.csv.gz": "diagnoses_icd",
    "labevents.csv.gz": "labevents",
    "patients.csv.gz": "patients",
    "d_labitems.csv.gz": "d_labitems",
    "prescriptions.csv.gz": "prescriptions",
    "pharmacy.csv.gz": "pharmacy",
    "d_icd_diagnoses.csv.gz": "d_icd_diagnoses",
}
ICU_TASKS: dict[str, str] = {
    "icustays.csv.gz": "icustays",
    "chartevents.csv.gz": "chartevents",
    "d_items.csv.gz": "d_items",
}

INDEX_COLUMNS: dict[str, list[str]] = {
    "admissions": ["subject_id", "hadm_id", "admittime", "dischtime"],
    "diagnoses_icd": ["subject_id", "hadm_id", "icd_code", "icd_version"],
    "labevents": ["subject_id", "hadm_id", "itemid", "charttime"],
    "prescriptions": ["subject_id", "hadm_id", "drug", "starttime", "stoptime"],
    "pharmacy": ["subject_id", "hadm_id", "medication", "starttime", "stoptime"],
    "record_list": ["subject_id", "study_id", "ecg_time"],
    "icustays": ["subject_id", "hadm_id", "intime", "outtime"],
    "chartevents": ["subject_id", "hadm_id", "itemid", "charttime"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import MIMIC-IV clinical + ECG index data into Postgres (demo or full)"
    )
    parser.add_argument("--database-url", default=DEFAULT_DB_URL)
    parser.add_argument("--clinical-root", default=str(DEFAULT_CLINICAL_ROOT))
    parser.add_argument("--ecg-root", default=str(DEFAULT_ECG_ROOT))
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--include-icu", action="store_true")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help="CSV chunk size. Use <=0 to load full file in-memory.",
    )
    parser.add_argument(
        "--if-exists",
        choices=["replace", "append", "fail"],
        default="replace",
        help="Behavior when destination table exists.",
    )
    parser.add_argument(
        "--tables",
        default="",
        help=(
            "Comma-separated table names to import. "
            "Default empty means import all enabled tables."
        ),
    )
    parser.add_argument("--skip-clinical", action="store_true")
    parser.add_argument("--skip-ecg-index", action="store_true")
    parser.add_argument("--create-indexes", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    return parser.parse_args()


def _validate_identifier(value: str, *, name: str) -> str:
    txt = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", txt):
        raise ValueError(f"invalid {name}: {value}")
    return txt


def _ensure_schema(engine: Engine, schema: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))


def _parse_selected_tables(raw: str) -> set[str] | None:
    items = [str(x).strip() for x in str(raw or "").split(",") if str(x).strip()]
    if not items:
        return None
    return {x.lower() for x in items}


def _should_import(*, table_name: str, selected: set[str] | None) -> bool:
    if selected is None:
        return True
    return str(table_name).lower() in selected


def _iter_csv_chunks(csv_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    compression = "gzip" if csv_path.suffix.lower() == ".gz" else None
    if chunksize and chunksize > 0:
        for chunk in pd.read_csv(
            csv_path,
            compression=compression,
            low_memory=False,
            chunksize=int(chunksize),
        ):
            yield chunk
    else:
        yield pd.read_csv(csv_path, compression=compression, low_memory=False)


def _get_table_column_types(
    engine: Engine,
    *,
    schema: str,
    table_name: str,
) -> dict[str, str]:
    q = text(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table_name
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"schema": schema, "table_name": table_name}).fetchall()
    return {
        str(r[0]): str(r[1]).strip().lower()
        for r in rows
        if str(r[0]).strip()
    }


def _coerce_chunk_to_table_types(
    chunk: pd.DataFrame,
    *,
    table_types: dict[str, str],
) -> pd.DataFrame:
    if not table_types:
        return chunk

    out = chunk.copy()
    for col, sql_type in table_types.items():
        if col not in out.columns:
            continue
        if sql_type in NUMERIC_SQL_TYPES and not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _import_table(
    *,
    engine: Engine,
    schema: str,
    csv_path: Path,
    table_name: str,
    chunksize: int,
    if_exists_mode: str,
) -> int:
    if not csv_path.exists():
        print(f"[skip] missing file: {csv_path}")
        return 0

    print(f"[import] {csv_path.name} -> {schema}.{table_name}")
    total_rows = 0
    first_chunk = True

    table_types: dict[str, str] = {}
    if if_exists_mode == "append":
        table_types = _get_table_column_types(engine, schema=schema, table_name=table_name)
        if table_types:
            print(f"  [coerce] detected existing schema for {schema}.{table_name}")

    for chunk_idx, chunk in enumerate(_iter_csv_chunks(csv_path, chunksize), start=1):
        if table_types:
            chunk = _coerce_chunk_to_table_types(chunk, table_types=table_types)

        if first_chunk:
            if_exists = if_exists_mode
            if if_exists_mode == "replace":
                if_exists = "replace"
            elif if_exists_mode == "append":
                if_exists = "append"
            elif if_exists_mode == "fail":
                if_exists = "fail"
            else:
                raise ValueError(f"unsupported if_exists mode: {if_exists_mode}")
        else:
            if_exists = "append"

        chunk.to_sql(table_name, engine, if_exists=if_exists, index=False, schema=schema)
        row_n = int(len(chunk))
        total_rows += row_n

        if first_chunk:
            # Refresh to the actual DB schema created/used by first write.
            table_types = _get_table_column_types(engine, schema=schema, table_name=table_name)
            first_chunk = False

        print(f"  [chunk {chunk_idx}] rows={row_n} total={total_rows}")

    print(f"[ok] rows={total_rows} table={schema}.{table_name}")
    return total_rows


def _column_exists(engine: Engine, *, schema: str, table_name: str, column_name: str) -> bool:
    q = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table_name
          AND column_name = :column_name
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            q,
            {"schema": schema, "table_name": table_name, "column_name": column_name},
        ).first()
    return row is not None


def _make_index_name(schema: str, table_name: str, col: str) -> str:
    base = f"idx_{schema}_{table_name}_{col}"
    if len(base) <= 60:
        return base
    return f"idx_{table_name}_{col}"[:60]


def _create_indexes(engine: Engine, *, schema: str, tables: set[str]) -> None:
    print("\n>>> Creating indexes")
    with engine.begin() as conn:
        for table_name, cols in INDEX_COLUMNS.items():
            if table_name not in tables:
                continue
            for col in cols:
                if not _column_exists(
                    engine,
                    schema=schema,
                    table_name=table_name,
                    column_name=col,
                ):
                    print(f"  [skip] {schema}.{table_name}.{col} column not found")
                    continue
                idx_name = _make_index_name(schema, table_name, col)
                sql = (
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {schema}.{table_name} ({col})"
                )
                conn.execute(text(sql))
                print(f"  [ok] {idx_name}")


def _analyze_tables(engine: Engine, *, schema: str, tables: set[str]) -> None:
    print("\n>>> ANALYZE imported tables")
    with engine.begin() as conn:
        for table_name in sorted(tables):
            sql = f"ANALYZE {schema}.{table_name}"
            conn.execute(text(sql))
            print(f"  [ok] ANALYZE {schema}.{table_name}")


def import_mimic_data(
    *,
    engine: Engine,
    schema: str,
    clinical_root: Path,
    include_icu: bool,
    chunksize: int,
    if_exists_mode: str,
    selected_tables: set[str] | None,
) -> set[str]:
    imported: set[str] = set()
    print("\n>>> Import MIMIC-IV clinical data")

    hosp_dir = clinical_root / "hosp"
    for file_name, table_name in HOSP_TASKS.items():
        if not _should_import(table_name=table_name, selected=selected_tables):
            print(f"[skip] filtered table: {table_name}")
            continue
        _import_table(
            engine=engine,
            schema=schema,
            csv_path=hosp_dir / file_name,
            table_name=table_name,
            chunksize=chunksize,
            if_exists_mode=if_exists_mode,
        )
        imported.add(table_name)

    if include_icu:
        icu_dir = clinical_root / "icu"
        for file_name, table_name in ICU_TASKS.items():
            if not _should_import(table_name=table_name, selected=selected_tables):
                print(f"[skip] filtered table: {table_name}")
                continue
            _import_table(
                engine=engine,
                schema=schema,
                csv_path=icu_dir / file_name,
                table_name=table_name,
                chunksize=chunksize,
                if_exists_mode=if_exists_mode,
            )
            imported.add(table_name)

    return imported


def import_ecg_index(
    *,
    engine: Engine,
    schema: str,
    ecg_root: Path,
    chunksize: int,
    if_exists_mode: str,
    selected_tables: set[str] | None,
) -> set[str]:
    imported: set[str] = set()
    table_name = "record_list"
    if not _should_import(table_name=table_name, selected=selected_tables):
        print(f"[skip] filtered table: {table_name}")
        return imported

    print("\n>>> Import ECG record_list index")
    record_list_csv = ecg_root / "record_list.csv"
    _import_table(
        engine=engine,
        schema=schema,
        csv_path=record_list_csv,
        table_name=table_name,
        chunksize=chunksize,
        if_exists_mode=if_exists_mode,
    )
    imported.add(table_name)
    return imported


def main() -> int:
    args = _parse_args()
    schema = _validate_identifier(args.schema, name="schema")
    clinical_root = Path(args.clinical_root).resolve()
    ecg_root = Path(args.ecg_root).resolve()

    if not args.skip_clinical and not clinical_root.exists():
        raise FileNotFoundError(f"clinical-root not found: {clinical_root}")
    if not args.skip_ecg_index and not ecg_root.exists():
        raise FileNotFoundError(f"ecg-root not found: {ecg_root}")

    engine = create_engine(str(args.database_url))
    _ensure_schema(engine, schema)

    selected_tables = _parse_selected_tables(args.tables)
    imported_tables: set[str] = set()

    if not args.skip_clinical:
        imported_tables.update(
            import_mimic_data(
                engine=engine,
                schema=schema,
                clinical_root=clinical_root,
                include_icu=bool(args.include_icu),
                chunksize=int(args.chunksize),
                if_exists_mode=str(args.if_exists),
                selected_tables=selected_tables,
            )
        )

    if not args.skip_ecg_index:
        imported_tables.update(
            import_ecg_index(
                engine=engine,
                schema=schema,
                ecg_root=ecg_root,
                chunksize=int(args.chunksize),
                if_exists_mode=str(args.if_exists),
                selected_tables=selected_tables,
            )
        )

    if args.create_indexes and imported_tables:
        _create_indexes(engine, schema=schema, tables=imported_tables)

    if args.analyze and imported_tables:
        _analyze_tables(engine, schema=schema, tables=imported_tables)

    print("\nDone.")
    print(f"Imported tables: {sorted(imported_tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

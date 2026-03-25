"""
Validate cohort artifacts for eval runs.

Checks per gold_id/run_id:
- cohort.parquet exists and has expected schema columns
- distinct subject_id count is within expected [min, max]
- required fields subject_id/index_time are present and non-missing
- optional admission-window check for index_time (if DB connectivity is available)
- key-variable missing-rate thresholds (template defaults or gold overrides)

Output:
- eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger("eval_check_cohort")

DEFAULT_KEY_MISSING_RATE_MAX: dict[str, dict[str, float]] = {
    "electrolyte_hyperkalemia": {
        "subject_id": 0.0,
        "index_time": 0.0,
        "cohort_label": 0.0,
        # labevents has some null hadm_id rows in MIMIC; allow moderate missingness.
        "hadm_id": 0.35,
    },
    "diagnosis_icd": {
        "subject_id": 0.0,
        "index_time": 0.0,
        "cohort_label": 0.0,
        "hadm_id": 0.01,
    },
    "medication_exposure": {
        "subject_id": 0.0,
        "index_time": 0.0,
        "cohort_label": 0.0,
        "hadm_id": 0.01,
    },
}


@dataclass(frozen=True)
class TimeWindowConfig:
    enabled: bool
    before_hours: float
    after_hours: float
    min_pass_rate: float
    min_rows: int


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    return out


def _read_gold(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"gold yaml must be a list: {path}")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"gold entry #{i} is not an object")
        out.append(item)
    return out


def _load_run_id(eval_runs_root: Path, gold_id: str) -> str | None:
    p = eval_runs_root / gold_id / "run_id.txt"
    if not p.exists():
        return None
    run_id = p.read_text(encoding="utf-8").strip()
    return run_id or None


def _normalize_time_window_cfg(expectations: dict[str, Any], args: argparse.Namespace) -> TimeWindowConfig:
    cohort_cfg = expectations.get("cohort") if isinstance(expectations.get("cohort"), dict) else {}
    time_cfg = cohort_cfg.get("time_window") if isinstance(cohort_cfg.get("time_window"), dict) else {}

    enabled = bool(time_cfg.get("enabled", True))
    before_hours = _safe_float(time_cfg.get("before_hours"), default=float(args.time_window_before_hours))
    after_hours = _safe_float(time_cfg.get("after_hours"), default=float(args.time_window_after_hours))
    min_pass_rate = _safe_float(time_cfg.get("min_pass_rate"), default=float(args.time_window_min_pass_rate))
    min_rows = int(time_cfg.get("min_rows", args.time_window_min_rows))

    return TimeWindowConfig(
        enabled=enabled,
        before_hours=max(0.0, before_hours),
        after_hours=max(0.0, after_hours),
        min_pass_rate=min(1.0, max(0.0, min_pass_rate)),
        min_rows=max(0, min_rows),
    )


def _missing_rate_thresholds(template_name: str, expectations: dict[str, Any]) -> dict[str, float]:
    cohort_cfg = expectations.get("cohort") if isinstance(expectations.get("cohort"), dict) else {}
    override = cohort_cfg.get("key_missing_rate_max")
    if not isinstance(override, dict):
        override = expectations.get("cohort_key_missing_rate_max")
    if isinstance(override, dict):
        out: dict[str, float] = {}
        for k, v in override.items():
            out[str(k)] = min(1.0, max(0.0, _safe_float(v, default=1.0)))
        return out
    return dict(DEFAULT_KEY_MISSING_RATE_MAX.get(template_name, {"subject_id": 0.0, "index_time": 0.0}))


def _import_psycopg() -> Any | None:
    try:
        import psycopg  # type: ignore

        return psycopg
    except Exception:
        return None


def _query_admissions_map(
    *,
    hadm_ids: list[int],
    db_url: str,
    schema: str,
    psycopg_mod: Any,
) -> pd.DataFrame:
    if not hadm_ids:
        return pd.DataFrame(columns=["hadm_id", "admittime", "dischtime"])

    safe_schema = schema.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", safe_schema):
        raise ValueError(f"invalid schema name: {schema}")

    sql = (
        f"SELECT a.hadm_id, a.admittime, a.dischtime "
        f"FROM {safe_schema}.admissions a "
        f"WHERE a.hadm_id = ANY(%s)"
    )

    rows: list[tuple[Any, Any, Any]] = []
    chunk_size = 2000
    with psycopg_mod.connect(db_url) as conn:
        with conn.cursor() as cur:
            for i in range(0, len(hadm_ids), chunk_size):
                chunk = hadm_ids[i : i + chunk_size]
                cur.execute(sql, (chunk,))
                rows.extend(cur.fetchall())

    if not rows:
        return pd.DataFrame(columns=["hadm_id", "admittime", "dischtime"])

    df = pd.DataFrame(rows, columns=["hadm_id", "admittime", "dischtime"])
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce", utc=True)
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce", utc=True)
    df = df.dropna(subset=["hadm_id", "admittime"]).drop_duplicates(subset=["hadm_id"], keep="first")
    return df


def _check_one(
    *,
    gold: dict[str, Any],
    run_id: str | None,
    artifacts_root: Path,
    db_url: str,
    data_schema: str,
    args: argparse.Namespace,
    psycopg_mod: Any | None,
) -> dict[str, Any]:
    gold_id = str(gold.get("id", "")).strip() or "UNKNOWN"
    name = str(gold.get("name", "")).strip()
    template_name = str(gold.get("cohort_template", "")).strip()
    expectations = gold.get("expectations") if isinstance(gold.get("expectations"), dict) else {}

    result: dict[str, Any] = {
        "gold_id": gold_id,
        "name": name,
        "template_name": template_name,
        "run_id": run_id,
        "pass": True,
        "reasons": [],
        "checks": {},
        "metrics": {},
    }

    def fail(msg: str) -> None:
        result["pass"] = False
        result["reasons"].append(msg)

    if not run_id:
        fail("missing run_id.txt")
        result["checks"]["cohort_file_exists"] = False
        return result

    run_dir = artifacts_root / run_id
    cohort_path = run_dir / "cohort.parquet"
    result["cohort_path"] = str(cohort_path)
    if not cohort_path.exists():
        result["checks"]["cohort_file_exists"] = False
        fail("cohort.parquet not found")
        return result
    result["checks"]["cohort_file_exists"] = True

    try:
        cohort_df = pd.read_parquet(cohort_path)
    except Exception as exc:
        fail(f"failed to read cohort.parquet: {exc}")
        return result

    row_count = int(len(cohort_df))
    result["metrics"]["row_count"] = row_count

    expected_cols = ["subject_id", "hadm_id", "index_time", "cohort_label"]
    missing_cols = [c for c in expected_cols if c not in cohort_df.columns]
    result["checks"]["schema_missing_columns"] = missing_cols
    if missing_cols:
        fail(f"schema missing columns: {missing_cols}")

    for c in ["subject_id", "index_time"]:
        if c not in cohort_df.columns:
            fail(f"required field missing: {c}")

    if "subject_id" in cohort_df.columns:
        sid = pd.to_numeric(cohort_df["subject_id"], errors="coerce")
        subject_missing_rate = float(sid.isna().mean()) if len(cohort_df) else 1.0
        distinct_subjects = int(sid.dropna().astype("int64").nunique()) if len(cohort_df) else 0
        result["metrics"]["distinct_subject_id"] = distinct_subjects
        result["metrics"]["subject_id_missing_rate"] = subject_missing_rate
        if subject_missing_rate > 0:
            fail(f"subject_id has missing/non-numeric rows: rate={subject_missing_rate:.4f}")

        min_subj = expectations.get("cohort_subjects_min")
        max_subj = expectations.get("cohort_subjects_max")
        result["checks"]["subject_range"] = {
            "actual": distinct_subjects,
            "min": min_subj,
            "max": max_subj,
        }
        if isinstance(min_subj, int) and distinct_subjects < min_subj:
            fail(f"cohort out of range: distinct_subject_id={distinct_subjects} < min={min_subj}")
        if isinstance(max_subj, int) and distinct_subjects > max_subj:
            fail(f"cohort out of range: distinct_subject_id={distinct_subjects} > max={max_subj}")

    index_ts = pd.Series(dtype="datetime64[ns, UTC]")
    if "index_time" in cohort_df.columns:
        index_ts = pd.to_datetime(cohort_df["index_time"], errors="coerce", utc=True)
        index_missing_rate = float(index_ts.isna().mean()) if len(cohort_df) else 1.0
        result["metrics"]["index_time_missing_rate"] = index_missing_rate
        if index_missing_rate > 0:
            fail(f"index_time has missing/unparseable rows: rate={index_missing_rate:.4f}")

    missing_limits = _missing_rate_thresholds(template_name, expectations)
    missing_report: dict[str, Any] = {}
    for col, limit in missing_limits.items():
        if col == "index_time" and "index_time" in cohort_df.columns:
            miss_rate = float(index_ts.isna().mean()) if len(cohort_df) else 1.0
        elif col in cohort_df.columns:
            miss_rate = float(cohort_df[col].isna().mean()) if len(cohort_df) else 1.0
        else:
            miss_rate = 1.0
        missing_report[col] = {
            "missing_rate": miss_rate,
            "max": float(limit),
        }
        if miss_rate > float(limit):
            fail(
                f"missing rate exceeds threshold: {col}={miss_rate:.4f} > {float(limit):.4f}"
            )
    result["checks"]["missing_rate_thresholds"] = missing_report

    tw_cfg = _normalize_time_window_cfg(expectations, args)
    time_window_result: dict[str, Any] = {
        "enabled": tw_cfg.enabled,
        "status": "skipped",
    }

    if tw_cfg.enabled:
        if "hadm_id" not in cohort_df.columns:
            time_window_result["reason"] = "hadm_id column missing"
        elif psycopg_mod is None:
            time_window_result["reason"] = "psycopg unavailable"
        else:
            hadm = pd.to_numeric(cohort_df["hadm_id"], errors="coerce").astype("Int64")
            with_hadm = cohort_df.copy()
            with_hadm["hadm_id"] = hadm
            with_hadm["index_time_parsed"] = index_ts
            hadm_ids = [int(v) for v in with_hadm["hadm_id"].dropna().drop_duplicates().tolist()]

            if not hadm_ids:
                time_window_result["reason"] = "no non-null hadm_id rows"
            else:
                try:
                    adm = _query_admissions_map(
                        hadm_ids=hadm_ids,
                        db_url=db_url,
                        schema=data_schema,
                        psycopg_mod=psycopg_mod,
                    )
                except Exception as exc:
                    time_window_result["reason"] = f"admissions query failed: {exc}"
                else:
                    merged = with_hadm.merge(adm, on="hadm_id", how="left")
                    checked = merged[
                        merged["hadm_id"].notna()
                        & merged["index_time_parsed"].notna()
                        & merged["admittime"].notna()
                    ].copy()

                    checked_n = int(len(checked))
                    matched_n = int(merged["admittime"].notna().sum())
                    time_window_result["checked_rows"] = checked_n
                    time_window_result["hadm_rows"] = int(merged["hadm_id"].notna().sum())
                    time_window_result["matched_admission_rows"] = matched_n

                    if checked_n < tw_cfg.min_rows:
                        time_window_result["reason"] = (
                            f"not enough rows for time-window check: {checked_n} < {tw_cfg.min_rows}"
                        )
                    else:
                        before = pd.to_timedelta(tw_cfg.before_hours, unit="h")
                        after = pd.to_timedelta(tw_cfg.after_hours, unit="h")
                        fallback_dischtime = checked["admittime"] + pd.to_timedelta(14, unit="D")
                        effective_dischtime = checked["dischtime"].fillna(fallback_dischtime)

                        lower = checked["admittime"] - before
                        upper = effective_dischtime + after
                        ok = (checked["index_time_parsed"] >= lower) & (checked["index_time_parsed"] <= upper)
                        pass_rate = float(ok.mean()) if checked_n else float("nan")
                        time_window_result.update(
                            {
                                "status": "passed" if pass_rate >= tw_cfg.min_pass_rate else "failed",
                                "pass_rate": pass_rate,
                                "min_pass_rate": tw_cfg.min_pass_rate,
                                "before_hours": tw_cfg.before_hours,
                                "after_hours": tw_cfg.after_hours,
                            }
                        )
                        if pass_rate < tw_cfg.min_pass_rate:
                            fail(
                                "index_time not near admission window: "
                                f"pass_rate={pass_rate:.4f} < {tw_cfg.min_pass_rate:.4f}"
                            )

    result["checks"]["time_window"] = time_window_result
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check cohort artifacts for eval runs")
    parser.add_argument("--gold", default="evals/gold_questions.yaml")
    parser.add_argument("--eval-runs-root", default="eval_runs")
    parser.add_argument("--artifacts-root", default="storage/artifacts")
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "postgresql://ecg:ecg@localhost:5432/ecg"))
    parser.add_argument("--data-schema", default=os.getenv("DATA_SCHEMA", "mimiciv"))
    parser.add_argument("--time-window-before-hours", type=float, default=24.0)
    parser.add_argument("--time-window-after-hours", type=float, default=24.0)
    parser.add_argument("--time-window-min-pass-rate", type=float, default=0.95)
    parser.add_argument("--time-window-min-rows", type=int, default=10)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    gold_path = Path(args.gold).resolve()
    eval_runs_root = Path(args.eval_runs_root).resolve()
    artifacts_root = Path(args.artifacts_root).resolve()
    output_path = Path(args.output).resolve()

    gold_items = _read_gold(gold_path)
    psycopg_mod = _import_psycopg()
    if psycopg_mod is None:
        LOGGER.warning("psycopg unavailable; time-window checks may be skipped")

    results: list[dict[str, Any]] = []
    for gold in gold_items:
        gold_id = str(gold.get("id", "")).strip() or "UNKNOWN"
        run_id = _load_run_id(eval_runs_root, gold_id)
        res = _check_one(
            gold=gold,
            run_id=run_id,
            artifacts_root=artifacts_root,
            db_url=str(args.db_url),
            data_schema=str(args.data_schema),
            args=args,
            psycopg_mod=psycopg_mod,
        )
        results.append(res)

    passed = sum(1 for r in results if r.get("pass") is True)
    failed = len(results) - passed
    out = {
        "generated_at": _utc_now_iso(),
        "gold_count": len(gold_items),
        "checked_count": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    output_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    LOGGER.info("check_cohort done checked=%d passed=%d failed=%d output=%s", len(results), passed, failed, output_path)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


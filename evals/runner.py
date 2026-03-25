"""
Run evaluation gold questions end-to-end.

Per gold item:
1) create run
2) build cohort
3) extract ECG features (queue + worker drain)
4) prepare analysis inputs
5) generate report

Outputs:
- eval_runs/<gold_id>/run_id.txt
- eval_runs/<gold_id>/result.json
- eval_runs/summary_<mode>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "services" / "api"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.main import app  # noqa: E402
from pipelines.assemble_analysis_dataset import assemble_analysis_dataset  # noqa: E402
from pipelines.build_analysis_tables import build_analysis_tables  # noqa: E402
from pipelines.build_report_plots import build_report_plots  # noqa: E402

LOGGER = logging.getLogger("eval_runner")


@dataclass(frozen=True)
class ModeConfig:
    max_questions: int
    max_records_per_run: int


def _read_yaml_list(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"gold yaml must be a list: {path}")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"gold item #{i} is not an object")
        out.append(item)
    return out


def _select_mode_config(args: argparse.Namespace, total: int) -> ModeConfig:
    if args.mode == "smoke":
        max_questions = min(total, max(1, int(args.smoke_n)))
        max_records = max(1, int(args.smoke_max_records))
        return ModeConfig(max_questions=max_questions, max_records_per_run=max_records)

    max_questions = total
    if int(args.full_n) > 0:
        max_questions = min(total, int(args.full_n))
    max_records = max(0, int(args.full_max_records))
    return ModeConfig(max_questions=max_questions, max_records_per_run=max_records)


def _post_json(client: TestClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = client.post(path, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"POST {path} failed [{resp.status_code}] {resp.text}")
    body = resp.json()
    if not isinstance(body, dict):
        raise RuntimeError(f"POST {path} unexpected response type")
    return body


def _load_record_ids_from_cohort(
    *,
    cohort_path: Path,
    manifest_path: Path,
    max_records: int,
) -> list[str]:
    if not cohort_path.exists():
        raise FileNotFoundError(f"cohort not found: {cohort_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    cohort_df = pd.read_parquet(cohort_path)
    if "subject_id" not in cohort_df.columns:
        raise RuntimeError("cohort.parquet missing subject_id")

    manifest_df = pd.read_parquet(manifest_path, columns=["record_id", "subject_id"])
    cohort_subjects = cohort_df[["subject_id"]].copy()
    cohort_subjects["subject_id"] = cohort_subjects["subject_id"].astype("string").str.strip()
    cohort_subjects = cohort_subjects[cohort_subjects["subject_id"].notna()]
    cohort_subjects["_ord"] = np.arange(len(cohort_subjects))

    manifest_df["subject_id"] = manifest_df["subject_id"].astype("string").str.strip()
    manifest_df["record_id"] = manifest_df["record_id"].astype("string").str.strip()
    manifest_df = manifest_df[
        manifest_df["subject_id"].notna() & manifest_df["record_id"].notna()
    ].copy()

    joined = cohort_subjects.merge(manifest_df, on="subject_id", how="left")
    joined = joined[joined["record_id"].notna()].copy()
    joined = joined.sort_values(["_ord", "record_id"]).drop_duplicates(subset=["record_id"], keep="first")

    record_ids = [str(v).strip() for v in joined["record_id"].tolist() if str(v).strip()]
    if max_records > 0:
        record_ids = record_ids[:max_records]
    return record_ids


def _run_worker_once(*, artifacts_root: Path, data_dir: Path, manifest_path: Path) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "ecg_features_worker.py"),
        "--once",
        "--artifacts-root",
        str(artifacts_root),
        "--data-dir",
        str(data_dir),
        "--global-manifest",
        str(manifest_path),
        "--log-level",
        "WARNING",
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "ecg worker failed with non-zero exit code: "
            f"{proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def _wait_for_file(path: Path, timeout_sec: float, poll_sec: float = 0.5) -> bool:
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) <= timeout_sec:
        if path.exists() and path.is_file():
            return True
        time.sleep(max(0.1, poll_sec))
    return False


def _prepare_report_inputs(*, run_id: str, artifacts_root: Path, manifest_path: Path) -> dict[str, str]:
    run_dir = artifacts_root / run_id
    cohort_path = run_dir / "cohort.parquet"
    features_path = run_dir / "ecg_features.parquet"
    ecg_map_path = run_dir / "ecg_map.parquet"

    analysis_dataset_path, _analysis_summary = assemble_analysis_dataset(
        run_id=run_id,
        artifacts_root=artifacts_root,
        cohort_path=cohort_path,
        features_path=features_path,
        ecg_map_path=ecg_map_path,
        global_manifest_path=manifest_path,
        covariates_path=None,
        age_bin_mode="fixed",
    )

    feature_summary_path, group_compare_path, _tables_summary = build_analysis_tables(
        run_id=run_id,
        artifacts_root=artifacts_root,
        analysis_dataset_path=analysis_dataset_path,
        group_cols=["cohort_label", "sex", "age_bin"],
        compare_by="cohort_label",
        compare_features=["mean_hr", "rr_std", "rr_mean"],
    )

    _plots_summary = build_report_plots(
        run_id=run_id,
        artifacts_root=artifacts_root,
        analysis_dataset_path=analysis_dataset_path,
        qc_path=run_dir / "ecg_qc.parquet",
        fail_top_n=8,
        feature_preferred=["rr_std", "qtc", "qtc_ms"],
    )

    return {
        "analysis_dataset": str(analysis_dataset_path),
        "feature_summary": str(feature_summary_path),
        "group_compare": str(group_compare_path),
        "ecg_map": str(ecg_map_path),
    }


def _evaluate_expectations(gold: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    exp = gold.get("expectations") if isinstance(gold.get("expectations"), dict) else {}
    checks: dict[str, Any] = {"passed": True}

    cohort_path = run_dir / "cohort.parquet"
    qc_path = run_dir / "ecg_qc.parquet"
    features_path = run_dir / "ecg_features.parquet"
    report_path = run_dir / "report.md"

    required = exp.get("required_artifacts") if isinstance(exp.get("required_artifacts"), list) else []
    missing_required = [name for name in required if not (run_dir / str(name)).exists()]
    checks["missing_required_artifacts"] = missing_required
    if missing_required:
        checks["passed"] = False

    if cohort_path.exists():
        cohort_df = pd.read_parquet(cohort_path)
        subjects = int(cohort_df["subject_id"].nunique()) if "subject_id" in cohort_df.columns else 0
    else:
        subjects = 0
    checks["cohort_subjects"] = subjects

    min_subj = exp.get("cohort_subjects_min")
    max_subj = exp.get("cohort_subjects_max")
    if isinstance(min_subj, int) and subjects < min_subj:
        checks["passed"] = False
        checks["cohort_subjects_min_violation"] = {"actual": subjects, "min": min_subj}
    if isinstance(max_subj, int) and subjects > max_subj:
        checks["passed"] = False
        checks["cohort_subjects_max_violation"] = {"actual": subjects, "max": max_subj}

    qc_cfg = exp.get("qc") if isinstance(exp.get("qc"), dict) else {}
    pass_rate_min = qc_cfg.get("pass_rate_min")
    if qc_path.exists() and "pass_rate_min" in qc_cfg:
        qc_df = pd.read_parquet(qc_path)
        if len(qc_df) == 0 or "qc_pass" not in qc_df.columns:
            pass_rate = 0.0
        else:
            pass_rate = float((qc_df["qc_pass"] == True).mean())  # noqa: E712
        checks["qc_pass_rate"] = pass_rate
        if isinstance(pass_rate_min, (int, float)) and pass_rate < float(pass_rate_min):
            checks["passed"] = False
            checks["qc_pass_rate_violation"] = {
                "actual": pass_rate,
                "min": float(pass_rate_min),
            }

    feat_cfg = exp.get("features") if isinstance(exp.get("features"), dict) else {}
    hr_min = feat_cfg.get("mean_hr_min")
    hr_max = feat_cfg.get("mean_hr_max")
    if features_path.exists():
        feat_df = pd.read_parquet(features_path)
        if "mean_hr" in feat_df.columns and len(feat_df) > 0:
            mean_hr = float(pd.to_numeric(feat_df["mean_hr"], errors="coerce").mean())
        else:
            mean_hr = float("nan")
        checks["features_mean_hr"] = mean_hr
        if isinstance(hr_min, (int, float)) and np.isfinite(mean_hr) and mean_hr < float(hr_min):
            checks["passed"] = False
            checks["mean_hr_min_violation"] = {"actual": mean_hr, "min": float(hr_min)}
        if isinstance(hr_max, (int, float)) and np.isfinite(mean_hr) and mean_hr > float(hr_max):
            checks["passed"] = False
            checks["mean_hr_max_violation"] = {"actual": mean_hr, "max": float(hr_max)}

    report_cfg = exp.get("report") if isinstance(exp.get("report"), dict) else {}
    must_sections = report_cfg.get("must_have_sections") if isinstance(report_cfg.get("must_have_sections"), list) else []
    must_mention = report_cfg.get("must_mention") if isinstance(report_cfg.get("must_mention"), list) else []
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8", errors="ignore")
        lower_text = text.lower()
        missing_sections = [s for s in must_sections if str(s).lower() not in lower_text]
        missing_mentions = [m for m in must_mention if str(m).lower() not in lower_text]
        checks["report_missing_sections"] = missing_sections
        checks["report_missing_mentions"] = missing_mentions
        if missing_sections or missing_mentions:
            checks["passed"] = False

    return checks


def _run_one_gold(
    *,
    client: TestClient,
    gold: dict[str, Any],
    artifacts_root: Path,
    eval_runs_root: Path,
    manifest_path: Path,
    data_dir: Path,
    max_records_per_run: int,
    wait_timeout_sec: float,
) -> dict[str, Any]:
    gold_id = str(gold.get("id", "")).strip() or "UNKNOWN"
    name = str(gold.get("name", "")).strip() or gold_id
    template_name = str(gold.get("cohort_template", "")).strip()
    params = gold.get("params") if isinstance(gold.get("params"), dict) else {}
    if not template_name:
        raise ValueError(f"gold {gold_id} missing cohort_template")

    question = f"[{gold_id}] {name}"
    run_create = _post_json(
        client,
        "/runs",
        {
            "question": question,
            "params": {
                "gold_id": gold_id,
                "gold_name": name,
                "template_name": template_name,
                "gold_params": params,
            },
        },
    )
    run_id = str(run_create["run_id"])

    record_dir = eval_runs_root / gold_id
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "run_id.txt").write_text(run_id + "\n", encoding="utf-8")

    build_res = _post_json(
        client,
        "/tools/build_cohort",
        {
            "template_name": template_name,
            "params": params,
            "run_id": run_id,
            "limit": 5000,
        },
    )

    run_dir = artifacts_root / run_id
    cohort_path = run_dir / "cohort.parquet"
    record_ids = _load_record_ids_from_cohort(
        cohort_path=cohort_path,
        manifest_path=manifest_path,
        max_records=max_records_per_run,
    )
    if not record_ids:
        raise RuntimeError(f"gold {gold_id} produced no record_ids after cohort mapping")

    extract_res = _post_json(
        client,
        "/tools/extract_ecg_features",
        {
            "run_id": run_id,
            "record_ids": record_ids,
            "params": {
                "limit": 0,
                "feature_version": "v1.0-eval",
            },
        },
    )

    feature_result_path = run_dir / "ecg_feature_task_result.json"
    if not _wait_for_file(feature_result_path, timeout_sec=2.0):
        _run_worker_once(
            artifacts_root=artifacts_root,
            data_dir=data_dir,
            manifest_path=manifest_path,
        )
    if not _wait_for_file(feature_result_path, timeout_sec=wait_timeout_sec):
        raise TimeoutError(f"timeout waiting ecg feature result: {feature_result_path}")

    feature_result = json.loads(feature_result_path.read_text(encoding="utf-8"))
    feature_status = str(feature_result.get("status", "")).upper()
    if feature_status != "SUCCEEDED":
        raise RuntimeError(f"ecg feature task failed status={feature_status} payload={feature_result}")

    report_inputs = _prepare_report_inputs(
        run_id=run_id,
        artifacts_root=artifacts_root,
        manifest_path=manifest_path,
    )

    generate_res = _post_json(
        client,
        "/tools/generate_report",
        {
            "run_id": run_id,
            "config": {
                "question": question,
                "params": {
                    "template_name": template_name,
                    "gold_id": gold_id,
                },
            },
        },
    )

    report_task_result_path = run_dir / "report_task_result.json"
    report_path = run_dir / "report.md"
    if not _wait_for_file(report_task_result_path, timeout_sec=wait_timeout_sec):
        raise TimeoutError(f"timeout waiting report task result: {report_task_result_path}")
    if not _wait_for_file(report_path, timeout_sec=2.0):
        raise TimeoutError(f"report not generated: {report_path}")

    report_task_result = json.loads(report_task_result_path.read_text(encoding="utf-8"))
    report_status = str(report_task_result.get("status", "")).upper()
    if report_status != "REPORT_SUCCEEDED":
        raise RuntimeError(f"report task failed status={report_status} payload={report_task_result}")

    checks = _evaluate_expectations(gold, run_dir)

    out = {
        "gold_id": gold_id,
        "name": name,
        "run_id": run_id,
        "template_name": template_name,
        "build_row_count": int(build_res.get("row_count", 0)),
        "record_count": len(record_ids),
        "extract_job_id": extract_res.get("job_id"),
        "report_job_id": generate_res.get("job_id"),
        "report_inputs": report_inputs,
        "checks": checks,
        "status": "PASSED" if checks.get("passed", False) else "FAILED",
    }

    (record_dir / "result.json").write_text(
        json.dumps(out, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gold evals end-to-end")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--gold", default=str(PROJECT_ROOT / "evals" / "gold_questions.yaml"))
    parser.add_argument("--artifacts-root", default=str(PROJECT_ROOT / "storage" / "artifacts"))
    parser.add_argument("--eval-runs-root", default=str(PROJECT_ROOT / "eval_runs"))
    parser.add_argument("--global-manifest", default=str(PROJECT_ROOT / "storage" / "ecg_manifest.parquet"))
    parser.add_argument(
        "--data-dir",
        default=str(
            PROJECT_ROOT
            / "data"
            / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
        ),
    )
    parser.add_argument("--smoke-n", type=int, default=3)
    parser.add_argument("--smoke-max-records", type=int, default=50)
    parser.add_argument("--full-n", type=int, default=0, help="0 means all")
    parser.add_argument("--full-max-records", type=int, default=0, help="0 means no cap")
    parser.add_argument("--wait-timeout-sec", type=float, default=300.0)
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
    artifacts_root = Path(args.artifacts_root).resolve()
    eval_runs_root = Path(args.eval_runs_root).resolve()
    manifest_path = Path(args.global_manifest).resolve()
    data_dir = Path(args.data_dir).resolve()

    if not gold_path.exists():
        LOGGER.error("gold file not found: %s", gold_path)
        return 2

    gold_items = _read_yaml_list(gold_path)
    if not gold_items:
        LOGGER.error("gold file is empty: %s", gold_path)
        return 2

    mode_cfg = _select_mode_config(args, total=len(gold_items))
    selected = gold_items[: mode_cfg.max_questions]
    eval_runs_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "eval_start mode=%s selected=%d/%d max_records_per_run=%d",
        args.mode,
        len(selected),
        len(gold_items),
        mode_cfg.max_records_per_run,
    )

    results: list[dict[str, Any]] = []
    with TestClient(app) as client:
        for idx, gold in enumerate(selected, start=1):
            gold_id = str(gold.get("id", f"ITEM{idx:03d}"))
            LOGGER.info("[%d/%d] running %s", idx, len(selected), gold_id)
            started = time.perf_counter()
            try:
                result = _run_one_gold(
                    client=client,
                    gold=gold,
                    artifacts_root=artifacts_root,
                    eval_runs_root=eval_runs_root,
                    manifest_path=manifest_path,
                    data_dir=data_dir,
                    max_records_per_run=mode_cfg.max_records_per_run,
                    wait_timeout_sec=float(args.wait_timeout_sec),
                )
                result["duration_sec"] = round(time.perf_counter() - started, 3)
                LOGGER.info(
                    "[%d/%d] done %s status=%s records=%s run_id=%s",
                    idx,
                    len(selected),
                    gold_id,
                    result["status"],
                    result["record_count"],
                    result["run_id"],
                )
            except Exception as exc:
                LOGGER.exception("[%d/%d] failed %s: %s", idx, len(selected), gold_id, exc)
                result = {
                    "gold_id": gold_id,
                    "status": "ERROR",
                    "error": str(exc),
                    "duration_sec": round(time.perf_counter() - started, 3),
                }
                record_dir = eval_runs_root / gold_id
                record_dir.mkdir(parents=True, exist_ok=True)
                (record_dir / "result.json").write_text(
                    json.dumps(result, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )
            results.append(result)

    passed = sum(1 for r in results if r.get("status") == "PASSED")
    failed = sum(1 for r in results if r.get("status") in {"FAILED", "ERROR"})
    summary = {
        "mode": args.mode,
        "gold_total": len(gold_items),
        "gold_selected": len(selected),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    summary_path = eval_runs_root / f"summary_{args.mode}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    LOGGER.info("eval_done mode=%s passed=%d failed=%d summary=%s", args.mode, passed, failed, summary_path)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


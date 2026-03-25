"""
Background worker for ECG QC + feature extraction jobs.

Workflow:
1) Claim one queued job from artifacts/_queue/ecg_features/pending
2) Update runs.status -> ECG_FEATURES_RUNNING
3) Run pipelines.ecg_qc.run_qc + pipelines.ecg_features.run_features
4) Update runs.status -> SUCCEEDED / FAILED
5) Move queue file to done/failed and write task result JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "services" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.ecg_task_queue import (  # noqa: E402
    claim_next_ecg_feature_job,
    get_queue_dirs,
    load_job_payload,
    move_ecg_feature_job,
)
from app.db.models import insert_audit, update_run_status  # noqa: E402
from pipelines.ecg_features import FEATURE_VERSION, run_features  # noqa: E402
from pipelines.ecg_plots import generate_qc_feature_plots  # noqa: E402
from pipelines.ecg_qc import run_qc  # noqa: E402

LOGGER = logging.getLogger("ecg_features_worker")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def _sync_run_params_status(run_dir: Path, *, status: str, error: str | None) -> None:
    params_path = run_dir / "params.json"
    payload: dict[str, Any]
    if params_path.exists():
        try:
            payload = json.loads(params_path.read_text("utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    payload["status"] = status
    payload["updated_at"] = _utc_now_iso()
    if error:
        payload["error"] = error
    _atomic_write_json(params_path, payload)


def _normalize_record_ids(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    ids = [str(v).strip() for v in raw if str(v).strip()]
    return list(dict.fromkeys(ids))


def _write_worker_heartbeat(
    *,
    artifacts_root: Path,
    state: str,
    processed_jobs: int,
    current_job_id: str | None = None,
    error: str | None = None,
) -> None:
    dirs = get_queue_dirs(artifacts_root)
    heartbeat_path = dirs["base"] / "worker_heartbeat.json"
    payload: dict[str, Any] = {
        "updated_at": _utc_now_iso(),
        "state": state,
        "processed_jobs": int(processed_jobs),
        "pid": os.getpid(),
    }
    if current_job_id:
        payload["current_job_id"] = current_job_id
    if error:
        payload["error"] = error
    _atomic_write_json(heartbeat_path, payload)


def _process_job(
    job_path: Path,
    *,
    artifacts_root: Path,
    data_dir: Path,
    global_manifest_path: Path,
    feature_version: str,
) -> bool:
    payload = load_job_payload(job_path)
    run_id = str(payload.get("run_id", "")).strip()
    job_id = str(payload.get("job_id", job_path.stem))
    run_uuid = uuid.UUID(run_id)

    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    record_ids = _normalize_record_ids(payload.get("record_ids"))
    if not record_ids:
        raise RuntimeError("Job has empty record_ids")

    effective_artifacts_root = Path(
        str(params.get("artifacts_root", artifacts_root))
    ).resolve()
    effective_data_dir = Path(str(params.get("data_dir", data_dir))).resolve()
    effective_global_manifest = Path(
        str(params.get("global_manifest", global_manifest_path))
    ).resolve()

    qc_thresholds = params.get("qc_thresholds") if isinstance(params.get("qc_thresholds"), dict) else None
    feature_thresholds = (
        params.get("feature_thresholds") if isinstance(params.get("feature_thresholds"), dict) else None
    )
    effective_feature_version = str(params.get("feature_version", feature_version))
    feature_limit = int(params.get("limit", 0))

    run_dir = effective_artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    task_result_path = run_dir / "ecg_feature_task_result.json"

    started_at = _utc_now_iso()
    update_run_status(run_uuid, "ECG_FEATURES_RUNNING")
    _sync_run_params_status(run_dir, status="ECG_FEATURES_RUNNING", error=None)
    insert_audit(
        run_uuid,
        "worker",
        "ECG_FEATURES_RUNNING",
        {
            "job_id": job_id,
            "record_count": len(record_ids),
            "started_at": started_at,
        },
    )

    qc_path, qc_summary = run_qc(
        run_id=run_id,
        data_dir=effective_data_dir,
        record_ids=record_ids,
        global_manifest_path=effective_global_manifest,
        artifacts_root=effective_artifacts_root,
        thresholds=qc_thresholds,
    )
    features_path, features_summary = run_features(
        run_id=run_id,
        data_dir=effective_data_dir,
        global_manifest_path=effective_global_manifest,
        artifacts_root=effective_artifacts_root,
        qc_path=qc_path,
        record_ids=record_ids,
        limit=feature_limit,
        feature_version=effective_feature_version,
        thresholds=feature_thresholds,
    )
    plots_summary = generate_qc_feature_plots(
        run_id=run_id,
        artifacts_root=effective_artifacts_root,
        data_dir=effective_data_dir,
        global_manifest_path=effective_global_manifest,
    )

    finished_at = _utc_now_iso()
    result_payload = {
        "job_id": job_id,
        "run_id": run_id,
        "status": "SUCCEEDED",
        "started_at": started_at,
        "finished_at": finished_at,
        "qc_path": str(qc_path),
        "features_path": str(features_path),
        "qc_summary": qc_summary,
        "features_summary": features_summary,
        "plots_summary": plots_summary,
    }
    _atomic_write_json(task_result_path, result_payload)
    update_run_status(run_uuid, "SUCCEEDED")
    _sync_run_params_status(run_dir, status="SUCCEEDED", error=None)
    insert_audit(
        run_uuid,
        "worker",
        "ECG_FEATURES_SUCCEEDED",
        {
            "job_id": job_id,
            "finished_at": finished_at,
            "qc_path": str(qc_path),
            "features_path": str(features_path),
            "plots_dir": str(run_dir / "plots"),
        },
    )
    move_ecg_feature_job(job_path, artifacts_root=effective_artifacts_root, state="done")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run worker for ECG feature extraction queue")
    parser.add_argument(
        "--artifacts-root",
        default=str(PROJECT_ROOT / "storage" / "artifacts"),
    )
    parser.add_argument(
        "--data-dir",
        default=str(
            PROJECT_ROOT
            / "data"
            / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
        ),
    )
    parser.add_argument(
        "--global-manifest",
        default=str(PROJECT_ROOT / "storage" / "ecg_manifest.parquet"),
    )
    parser.add_argument("--feature-version", default=FEATURE_VERSION)
    parser.add_argument("--poll-interval-sec", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Process at most one job and exit")
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means unlimited")
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
    artifacts_root = Path(args.artifacts_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    global_manifest_path = Path(args.global_manifest).resolve()

    processed = 0
    _write_worker_heartbeat(
        artifacts_root=artifacts_root,
        state="starting",
        processed_jobs=processed,
    )
    while True:
        if args.max_jobs > 0 and processed >= args.max_jobs:
            break

        _write_worker_heartbeat(
            artifacts_root=artifacts_root,
            state="idle",
            processed_jobs=processed,
        )
        job_path = claim_next_ecg_feature_job(artifacts_root)
        if job_path is None:
            if args.once:
                break
            time.sleep(max(0.2, float(args.poll_interval_sec)))
            continue

        _write_worker_heartbeat(
            artifacts_root=artifacts_root,
            state="running",
            processed_jobs=processed,
            current_job_id=job_path.stem,
        )
        try:
            ok = _process_job(
                job_path,
                artifacts_root=artifacts_root,
                data_dir=data_dir,
                global_manifest_path=global_manifest_path,
                feature_version=str(args.feature_version),
            )
            processed += 1
            LOGGER.info("job_done path=%s ok=%s", job_path, ok)
            _write_worker_heartbeat(
                artifacts_root=artifacts_root,
                state="idle",
                processed_jobs=processed,
            )
        except Exception as exc:
            run_id = "unknown"
            job_id = job_path.stem
            try:
                payload = load_job_payload(job_path)
                run_id = str(payload.get("run_id", "unknown"))
                job_id = str(payload.get("job_id", job_path.stem))
            except Exception:
                payload = {}

            run_dir = artifacts_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            task_result_path = run_dir / "ecg_feature_task_result.json"
            _atomic_write_json(
                task_result_path,
                {
                    "job_id": job_id,
                    "run_id": run_id,
                    "status": "FAILED",
                    "finished_at": _utc_now_iso(),
                    "error": str(exc),
                },
            )

            try:
                run_uuid = uuid.UUID(run_id)
                update_run_status(run_uuid, "FAILED")
                _sync_run_params_status(run_dir, status="FAILED", error=str(exc))
                insert_audit(
                    run_uuid,
                    "worker",
                    "ECG_FEATURES_FAILED",
                    {
                        "job_id": job_id,
                        "error": str(exc),
                    },
                )
            except Exception:
                LOGGER.exception("Failed to update run status after worker error")

            try:
                move_ecg_feature_job(job_path, artifacts_root=artifacts_root, state="failed")
            except Exception:
                LOGGER.exception("Failed to move job to failed queue path=%s", job_path)
            processed += 1
            _write_worker_heartbeat(
                artifacts_root=artifacts_root,
                state="error",
                processed_jobs=processed,
                current_job_id=job_id,
                error=str(exc),
            )
            LOGGER.exception("job_failed path=%s err=%s", job_path, exc)

    _write_worker_heartbeat(
        artifacts_root=artifacts_root,
        state="stopped",
        processed_jobs=processed,
    )
    LOGGER.info("worker_exit processed=%d", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

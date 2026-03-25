"""
ECG batch loader with streaming yield and lightweight cache.

Features:
- Input: a list of record_ids
- Output: yields one record result at a time (memory friendly)
- Manifest cache: skip rebuilding artifacts/<run_id>/ecg_manifest.parquet if hash matches
- Per-record cache: skip QC/feature computation if same parameter-version hash already exists
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

# Allow importing app.core.datasets.reader from services/api
PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "services" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.core.datasets.reader import read_ecg_record  # noqa: E402
from pipelines.ecg_artifacts import write_run_ecg_map  # noqa: E402

LOGGER = logging.getLogger("ecg_load")
PIPELINE_VERSION = "ecg_load_v1"


@dataclass(frozen=True)
class CachePaths:
    run_dir: Path
    manifest_path: Path
    manifest_meta_path: Path
    records_dir: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(data: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _sha256_list(values: Sequence[str]) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(v.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _sanitize_record_id(record_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in record_id)


def _manifest_signature(global_manifest_path: Path) -> dict[str, Any]:
    st = global_manifest_path.stat()
    return {
        "path": str(global_manifest_path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _build_manifest_hash(record_ids: Sequence[str], global_manifest_path: Path) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "record_ids_sha256": _sha256_list(record_ids),
        "record_count": len(record_ids),
        "global_manifest_signature": _manifest_signature(global_manifest_path),
    }
    return _sha256_json(payload)


def _prepare_cache_paths(artifacts_root: Path, run_id: str) -> CachePaths:
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_dir = run_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    return CachePaths(
        run_dir=run_dir,
        manifest_path=run_dir / "ecg_manifest.parquet",
        manifest_meta_path=run_dir / "ecg_manifest.meta.json",
        records_dir=records_dir,
    )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Invalid JSON cache at %s; ignoring", path)
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2), "utf-8"
    )
    tmp.replace(path)


def _write_run_manifest(
    record_ids: Sequence[str],
    global_manifest_path: Path,
    cache_paths: CachePaths,
    manifest_hash: str,
) -> pd.DataFrame:
    df = pd.read_parquet(global_manifest_path, columns=["record_id", "path", "source"])
    req = pd.DataFrame({"record_id": list(record_ids)})
    merged = req.merge(df, on="record_id", how="left")

    missing = merged[merged["path"].isna()]["record_id"].tolist()
    if missing:
        LOGGER.warning("Missing %d record_ids in manifest", len(missing))

    run_manifest_df = merged.dropna(subset=["path"]).copy()
    run_manifest_df.to_parquet(cache_paths.manifest_path, index=False)
    write_run_ecg_map(
        run_dir=cache_paths.run_dir,
        global_manifest_path=global_manifest_path,
        record_ids=run_manifest_df["record_id"].astype(str).tolist(),
    )
    _atomic_write_json(
        cache_paths.manifest_meta_path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "created_at": _utc_now_iso(),
            "manifest_hash": manifest_hash,
            "input_record_count": len(record_ids),
            "resolved_record_count": int(len(run_manifest_df)),
            "missing_record_ids": missing,
        },
    )
    return run_manifest_df


def _ensure_run_manifest(
    record_ids: Sequence[str],
    global_manifest_path: Path,
    cache_paths: CachePaths,
) -> pd.DataFrame:
    manifest_hash = _build_manifest_hash(record_ids, global_manifest_path)
    meta = _load_json_if_exists(cache_paths.manifest_meta_path)

    if (
        cache_paths.manifest_path.exists()
        and meta is not None
        and meta.get("manifest_hash") == manifest_hash
    ):
        LOGGER.info("Manifest cache hit: %s", cache_paths.manifest_path)
        return pd.read_parquet(cache_paths.manifest_path, columns=["record_id", "path", "source"])

    LOGGER.info("Manifest cache miss; rebuilding %s", cache_paths.manifest_path)
    return _write_run_manifest(record_ids, global_manifest_path, cache_paths, manifest_hash)


def _compute_qc_features(
    waveform: np.ndarray,
    fs: int,
    qc_threshold_abs_mv: float = 10.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if waveform.ndim != 2:
        raise ValueError(f"Expected 2D waveform, got shape={waveform.shape}")

    n_samples, n_leads = waveform.shape
    nan_count = int(np.isnan(waveform).sum())
    max_abs = float(np.nanmax(np.abs(waveform))) if n_samples > 0 else 0.0

    qc = {
        "nan_count": nan_count,
        "max_abs_mv": max_abs,
        "duration_sec": float(n_samples / fs) if fs > 0 else 0.0,
        "is_valid": bool(nan_count == 0 and max_abs <= qc_threshold_abs_mv),
    }

    # Lightweight, deterministic features.
    features = {
        "n_samples": int(n_samples),
        "n_leads": int(n_leads),
        "fs": int(fs),
        "mean_per_lead": np.nanmean(waveform, axis=0).round(6).tolist(),
        "std_per_lead": np.nanstd(waveform, axis=0).round(6).tolist(),
        "min_per_lead": np.nanmin(waveform, axis=0).round(6).tolist(),
        "max_per_lead": np.nanmax(waveform, axis=0).round(6).tolist(),
    }
    return qc, features


def iter_ecg_records(
    record_ids: Sequence[str],
    *,
    run_id: str,
    data_dir: str | Path,
    global_manifest_path: str | Path = PROJECT_ROOT / "storage" / "ecg_manifest.parquet",
    artifacts_root: str | Path = PROJECT_ROOT / "storage" / "artifacts",
    qc_params: dict[str, Any] | None = None,
    feature_params: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Stream ECG processing results one by one.

    Yields:
      {
        "record_id": str,
        "cached": bool,
        "status": "ok" | "error",
        "qc": {...},
        "features": {...},
        "error": str | None
      }
    """
    if not record_ids:
        return

    data_dir = Path(data_dir)
    global_manifest_path = Path(global_manifest_path)
    artifacts_root = Path(artifacts_root)
    cache_paths = _prepare_cache_paths(artifacts_root, run_id)

    qc_params = qc_params or {"threshold_abs_mv": 10.0}
    feature_params = feature_params or {"version": "basic_stats_v1"}
    params_hash = _sha256_json(
        {
            "pipeline_version": PIPELINE_VERSION,
            "qc_params": qc_params,
            "feature_params": feature_params,
        }
    )

    run_manifest_df = _ensure_run_manifest(record_ids, global_manifest_path, cache_paths)

    for row in run_manifest_df.itertuples(index=False):
        record_id = str(row.record_id)
        cache_file = cache_paths.records_dir / f"{_sanitize_record_id(record_id)}.json"
        cached_payload = _load_json_if_exists(cache_file)

        if (
            cached_payload is not None
            and cached_payload.get("status") == "ok"
            and cached_payload.get("params_hash") == params_hash
        ):
            yield {
                "record_id": record_id,
                "cached": True,
                "status": "ok",
                "qc": cached_payload.get("qc"),
                "features": cached_payload.get("features"),
                "error": None,
            }
            continue

        try:
            waveform, fs, _lead_names, _meta = read_ecg_record(
                base_dir=data_dir,
                record_path=str(row.path),
                source=str(getattr(row, "source", "mimic_ecg")),
            )
            qc, features = _compute_qc_features(
                waveform, fs, qc_threshold_abs_mv=float(qc_params["threshold_abs_mv"])
            )
            payload = {
                "record_id": record_id,
                "status": "ok",
                "created_at": _utc_now_iso(),
                "params_hash": params_hash,
                "qc": qc,
                "features": features,
            }
            _atomic_write_json(cache_file, payload)
            yield {
                "record_id": record_id,
                "cached": False,
                "status": "ok",
                "qc": qc,
                "features": features,
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - best-effort error capture
            err_payload = {
                "record_id": record_id,
                "status": "error",
                "created_at": _utc_now_iso(),
                "params_hash": params_hash,
                "error": str(exc),
            }
            _atomic_write_json(cache_file, err_payload)
            yield {
                "record_id": record_id,
                "cached": False,
                "status": "error",
                "qc": None,
                "features": None,
                "error": str(exc),
            }
        finally:
            # Release per-record memory aggressively to keep batch RAM stable.
            gc.collect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream ECG loading with resumable cache")
    parser.add_argument("--run-id", required=True, help="Run id used under artifacts/<run_id>")
    parser.add_argument(
        "--data-dir",
        default=str(
            PROJECT_ROOT
            / "data"
            / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
        ),
        help="ECG dataset root directory",
    )
    parser.add_argument(
        "--global-manifest",
        default=str(PROJECT_ROOT / "storage" / "ecg_manifest.parquet"),
        help="Global ECG manifest path",
    )
    parser.add_argument(
        "--artifacts-root",
        default=str(PROJECT_ROOT / "storage" / "artifacts"),
        help="Artifacts root directory",
    )
    parser.add_argument(
        "--record-id",
        action="append",
        default=[],
        help="Record id, can be specified multiple times",
    )
    parser.add_argument(
        "--record-ids-file",
        default=None,
        help="Text file containing one record_id per line",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If record ids are not provided, take first N from global manifest",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _collect_record_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = list(args.record_id)
    if args.record_ids_file:
        path = Path(args.record_ids_file)
        ids.extend(
            [line.strip() for line in path.read_text("utf-8").splitlines() if line.strip()]
        )
    if not ids and args.limit > 0:
        df = pd.read_parquet(args.global_manifest, columns=["record_id"]).head(args.limit)
        ids = df["record_id"].astype(str).tolist()
    # keep order while deduplicating
    return list(dict.fromkeys(ids))


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    record_ids = _collect_record_ids(args)
    if not record_ids:
        LOGGER.error("No record_id provided. Use --record-id/--record-ids-file/--limit.")
        return 2

    processed = 0
    skipped = 0
    errors = 0

    for out in iter_ecg_records(
        record_ids=record_ids,
        run_id=args.run_id,
        data_dir=args.data_dir,
        global_manifest_path=args.global_manifest,
        artifacts_root=args.artifacts_root,
    ):
        if out["status"] == "error":
            errors += 1
        elif out["cached"]:
            skipped += 1
        else:
            processed += 1
        LOGGER.info(
            "record=%s status=%s cached=%s",
            out["record_id"],
            out["status"],
            out["cached"],
        )

    LOGGER.info(
        "Done. total=%d processed=%d skipped=%d errors=%d",
        len(record_ids),
        processed,
        skipped,
        errors,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

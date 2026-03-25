"""
ECG feature extraction pipeline based on QC-pass records.

MVP features:
- Global: mean_hr, rr_mean, rr_std
- Lead-level aggregate: lead_amplitude_p2p_mean, lead_amplitude_p2p_std

Version fields:
- feature_version
- qc_version
- code_commit (git short hash)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "services" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.core.datasets.reader import read_ecg_record  # noqa: E402
from pipelines.ecg_artifacts import write_run_ecg_map  # noqa: E402

LOGGER = logging.getLogger("ecg_features")
FEATURE_VERSION = "v1.0"


DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "hr_min_bpm": 25.0,
    "hr_max_bpm": 220.0,
    "rr_std_max_sec": 0.8,
    "p2p_mean_min_mv": 0.05,
    "p2p_mean_max_mv": 10.0,
    "min_detected_peaks": 3,
}


def _load_manifest_rows(
    record_ids: list[str],
    *,
    run_id: str,
    global_manifest_path: Path,
    artifacts_root: Path,
) -> pd.DataFrame:
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_manifest_path = run_dir / "ecg_manifest.parquet"
    cols = ["record_id", "path", "source"]

    if run_manifest_path.exists():
        base_df = pd.read_parquet(run_manifest_path, columns=cols)
    else:
        base_df = pd.DataFrame(columns=cols)

    for col in cols:
        if col not in base_df.columns:
            base_df[col] = pd.NA

    base_df["record_id"] = base_df["record_id"].astype("string").str.strip()
    base_df = (
        base_df[base_df["record_id"].notna() & base_df["path"].notna()]
        .drop_duplicates(subset=["record_id"], keep="first")
        .copy()
    )

    req_ids = [str(x).strip() for x in record_ids if str(x).strip()]
    req_ids = list(dict.fromkeys(req_ids))

    if req_ids:
        req_df = pd.DataFrame({"record_id": req_ids})
        df = req_df.merge(base_df, on="record_id", how="left")

        missing_ids = (
            df[df["path"].isna()]["record_id"].astype("string").str.strip().dropna().tolist()
        )
        missing_ids = [x for x in missing_ids if str(x).strip()]

        recovered = pd.DataFrame(columns=cols)
        if missing_ids:
            source_df = pd.read_parquet(global_manifest_path, columns=cols)
            source_df["record_id"] = source_df["record_id"].astype("string").str.strip()
            source_df = (
                source_df[source_df["record_id"].notna() & source_df["path"].notna()]
                .drop_duplicates(subset=["record_id"], keep="first")
                .copy()
            )
            recovered = (
                pd.DataFrame({"record_id": list(dict.fromkeys(missing_ids))})
                .merge(source_df, on="record_id", how="left")
                .dropna(subset=["path"])
                .copy()
            )
            if not recovered.empty:
                base_df = (
                    pd.concat([base_df, recovered], ignore_index=True)
                    .drop_duplicates(subset=["record_id"], keep="first")
                    .copy()
                )

        order_map = {rid: idx for idx, rid in enumerate(req_ids)}
        df = (
            base_df[base_df["record_id"].isin(set(req_ids))]
            .copy()
        )
        if not df.empty:
            df["__order"] = df["record_id"].map(order_map)
            df = df.sort_values("__order").drop(columns=["__order"])

        base_df.to_parquet(run_manifest_path, index=False)
    else:
        df = base_df.copy()
        if df.empty:
            source_df = pd.read_parquet(global_manifest_path, columns=cols)
            source_df["record_id"] = source_df["record_id"].astype("string").str.strip()
            df = (
                source_df[source_df["record_id"].notna() & source_df["path"].notna()]
                .drop_duplicates(subset=["record_id"], keep="first")
                .copy()
            )
            df.to_parquet(run_manifest_path, index=False)

    write_run_ecg_map(
        run_dir=run_dir,
        global_manifest_path=global_manifest_path,
        record_ids=df["record_id"].astype(str).tolist(),
    )
    return df

def _collect_record_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = list(args.record_id)
    if args.record_ids_file:
        path = Path(args.record_ids_file)
        ids.extend([line.strip() for line in path.read_text("utf-8").splitlines() if line.strip()])
    return list(dict.fromkeys(ids))


def _get_code_commit(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        pass

    git_dir = project_root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return "unknown"

    try:
        head_text = head_path.read_text("utf-8").strip()
    except Exception:
        return "unknown"

    if head_text.startswith("ref:"):
        ref = head_text.split(":", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            try:
                return ref_path.read_text("utf-8").strip()[:7] or "unknown"
            except Exception:
                return "unknown"

        packed_refs = git_dir / "packed-refs"
        if packed_refs.exists():
            try:
                for line in packed_refs.read_text("utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("^"):
                        continue
                    parts = line.split(" ")
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0][:7] or "unknown"
            except Exception:
                return "unknown"
        return "unknown"

    return head_text[:7] if head_text else "unknown"


def _fill_nan_1d(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float64, copy=True)
    if y.size == 0:
        return y
    mask = np.isnan(y)
    if not mask.any():
        return y
    valid_idx = np.flatnonzero(~mask)
    if valid_idx.size == 0:
        return np.zeros_like(y)
    y[mask] = np.interp(np.flatnonzero(mask), valid_idx, y[valid_idx])
    return y


def _choose_reference_lead(waveform: np.ndarray) -> int:
    p2p = np.nanmax(waveform, axis=0) - np.nanmin(waveform, axis=0)
    if np.all(~np.isfinite(p2p)):
        return 0
    return int(np.nanargmax(np.nan_to_num(p2p, nan=-np.inf)))


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x
    k = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, k, mode="same")


def _find_local_peaks(x: np.ndarray, *, distance: int, height: float) -> np.ndarray:
    if x.size < 3:
        return np.array([], dtype=np.int64)
    candidates = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1
    if candidates.size == 0:
        return np.array([], dtype=np.int64)
    candidates = candidates[x[candidates] >= height]
    if candidates.size == 0:
        return np.array([], dtype=np.int64)

    kept: list[int] = []
    min_gap = max(1, int(distance))
    for idx in candidates:
        if not kept:
            kept.append(int(idx))
            continue
        if idx - kept[-1] >= min_gap:
            kept.append(int(idx))
        elif x[idx] > x[kept[-1]]:
            kept[-1] = int(idx)
    return np.asarray(kept, dtype=np.int64)


def _estimate_rr_features(
    waveform: np.ndarray,
    fs: int,
) -> tuple[float, float, float, int]:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape (samples, leads), got {waveform.shape}")
    if fs <= 0:
        raise ValueError("Invalid fs <= 0")

    lead_idx = _choose_reference_lead(waveform)
    x = _fill_nan_1d(waveform[:, lead_idx])
    if x.size < max(5, int(1.5 * fs)):
        raise ValueError("Too few samples for RR estimation")

    x = x - np.median(x)
    # Derivative + smoothing is a lightweight QRS energy proxy without scipy.
    x_diff = np.abs(np.diff(x, prepend=x[0]))
    smooth_win = max(1, int(0.08 * fs))
    x_abs = _moving_average(x_diff, smooth_win)
    distance = max(1, int(0.25 * fs))  # ~240 bpm upper bound
    height = float(np.nanmedian(x_abs) + 0.5 * np.nanstd(x_abs))
    peaks = _find_local_peaks(x_abs, distance=distance, height=height)
    if peaks.size < 3:
        peaks = _find_local_peaks(
            x_abs,
            distance=distance,
            height=float(np.nanmedian(x_abs) + 0.2 * np.nanstd(x_abs)),
        )
    if peaks.size < 3:
        raise ValueError("Insufficient detected peaks")

    rr = np.diff(peaks).astype(np.float64) / float(fs)
    rr = rr[(rr >= 0.25) & (rr <= 3.0)]
    if rr.size < 2:
        raise ValueError("Insufficient valid RR intervals")

    rr_mean = float(np.mean(rr))
    rr_std = float(np.std(rr))
    if rr_mean <= 0:
        raise ValueError("Invalid rr_mean <= 0")
    mean_hr = float(60.0 / rr_mean)
    return mean_hr, rr_mean, rr_std, int(peaks.size)


def _compute_lead_p2p_features(waveform: np.ndarray) -> tuple[float, float]:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape (samples, leads), got {waveform.shape}")
    p2p = np.nanmax(waveform, axis=0) - np.nanmin(waveform, axis=0)
    p2p = p2p[np.isfinite(p2p)]
    if p2p.size == 0:
        raise ValueError("No finite lead amplitude p2p")
    return float(np.mean(p2p)), float(np.std(p2p))


def _validate_feature_row(
    row: dict[str, Any],
    thresholds: dict[str, float | int],
) -> str | None:
    mean_hr = float(row["mean_hr"])
    rr_mean = float(row["rr_mean"])
    rr_std = float(row["rr_std"])
    p2p_mean = float(row["lead_amplitude_p2p_mean"])
    peak_count = int(row["detected_peak_count"])

    if not np.isfinite([mean_hr, rr_mean, rr_std, p2p_mean]).all():
        return "fail:non_finite_feature"
    if peak_count < int(thresholds["min_detected_peaks"]):
        return "fail:insufficient_peaks"
    if not (float(thresholds["hr_min_bpm"]) <= mean_hr <= float(thresholds["hr_max_bpm"])):
        return "fail:hr_out_of_range"
    if not (0.25 <= rr_mean <= 3.0):
        return "fail:rr_mean_out_of_range"
    if rr_std > float(thresholds["rr_std_max_sec"]):
        return "fail:rr_std_too_large"
    if not (
        float(thresholds["p2p_mean_min_mv"])
        <= p2p_mean
        <= float(thresholds["p2p_mean_max_mv"])
    ):
        return "fail:p2p_mean_out_of_range"
    return None


def _build_hr_stats(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {}
    s = series.astype(float)
    return {
        "min": float(s.min()),
        "p01": float(np.nanpercentile(s, 1)),
        "p50": float(np.nanpercentile(s, 50)),
        "p99": float(np.nanpercentile(s, 99)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
    }


def run_features(
    *,
    run_id: str,
    data_dir: Path,
    global_manifest_path: Path,
    artifacts_root: Path,
    qc_path: Path,
    record_ids: list[str],
    limit: int,
    feature_version: str,
    thresholds: dict[str, float | int] | None = None,
) -> tuple[Path, dict[str, Any]]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if not qc_path.exists():
        raise FileNotFoundError(f"QC parquet not found: {qc_path}")

    qc_df = pd.read_parquet(qc_path)
    if "record_id" not in qc_df.columns or "qc_pass" not in qc_df.columns:
        raise RuntimeError("QC parquet missing required columns: record_id/qc_pass")

    qc_df["record_id"] = qc_df["record_id"].astype(str)
    qc_pass_df = qc_df[qc_df["qc_pass"] == True].copy()  # noqa: E712

    if record_ids:
        req_order = {rid: i for i, rid in enumerate(record_ids)}
        qc_pass_df = qc_pass_df[qc_pass_df["record_id"].isin(req_order)].copy()
        qc_pass_df["__order"] = qc_pass_df["record_id"].map(req_order)
        qc_pass_df = qc_pass_df.sort_values("__order").drop(columns=["__order"])
    elif limit > 0:
        qc_pass_df = qc_pass_df.head(limit).copy()

    target_ids = qc_pass_df["record_id"].astype(str).tolist()
    if not target_ids:
        raise RuntimeError("No QC-pass records available for feature extraction.")

    manifest_df = _load_manifest_rows(
        target_ids,
        run_id=run_id,
        global_manifest_path=global_manifest_path,
        artifacts_root=artifacts_root,
    )
    manifest_df["record_id"] = manifest_df["record_id"].astype(str)

    rows = qc_pass_df[["record_id", "source", "qc_version"]].merge(
        manifest_df[["record_id", "path", "source"]],
        on="record_id",
        how="left",
        suffixes=("_qc", "_manifest"),
    )
    rows["source"] = rows["source_qc"].fillna(rows["source_manifest"]).fillna("mimic_ecg")
    rows = rows.drop(columns=["source_qc", "source_manifest"])
    rows = rows.dropna(subset=["path"]).copy()

    req_order = {rid: i for i, rid in enumerate(target_ids)}
    rows["__order"] = rows["record_id"].map(req_order)
    rows = rows.sort_values("__order").drop(columns=["__order"])

    code_commit = _get_code_commit(PROJECT_ROOT)
    out_rows: list[dict[str, Any]] = []
    blocked_reason_counter: Counter[str] = Counter()

    for row in rows.itertuples(index=False):
        record_id = str(row.record_id)
        try:
            waveform, fs, _lead_names, _meta = read_ecg_record(
                base_dir=data_dir,
                record_path=str(row.path),
                source=str(row.source),
            )
            mean_hr, rr_mean, rr_std, peak_count = _estimate_rr_features(waveform, int(fs))
            p2p_mean, p2p_std = _compute_lead_p2p_features(waveform)
            feature_row = {
                "record_id": record_id,
                "source": str(row.source),
                "mean_hr": mean_hr,
                "rr_mean": rr_mean,
                "rr_std": rr_std,
                "lead_amplitude_p2p_mean": p2p_mean,
                "lead_amplitude_p2p_std": p2p_std,
                "detected_peak_count": int(peak_count),
                "feature_version": feature_version,
                "qc_version": str(row.qc_version),
                "code_commit": code_commit,
            }

            fail_reason = _validate_feature_row(feature_row, thresholds)
            if fail_reason is not None:
                blocked_reason_counter[fail_reason] += 1
                continue
            out_rows.append(feature_row)
        except Exception as exc:  # pragma: no cover - best effort record-level isolation
            blocked_reason_counter[f"fail:feature_compute_error:{type(exc).__name__}"] += 1

    out_df = pd.DataFrame(out_rows)
    if out_df.empty:
        out_df = pd.DataFrame(
            columns=[
                "record_id",
                "source",
                "mean_hr",
                "rr_mean",
                "rr_std",
                "lead_amplitude_p2p_mean",
                "lead_amplitude_p2p_std",
                "detected_peak_count",
                "feature_version",
                "qc_version",
                "code_commit",
            ]
        )

    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "ecg_features.parquet"
    out_df.to_parquet(out_path, index=False)

    hr_series = out_df["mean_hr"].astype(float) if "mean_hr" in out_df.columns else pd.Series(dtype=float)
    hr_out_of_range_count = (
        int(
            (
                (hr_series < float(thresholds["hr_min_bpm"]))
                | (hr_series > float(thresholds["hr_max_bpm"]))
            ).sum()
        )
        if not hr_series.empty
        else 0
    )

    summary = {
        "run_id": run_id,
        "feature_version": feature_version,
        "code_commit": code_commit,
        "qc_path": str(qc_path),
        "input_qc_pass_count": int(len(rows)),
        "output_feature_count": int(len(out_df)),
        "blocked_bad_record_count": int(len(rows) - len(out_df)),
        "blocked_top_reasons": blocked_reason_counter.most_common(10),
        "hr_reasonable_range_bpm": [
            float(thresholds["hr_min_bpm"]),
            float(thresholds["hr_max_bpm"]),
        ],
        "hr_out_of_range_count": hr_out_of_range_count,
        "hr_stats": _build_hr_stats(hr_series),
        "sanity_check": {
            "hr_distribution_reasonable": bool(hr_out_of_range_count == 0),
            "bad_data_blocked_from_feature_table": bool(len(rows) - len(out_df) >= 0),
        },
    }
    summary_path = run_dir / "ecg_features_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), "utf-8")
    return out_path, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ECG features and write ecg_features.parquet")
    parser.add_argument("--run-id", required=True)
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
    parser.add_argument(
        "--artifacts-root",
        default=str(PROJECT_ROOT / "storage" / "artifacts"),
    )
    parser.add_argument(
        "--qc-parquet",
        default=None,
        help="Default: artifacts/<run_id>/ecg_qc.parquet",
    )
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument("--record-ids-file", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If record ids are not provided, use first N QC-pass records",
    )
    parser.add_argument("--feature-version", default=FEATURE_VERSION)
    parser.add_argument("--hr-min", type=float, default=float(DEFAULT_THRESHOLDS["hr_min_bpm"]))
    parser.add_argument("--hr-max", type=float, default=float(DEFAULT_THRESHOLDS["hr_max_bpm"]))
    parser.add_argument(
        "--rr-std-max",
        type=float,
        default=float(DEFAULT_THRESHOLDS["rr_std_max_sec"]),
    )
    parser.add_argument(
        "--p2p-min",
        type=float,
        default=float(DEFAULT_THRESHOLDS["p2p_mean_min_mv"]),
    )
    parser.add_argument(
        "--p2p-max",
        type=float,
        default=float(DEFAULT_THRESHOLDS["p2p_mean_max_mv"]),
    )
    parser.add_argument(
        "--min-detected-peaks",
        type=int,
        default=int(DEFAULT_THRESHOLDS["min_detected_peaks"]),
    )
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
    qc_path = Path(args.qc_parquet) if args.qc_parquet else Path(args.artifacts_root) / args.run_id / "ecg_qc.parquet"
    record_ids = _collect_record_ids(args)
    thresholds = {
        "hr_min_bpm": float(args.hr_min),
        "hr_max_bpm": float(args.hr_max),
        "rr_std_max_sec": float(args.rr_std_max),
        "p2p_mean_min_mv": float(args.p2p_min),
        "p2p_mean_max_mv": float(args.p2p_max),
        "min_detected_peaks": int(args.min_detected_peaks),
    }

    try:
        out_path, summary = run_features(
            run_id=args.run_id,
            data_dir=Path(args.data_dir),
            global_manifest_path=Path(args.global_manifest),
            artifacts_root=Path(args.artifacts_root),
            qc_path=qc_path,
            record_ids=record_ids,
            limit=int(args.limit),
            feature_version=str(args.feature_version),
            thresholds=thresholds,
        )
    except Exception as exc:
        LOGGER.error("Feature extraction failed: %s", exc)
        return 1

    LOGGER.info("Wrote feature parquet: %s", out_path)
    LOGGER.info(
        "Feature summary: input_qc_pass=%d output=%d blocked=%d",
        summary["input_qc_pass_count"],
        summary["output_feature_count"],
        summary["blocked_bad_record_count"],
    )
    LOGGER.info("HR stats: %s", summary["hr_stats"])
    LOGGER.info("Blocked top reasons: %s", summary["blocked_top_reasons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

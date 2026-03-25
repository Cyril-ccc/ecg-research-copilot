"""
Per-record ECG QC pipeline.

Outputs (per ECG):
- qc_pass: bool
- qc_reasons: list[str]
- metrics:
  - missing_leads
  - flatline_ratio
  - clipping_ratio
  - nan_ratio
  - amplitude_range
  - powerline_score
  - baseline_wander_score
"""

from __future__ import annotations

import argparse
import json
import logging
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

LOGGER = logging.getLogger("ecg_qc")
QC_VERSION = "ecg_qc_v1"


DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "lead_count_warn_threshold": 12,
    "nan_ratio_fail_threshold": 0.20,
    "flatline_ratio_fail_threshold": 0.25,
    "clipping_ratio_fail_threshold": 0.05,
    "powerline_warn_threshold": 0.30,
    "baseline_wander_warn_threshold": 0.40,
    "clip_abs_mv_threshold": 5.0,
    "flatline_diff_eps": 1e-5,
    "flatline_min_duration_sec": 0.2,
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

def _max_run_ratio(mask_1d: np.ndarray) -> float:
    n = int(mask_1d.size)
    if n == 0:
        return 0.0
    max_run = 0
    run = 0
    for v in mask_1d:
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return float(max_run / n)


def _band_energy_ratio(
    x: np.ndarray,
    fs: int,
    *,
    band_low: float,
    band_high: float,
    ref_low: float,
    ref_high: float,
) -> float:
    if x.size == 0 or fs <= 0:
        return 0.0
    x = x.astype(np.float64, copy=False)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0)
    n = x.size
    if n < 4:
        return 0.0
    spec = np.fft.rfft(x)
    power = (np.abs(spec) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    band_mask = (freqs >= band_low) & (freqs <= band_high)
    ref_mask = (freqs >= ref_low) & (freqs <= ref_high)
    band_e = float(power[band_mask].sum())
    ref_e = float(power[ref_mask].sum())
    if ref_e <= 0:
        return 0.0
    return band_e / ref_e


def _compute_metrics(
    waveform: np.ndarray,
    fs: int,
    lead_names: list[str],
    thresholds: dict[str, float | int],
) -> dict[str, Any]:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape (samples, leads), got {waveform.shape}")
    n_samples, n_leads = waveform.shape
    if len(lead_names) != n_leads:
        lead_names = [f"lead_{i+1}" for i in range(n_leads)]

    diff_eps = float(thresholds["flatline_diff_eps"])
    clip_abs_mv = float(thresholds["clip_abs_mv_threshold"])
    min_flatline_len = max(2, int(float(thresholds["flatline_min_duration_sec"]) * max(fs, 1)))

    missing_leads: list[str] = []
    nan_ratio_per_lead: dict[str, float] = {}
    flatline_ratio_per_lead: dict[str, float] = {}
    clipping_ratio_per_lead: dict[str, float] = {}
    amplitude_range: dict[str, dict[str, float]] = {}
    powerline_scores: dict[str, float] = {}
    baseline_scores: dict[str, float] = {}

    # Infer whether powerline should be 50Hz or 60Hz.
    powerline_hz = 60.0 if fs >= 120 else 50.0
    powerline_low = max(1.0, powerline_hz - 1.0)
    powerline_high = powerline_hz + 1.0

    for i, lead in enumerate(lead_names):
        x = waveform[:, i]
        nan_mask = np.isnan(x)
        nan_ratio = float(nan_mask.mean()) if n_samples > 0 else 1.0
        nan_ratio_per_lead[lead] = nan_ratio

        non_nan = x[~nan_mask]
        all_nan = non_nan.size == 0
        all_zero = bool(non_nan.size > 0 and np.all(np.abs(non_nan) <= diff_eps))
        if all_nan or all_zero:
            missing_leads.append(lead)

        if non_nan.size > 0:
            lead_min = float(np.min(non_nan))
            lead_max = float(np.max(non_nan))
        else:
            lead_min = 0.0
            lead_max = 0.0
        amplitude_range[lead] = {
            "min": lead_min,
            "max": lead_max,
            "p2p": float(lead_max - lead_min),
        }

        # flatline_ratio: ratio of samples belonging to the longest near-constant segment
        # (diff below eps), converted to sample ratio.
        if non_nan.size >= 2:
            d = np.abs(np.diff(np.nan_to_num(x, nan=0.0)))
            flat_mask = d <= diff_eps
            max_run_ratio = _max_run_ratio(flat_mask)
            # apply minimum run constraint; if below it, count as no flatline.
            max_run_len = int(max_run_ratio * d.size)
            flatline_ratio = float(max_run_ratio if max_run_len >= min_flatline_len else 0.0)
        else:
            flatline_ratio = 1.0
        flatline_ratio_per_lead[lead] = flatline_ratio

        if n_samples > 0:
            clip_mask = np.abs(np.nan_to_num(x, nan=0.0)) >= clip_abs_mv
            clipping_ratio = float(np.mean(clip_mask))
        else:
            clipping_ratio = 0.0
        clipping_ratio_per_lead[lead] = clipping_ratio

        powerline_scores[lead] = _band_energy_ratio(
            x,
            fs,
            band_low=powerline_low,
            band_high=powerline_high,
            ref_low=1.0,
            ref_high=min(120.0, fs / 2.0 - 1.0) if fs > 4 else 2.0,
        )
        baseline_scores[lead] = _band_energy_ratio(
            x,
            fs,
            band_low=0.0,
            band_high=0.5,
            ref_low=0.0,
            ref_high=min(40.0, fs / 2.0 - 1.0) if fs > 4 else 2.0,
        )

    return {
        "n_samples": int(n_samples),
        "n_leads": int(n_leads),
        "powerline_hz": powerline_hz,
        "missing_leads": missing_leads,
        "nan_ratio": float(max(nan_ratio_per_lead.values(), default=0.0)),
        "nan_ratio_per_lead": nan_ratio_per_lead,
        "flatline_ratio": float(max(flatline_ratio_per_lead.values(), default=0.0)),
        "flatline_ratio_per_lead": flatline_ratio_per_lead,
        "clipping_ratio": float(max(clipping_ratio_per_lead.values(), default=0.0)),
        "clipping_ratio_per_lead": clipping_ratio_per_lead,
        "amplitude_range": amplitude_range,
        "powerline_score": float(max(powerline_scores.values(), default=0.0)),
        "powerline_score_per_lead": powerline_scores,
        "baseline_wander_score": float(max(baseline_scores.values(), default=0.0)),
        "baseline_wander_score_per_lead": baseline_scores,
    }


def _apply_qc_rules(
    metrics: dict[str, Any],
    thresholds: dict[str, float | int],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    n_leads = int(metrics["n_leads"])
    if n_leads < int(thresholds["lead_count_warn_threshold"]):
        reasons.append("warn:lead_count_below_12")

    missing_leads = metrics["missing_leads"]
    if missing_leads:
        reasons.append(f"fail:missing_leads:{','.join(missing_leads)}")

    nan_fail_leads = [
        lead
        for lead, ratio in metrics["nan_ratio_per_lead"].items()
        if ratio > float(thresholds["nan_ratio_fail_threshold"])
    ]
    if nan_fail_leads:
        reasons.append(f"fail:nan_ratio_high:{','.join(nan_fail_leads)}")

    if float(metrics["flatline_ratio"]) >= float(thresholds["flatline_ratio_fail_threshold"]):
        reasons.append("fail:flatline_ratio_high")

    if float(metrics["clipping_ratio"]) >= float(thresholds["clipping_ratio_fail_threshold"]):
        reasons.append("fail:clipping_ratio_high")

    if float(metrics["powerline_score"]) >= float(thresholds["powerline_warn_threshold"]):
        reasons.append("warn:powerline_noise_high")

    if float(metrics["baseline_wander_score"]) >= float(thresholds["baseline_wander_warn_threshold"]):
        reasons.append("warn:baseline_wander_high")

    qc_pass = not any(r.startswith("fail:") for r in reasons)
    return qc_pass, reasons


def run_qc(
    *,
    run_id: str,
    data_dir: Path,
    record_ids: list[str],
    global_manifest_path: Path,
    artifacts_root: Path,
    thresholds: dict[str, float | int] | None = None,
) -> tuple[Path, dict[str, Any]]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rows = _load_manifest_rows(
        record_ids,
        run_id=run_id,
        global_manifest_path=global_manifest_path,
        artifacts_root=artifacts_root,
    )
    if rows.empty:
        raise RuntimeError("No valid records found in manifest for given record_ids.")

    out_rows: list[dict[str, Any]] = []
    fail_reason_counter: Counter[str] = Counter()

    for row in rows.itertuples(index=False):
        record_id = str(row.record_id)
        try:
            waveform, fs, lead_names, _meta = read_ecg_record(
                base_dir=data_dir,
                record_path=str(row.path),
                source=str(getattr(row, "source", "mimic_ecg")),
            )
            metrics = _compute_metrics(waveform, int(fs), list(lead_names), thresholds)
            qc_pass, reasons = _apply_qc_rules(metrics, thresholds)

            for r in reasons:
                if r.startswith("fail:"):
                    fail_reason_counter[r] += 1

            out_rows.append(
                {
                    "record_id": record_id,
                    "source": str(getattr(row, "source", "mimic_ecg")),
                    "fs": int(fs),
                    "n_leads": int(metrics["n_leads"]),
                    "qc_pass": bool(qc_pass),
                    "qc_reasons": reasons,
                    "missing_leads": metrics["missing_leads"],
                    "flatline_ratio": float(metrics["flatline_ratio"]),
                    "clipping_ratio": float(metrics["clipping_ratio"]),
                    "nan_ratio": float(metrics["nan_ratio"]),
                    "amplitude_range": metrics["amplitude_range"],
                    "powerline_score": float(metrics["powerline_score"]),
                    "baseline_wander_score": float(metrics["baseline_wander_score"]),
                    "qc_version": QC_VERSION,
                }
            )
        except Exception as exc:
            fail_reason_counter["fail:read_or_compute_error"] += 1
            out_rows.append(
                {
                    "record_id": record_id,
                    "source": str(getattr(row, "source", "mimic_ecg")),
                    "fs": None,
                    "n_leads": None,
                    "qc_pass": False,
                    "qc_reasons": [f"fail:read_or_compute_error:{exc}"],
                    "missing_leads": [],
                    "flatline_ratio": None,
                    "clipping_ratio": None,
                    "nan_ratio": None,
                    "amplitude_range": {},
                    "powerline_score": None,
                    "baseline_wander_score": None,
                    "qc_version": QC_VERSION,
                }
            )

    out_df = pd.DataFrame(out_rows)
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "ecg_qc.parquet"
    out_df.to_parquet(out_path, index=False)

    total = len(out_df)
    pass_count = int(out_df["qc_pass"].sum()) if total > 0 else 0
    fail_count = total - pass_count
    pass_ratio = float(pass_count / total) if total > 0 else 0.0

    summary = {
        "qc_version": QC_VERSION,
        "run_id": run_id,
        "total": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_ratio": pass_ratio,
        "fail_top_reasons": fail_reason_counter.most_common(10),
    }
    summary_path = run_dir / "ecg_qc_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), "utf-8")
    return out_path, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ECG QC and write ecg_qc.parquet")
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
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument("--record-ids-file", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _collect_record_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = list(args.record_id)
    if args.record_ids_file:
        p = Path(args.record_ids_file)
        ids.extend([line.strip() for line in p.read_text("utf-8").splitlines() if line.strip()])
    if not ids and args.limit > 0:
        df = pd.read_parquet(args.global_manifest, columns=["record_id"]).head(args.limit)
        ids = df["record_id"].astype(str).tolist()
    return list(dict.fromkeys(ids))


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    record_ids = _collect_record_ids(args)
    if not record_ids:
        LOGGER.error("No record ids provided.")
        return 2

    out_path, summary = run_qc(
        run_id=args.run_id,
        data_dir=Path(args.data_dir),
        record_ids=record_ids,
        global_manifest_path=Path(args.global_manifest),
        artifacts_root=Path(args.artifacts_root),
    )

    LOGGER.info("Wrote QC parquet: %s", out_path)
    LOGGER.info(
        "QC summary: total=%d pass=%d fail=%d pass_ratio=%.4f",
        summary["total"],
        summary["pass_count"],
        summary["fail_count"],
        summary["pass_ratio"],
    )
    LOGGER.info("Fail top reasons: %s", summary["fail_top_reasons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

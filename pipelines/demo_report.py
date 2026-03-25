"""
Run a demo end-to-end reporting pipeline on local demo ECG data.

Chain:
build_cohort (demo cohort parquet) -> ecg_qc -> ecg_features
-> assemble_analysis_dataset -> build_analysis_tables
-> build_report_plots -> generate_report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "services" / "api"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from pipelines.assemble_analysis_dataset import assemble_analysis_dataset  # noqa: E402
from pipelines.build_analysis_tables import build_analysis_tables  # noqa: E402
from pipelines.build_report_plots import build_report_plots  # noqa: E402
from pipelines.ecg_features import run_features  # noqa: E402
from pipelines.ecg_plots import generate_qc_feature_plots  # noqa: E402
from pipelines.ecg_qc import run_qc  # noqa: E402
from pipelines.generate_report import generate_report  # noqa: E402

LOGGER = logging.getLogger("demo_report")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _build_demo_cohort(
    *,
    run_dir: Path,
    manifest_subset: pd.DataFrame,
) -> tuple[Path, Path, dict[str, Any]]:
    cohort_df = manifest_subset.copy()
    cohort_df["subject_id"] = cohort_df["subject_id"].astype("string")

    if "ecg_time" in cohort_df.columns:
        cohort_df["index_time"] = pd.to_datetime(cohort_df["ecg_time"], errors="coerce", utc=True)
    else:
        cohort_df["index_time"] = pd.NaT
    if cohort_df["index_time"].isna().all():
        cohort_df["index_time"] = pd.Timestamp("2100-01-01T00:00:00+00:00")

    cohort_df = (
        cohort_df.sort_values(["subject_id", "index_time"], na_position="last")
        .drop_duplicates(subset=["subject_id"], keep="first")
        .reset_index(drop=True)
    )
    cohort_df["hadm_id"] = pd.NA
    if len(cohort_df) >= 2:
        cohort_df["cohort_label"] = [
            "demo_A" if i % 2 == 0 else "demo_B" for i in range(len(cohort_df))
        ]
    else:
        cohort_df["cohort_label"] = "demo_A"

    cohort_df = cohort_df[["subject_id", "hadm_id", "index_time", "cohort_label"]]

    cohort_path = run_dir / "cohort.parquet"
    cohort_df.to_parquet(cohort_path, index=False)

    summary: dict[str, Any] = {
        "distinct_subjects": int(cohort_df["subject_id"].nunique(dropna=True)),
        "total_rows": int(len(cohort_df)),
        "missing_rates": {
            k: float(v)
            for k, v in (
                cohort_df.isna().sum() / max(1, len(cohort_df))
            ).to_dict().items()
        },
    }
    if "index_time" in cohort_df.columns and len(cohort_df) > 0:
        summary["index_time_range"] = {
            "min": str(cohort_df["index_time"].min()),
            "max": str(cohort_df["index_time"].max()),
        }

    summary_path = run_dir / "cohort_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return cohort_path, summary_path, summary


def run_demo_report(
    *,
    run_id: str,
    artifacts_root: Path,
    data_dir: Path,
    global_manifest_path: Path,
    sample_n: int = 10,
    question: str = "Demo end-to-end report on local ECG sample",
) -> dict[str, Any]:
    if sample_n < 1:
        raise ValueError("sample_n must be >= 1")
    if not data_dir.exists():
        raise FileNotFoundError(f"ECG data dir not found: {data_dir}")
    if not global_manifest_path.exists():
        raise FileNotFoundError(f"Global manifest not found: {global_manifest_path}")

    manifest_df = pd.read_parquet(global_manifest_path)
    required_cols = {"record_id", "subject_id"}
    missing = sorted(required_cols - set(manifest_df.columns))
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    subset = manifest_df.copy()
    subset["record_id"] = subset["record_id"].astype("string").str.strip()
    subset["subject_id"] = subset["subject_id"].astype("string").str.strip()
    subset = subset.dropna(subset=["record_id", "subject_id"]).copy()
    subset = subset[subset["record_id"] != ""]
    subset = subset[subset["subject_id"] != ""]
    subset = subset.drop_duplicates(subset=["record_id"], keep="first")
    subset = subset.head(sample_n).copy()
    if subset.empty:
        raise RuntimeError("No valid demo record_id/subject_id rows available in manifest")

    record_ids = subset["record_id"].astype(str).tolist()
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    params_payload = {
        "run_id": run_id,
        "created_at": _utc_now_iso(),
        "question": question,
        "params": {
            "template_name": "demo_build_cohort",
            "sample_n": len(record_ids),
            "record_ids_preview": record_ids[:5],
        },
        "status": "RUNNING",
        "artifacts_path": str(run_dir),
    }
    (run_dir / "params.json").write_text(
        json.dumps(params_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    cohort_path, cohort_summary_path, cohort_summary = _build_demo_cohort(
        run_dir=run_dir,
        manifest_subset=subset,
    )

    qc_path, qc_summary = run_qc(
        run_id=run_id,
        data_dir=data_dir,
        record_ids=record_ids,
        global_manifest_path=global_manifest_path,
        artifacts_root=artifacts_root,
        thresholds={
            "lead_count_warn_threshold": 0,
            "nan_ratio_fail_threshold": 1.1,
            "flatline_ratio_fail_threshold": 1.1,
            "clipping_ratio_fail_threshold": 1.1,
            "powerline_warn_threshold": 99.0,
            "baseline_wander_warn_threshold": 99.0,
        },
    )

    features_path, features_summary = run_features(
        run_id=run_id,
        data_dir=data_dir,
        global_manifest_path=global_manifest_path,
        artifacts_root=artifacts_root,
        qc_path=qc_path,
        record_ids=record_ids,
        limit=0,
        feature_version="v1.0-demo",
        thresholds={
            "hr_min_bpm": 1.0,
            "hr_max_bpm": 300.0,
            "rr_std_max_sec": 10.0,
            "p2p_mean_min_mv": 0.0,
            "p2p_mean_max_mv": 100.0,
            "min_detected_peaks": 2,
        },
    )

    qc_plots_summary = generate_qc_feature_plots(
        run_id=run_id,
        artifacts_root=artifacts_root,
        data_dir=data_dir,
        global_manifest_path=global_manifest_path,
    )

    ecg_map_path = run_dir / "ecg_map.parquet"
    analysis_dataset_path, analysis_summary = assemble_analysis_dataset(
        run_id=run_id,
        artifacts_root=artifacts_root,
        cohort_path=cohort_path,
        features_path=features_path,
        ecg_map_path=ecg_map_path,
        global_manifest_path=global_manifest_path,
        covariates_path=None,
        age_bin_mode="fixed",
    )

    feature_summary_path, group_compare_path, analysis_tables_summary = build_analysis_tables(
        run_id=run_id,
        artifacts_root=artifacts_root,
        analysis_dataset_path=analysis_dataset_path,
        group_cols=["cohort_label", "sex", "age_bin"],
        compare_by="cohort_label",
        compare_features=["mean_hr", "rr_std", "rr_mean"],
    )

    report_plots_summary = build_report_plots(
        run_id=run_id,
        artifacts_root=artifacts_root,
        analysis_dataset_path=analysis_dataset_path,
        qc_path=qc_path,
        fail_top_n=8,
        feature_preferred=["rr_std", "qtc", "qtc_ms"],
    )

    report_path, metadata_path, report_meta = generate_report(
        run_id=run_id,
        artifacts_root=artifacts_root,
        question_arg=question,
        params_json_arg=json.dumps(params_payload["params"], ensure_ascii=True),
    )

    params_payload["status"] = "SUCCEEDED"
    params_payload["updated_at"] = _utc_now_iso()
    (run_dir / "params.json").write_text(
        json.dumps(params_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    summary = {
        "run_id": run_id,
        "sample_n": len(record_ids),
        "cohort_summary": cohort_summary,
        "qc_summary": qc_summary,
        "features_summary": features_summary,
        "qc_plots_summary": qc_plots_summary,
        "analysis_summary": analysis_summary,
        "analysis_tables_summary": analysis_tables_summary,
        "report_plots_summary": report_plots_summary,
        "report_status": report_meta.get("status"),
        "report_path": str(report_path),
        "metadata_path": str(metadata_path),
        "output_files": [
            str(cohort_path),
            str(cohort_summary_path),
            str(qc_path),
            str(features_path),
            str(ecg_map_path),
            str(feature_summary_path),
            str(group_compare_path),
            str(report_path),
            str(metadata_path),
        ],
    }
    (run_dir / "demo_report_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end demo report pipeline")
    parser.add_argument("--run-id", default="demo-report")
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
    parser.add_argument("--sample-n", type=int, default=10)
    parser.add_argument("--question", default="Demo end-to-end report on local ECG sample")
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

    try:
        summary = run_demo_report(
            run_id=str(args.run_id),
            artifacts_root=Path(args.artifacts_root).resolve(),
            data_dir=Path(args.data_dir).resolve(),
            global_manifest_path=Path(args.global_manifest).resolve(),
            sample_n=int(args.sample_n),
            question=str(args.question),
        )
    except Exception as exc:
        LOGGER.error("demo_report failed: %s", exc)
        return 1

    LOGGER.info("demo_report done run_id=%s sample_n=%s", summary["run_id"], summary["sample_n"])
    LOGGER.info("report=%s", summary["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

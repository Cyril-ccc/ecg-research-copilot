from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.build_report_plots import build_report_plots  # noqa: E402


def test_build_report_plots(tmp_path: Path):
    run_id = "plots-unit"
    artifacts_root = tmp_path
    run_dir = artifacts_root / run_id
    analysis_dir = run_dir / "analysis_tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = pd.DataFrame(
        [
            {"record_id": "r1", "cohort_label": "A", "sex": "F", "age_bin": "<40", "mean_hr": 70.0, "rr_std": 0.08},
            {"record_id": "r2", "cohort_label": "A", "sex": "M", "age_bin": "40-60", "mean_hr": 88.0, "rr_std": 0.12},
            {"record_id": "r3", "cohort_label": "B", "sex": "F", "age_bin": "<40", "mean_hr": 95.0, "rr_std": 0.19},
            {"record_id": "r4", "cohort_label": "B", "sex": "M", "age_bin": ">60", "mean_hr": 110.0, "rr_std": 0.15},
            {"record_id": "r5", "cohort_label": "B", "sex": "M", "age_bin": ">60", "mean_hr": 105.0, "rr_std": 0.09},
        ]
    )
    qc_df = pd.DataFrame(
        [
            {"record_id": "r1", "qc_pass": True, "qc_reasons": []},
            {"record_id": "r2", "qc_pass": False, "qc_reasons": ["flatline"]},
            {"record_id": "r3", "qc_pass": True, "qc_reasons": []},
            {"record_id": "r4", "qc_pass": False, "qc_reasons": ["nan_ratio", "baseline_wander"]},
            {"record_id": "r5", "qc_pass": True, "qc_reasons": []},
        ]
    )

    analysis_path = analysis_dir / "analysis_dataset.parquet"
    qc_path = run_dir / "ecg_qc.parquet"
    analysis_df.to_parquet(analysis_path, index=False)
    qc_df.to_parquet(qc_path, index=False)

    summary = build_report_plots(
        run_id=run_id,
        artifacts_root=artifacts_root,
        analysis_dataset_path=analysis_path,
        qc_path=qc_path,
        fail_top_n=3,
        feature_preferred=["rr_std", "qtc"],
    )

    plots_dir = run_dir / "plots"
    expected = [
        plots_dir / "qc_pass_rate.png",
        plots_dir / "hr_distribution_by_group.png",
        plots_dir / "feature_boxplot_qtc_or_rrstd.png",
    ]
    for p in expected:
        assert p.exists()
        assert p.stat().st_size > 0

    assert summary["box_summary"]["feature_name"] == "rr_std"
    assert summary["hr_summary"]["display_mode"] == "cohort_x_sex"
    assert " x sex " in summary["hr_summary"]["title"]


def test_build_report_plots_degrade_when_sex_uninformative(tmp_path: Path):
    run_id = "plots-unit-degrade"
    artifacts_root = tmp_path
    run_dir = artifacts_root / run_id
    analysis_dir = run_dir / "analysis_tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = pd.DataFrame(
        [
            {"record_id": "r1", "cohort_label": "A", "sex": "Unknown", "age_bin": "Unknown", "mean_hr": 70.0, "rr_std": 0.08},
            {"record_id": "r2", "cohort_label": "A", "sex": "Unknown", "age_bin": "Unknown", "mean_hr": 88.0, "rr_std": 0.12},
            {"record_id": "r3", "cohort_label": "B", "sex": "Unknown", "age_bin": "Unknown", "mean_hr": 95.0, "rr_std": 0.19},
            {"record_id": "r4", "cohort_label": "B", "sex": "Unknown", "age_bin": "Unknown", "mean_hr": 110.0, "rr_std": 0.15},
        ]
    )
    qc_df = pd.DataFrame(
        [
            {"record_id": "r1", "qc_pass": True, "qc_reasons": []},
            {"record_id": "r2", "qc_pass": True, "qc_reasons": []},
            {"record_id": "r3", "qc_pass": True, "qc_reasons": []},
            {"record_id": "r4", "qc_pass": True, "qc_reasons": []},
        ]
    )

    analysis_path = analysis_dir / "analysis_dataset.parquet"
    qc_path = run_dir / "ecg_qc.parquet"
    analysis_df.to_parquet(analysis_path, index=False)
    qc_df.to_parquet(qc_path, index=False)

    summary = build_report_plots(
        run_id=run_id,
        artifacts_root=artifacts_root,
        analysis_dataset_path=analysis_path,
        qc_path=qc_path,
        fail_top_n=3,
        feature_preferred=["rr_std", "qtc"],
    )

    assert summary["hr_summary"]["display_mode"] == "cohort_only"
    assert " x sex " not in summary["hr_summary"]["title"]



def test_build_report_plots_with_window_group(tmp_path: Path):
    run_id = "plots-window-group"
    artifacts_root = tmp_path
    run_dir = artifacts_root / run_id
    analysis_dir = run_dir / "analysis_tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = pd.DataFrame(
        [
            {"record_id": "r1", "window_group": "pre", "sex": "Unknown", "mean_hr": 70.0, "rr_std": 0.08},
            {"record_id": "r2", "window_group": "pre", "sex": "Unknown", "mean_hr": 74.0, "rr_std": 0.09},
            {"record_id": "r3", "window_group": "post", "sex": "Unknown", "mean_hr": 92.0, "rr_std": 0.19},
            {"record_id": "r4", "window_group": "post", "sex": "Unknown", "mean_hr": 96.0, "rr_std": 0.15},
        ]
    )
    qc_df = pd.DataFrame(
        [
            {"record_id": "r1", "qc_pass": True, "qc_reasons": []},
            {"record_id": "r2", "qc_pass": True, "qc_reasons": []},
            {"record_id": "r3", "qc_pass": True, "qc_reasons": []},
            {"record_id": "r4", "qc_pass": True, "qc_reasons": []},
        ]
    )

    analysis_path = analysis_dir / "analysis_dataset.parquet"
    qc_path = run_dir / "ecg_qc.parquet"
    analysis_df.to_parquet(analysis_path, index=False)
    qc_df.to_parquet(qc_path, index=False)

    summary = build_report_plots(
        run_id=run_id,
        artifacts_root=artifacts_root,
        analysis_dataset_path=analysis_path,
        qc_path=qc_path,
        fail_top_n=3,
        feature_preferred=["rr_std", "qtc"],
        group_col="window_group",
    )

    assert summary["group_col"] == "window_group"
    assert "window_group" in summary["hr_summary"]["title"]

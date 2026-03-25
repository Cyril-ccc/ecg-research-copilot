from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.generate_report import generate_report  # noqa: E402


def test_generate_report(tmp_path: Path):
    run_id = "report-unit"
    artifacts_root = tmp_path
    run_dir = artifacts_root / run_id
    analysis_dir = run_dir / "analysis_tables"
    plots_dir = run_dir / "plots"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = pd.DataFrame(
        [
            {
                "record_id": "r1",
                "cohort_label": "A",
                "sex": "F",
                "mean_hr": 70.0,
                "rr_std": 0.05,
                "feature_version": "v1.0",
                "qc_version": "qc_v1",
            },
            {
                "record_id": "r2",
                "cohort_label": "B",
                "sex": "M",
                "mean_hr": 95.0,
                "rr_std": 0.12,
                "feature_version": "v1.0",
                "qc_version": "qc_v1",
            },
        ]
    )
    feature_summary = pd.DataFrame(
        [
            {
                "feature_name": "mean_hr",
                "group_value": "cohort_label=A|sex=F|age_bin=Other",
                "group_n": 1,
                "n": 1,
                "mean": 70.0,
                "std": None,
                "p50": 70.0,
                "missing_rate": 0.0,
            },
            {
                "feature_name": "rr_std",
                "group_value": "cohort_label=B|sex=M|age_bin=Other",
                "group_n": 1,
                "n": 1,
                "mean": 0.12,
                "std": None,
                "p50": 0.12,
                "missing_rate": 0.0,
            },
        ]
    )
    group_compare = pd.DataFrame(
        [
            {
                "feature_name": "mean_hr",
                "group_a": "A",
                "group_b": "B",
                "n_a": 1,
                "n_b": 1,
                "diff_mean": -25.0,
                "diff_median": -25.0,
                "p_value": 0.04,
                "effect_size": -1.0,
            },
            {
                "feature_name": "rr_std",
                "group_a": "A",
                "group_b": "B",
                "n_a": 1,
                "n_b": 1,
                "diff_mean": -0.07,
                "diff_median": -0.07,
                "p_value": 0.01,
                "effect_size": -0.9,
            },
            {
                "feature_name": "rr_mean",
                "group_a": "A",
                "group_b": "B",
                "n_a": 1,
                "n_b": 1,
                "diff_mean": 0.2,
                "diff_median": 0.2,
                "p_value": 0.03,
                "effect_size": 0.7,
            },
        ]
    )

    analysis_df.to_parquet(analysis_dir / "analysis_dataset.parquet", index=False)
    feature_summary.to_parquet(analysis_dir / "feature_summary.parquet", index=False)
    group_compare.to_parquet(analysis_dir / "group_compare.parquet", index=False)
    (analysis_dir / "analysis_dataset_summary.json").write_text("{}", encoding="utf-8")
    (analysis_dir / "analysis_tables_summary.json").write_text("{}", encoding="utf-8")

    for name in [
        "qc_pass_rate.png",
        "hr_distribution_by_group.png",
        "feature_boxplot_qtc_or_rrstd.png",
    ]:
        (plots_dir / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    (plots_dir / "report_plots_summary.json").write_text(
        json.dumps(
            {
                "plots": [
                    "plots/qc_pass_rate.png",
                    "plots/hr_distribution_by_group.png",
                    "plots/feature_boxplot_qtc_or_rrstd.png",
                ],
                "qc_summary": {
                    "total_n": 2,
                    "pass_n": 1,
                    "fail_n": 1,
                    "top_fail_reasons": [{"reason": "flatline", "n": 1}],
                },
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "params.json").write_text(
        json.dumps({"question": "unit question", "params": {"k_threshold": 5.5}}),
        encoding="utf-8",
    )

    report_path, metadata_path, metadata = generate_report(
        run_id=run_id,
        artifacts_root=artifacts_root,
    )

    assert report_path.exists()
    assert metadata_path.exists()

    report_txt = report_path.read_text(encoding="utf-8")
    assert "## Results" in report_txt
    assert "group_compare.parquet: row" in report_txt
    assert "feature_summary.parquet: row" in report_txt
    assert "diff_mean=-25" in report_txt

    meta_txt = json.loads(metadata_path.read_text(encoding="utf-8"))
    for k in [
        "run_id",
        "question",
        "params",
        "qc_version",
        "feature_version",
        "git_commit",
        "created_at",
        "duration_seconds",
    ]:
        assert k in meta_txt
    assert metadata["pipeline_name"] == "report_pipeline"

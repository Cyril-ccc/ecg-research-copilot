from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.demo_report import run_demo_report  # noqa: E402


def test_demo_report_e2e_smoke(tmp_path: Path):
    manifest_path = REPO_ROOT / "storage" / "ecg_manifest.parquet"
    data_dir = (
        REPO_ROOT
        / "data"
        / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
    )
    if not manifest_path.exists() or not data_dir.exists():
        pytest.skip("demo ECG data/manifest not found")

    run_id = "pytest-demo-report-e2e"
    summary = run_demo_report(
        run_id=run_id,
        artifacts_root=tmp_path,
        data_dir=data_dir,
        global_manifest_path=manifest_path,
        sample_n=8,
        question="pytest demo report e2e",
    )

    run_dir = tmp_path / run_id
    expected_outputs = [
        run_dir / "cohort.parquet",
        run_dir / "cohort_summary.json",
        run_dir / "ecg_qc.parquet",
        run_dir / "ecg_features.parquet",
        run_dir / "ecg_map.parquet",
        run_dir / "analysis_tables" / "analysis_dataset.parquet",
        run_dir / "analysis_tables" / "feature_summary.parquet",
        run_dir / "analysis_tables" / "group_compare.parquet",
        run_dir / "plots" / "report_plots_summary.json",
        run_dir / "report.md",
        run_dir / "run_metadata.json",
        run_dir / "demo_report_summary.json",
    ]
    for path in expected_outputs:
        assert path.exists(), f"missing expected artifact: {path}"

    report_txt = (run_dir / "report.md").read_text(encoding="utf-8")
    for section in ["## Methods", "## Results", "## Limitations"]:
        assert section in report_txt

    assert summary["run_id"] == run_id
    assert int(summary["sample_n"]) == 8

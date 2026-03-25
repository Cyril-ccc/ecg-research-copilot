from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.build_analysis_tables import build_analysis_tables  # noqa: E402


def test_build_analysis_tables(tmp_path: Path):
    run_id = "analysis-unit"
    artifacts_root = tmp_path
    analysis_dir = artifacts_root / run_id / "analysis_tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [
            {"record_id": "r1", "cohort_label": "A", "sex": "F", "age_bin": "<40", "mean_hr": 60.0, "rr_std": 0.10},
            {"record_id": "r2", "cohort_label": "A", "sex": "F", "age_bin": "<40", "mean_hr": 80.0, "rr_std": 0.20},
            {"record_id": "r3", "cohort_label": "A", "sex": "F", "age_bin": "<40", "mean_hr": None, "rr_std": 0.15},
            {"record_id": "r4", "cohort_label": "B", "sex": "M", "age_bin": ">60", "mean_hr": 100.0, "rr_std": 0.05},
            {"record_id": "r5", "cohort_label": "B", "sex": "M", "age_bin": ">60", "mean_hr": 110.0, "rr_std": None},
            {"record_id": "r6", "cohort_label": "B", "sex": "M", "age_bin": ">60", "mean_hr": 90.0, "rr_std": 0.07},
        ]
    )
    analysis_dataset_path = analysis_dir / "analysis_dataset.parquet"
    df.to_parquet(analysis_dataset_path, index=False)

    feature_summary_path, group_compare_path, summary = build_analysis_tables(
        run_id=run_id,
        artifacts_root=artifacts_root,
        analysis_dataset_path=analysis_dataset_path,
        group_cols=["cohort_label", "sex", "age_bin"],
        compare_by="cohort_label",
        compare_features=["mean_hr", "rr_std"],
    )

    assert feature_summary_path.exists()
    assert group_compare_path.exists()
    assert summary["group_cols"] == ["cohort_label", "sex", "age_bin"]
    assert summary["compare_features"] == ["mean_hr", "rr_std"]

    feature_summary = pd.read_parquet(feature_summary_path)
    group_compare = pd.read_parquet(group_compare_path)

    row = feature_summary[
        (feature_summary["feature_name"] == "mean_hr")
        & (feature_summary["cohort_label"] == "A")
        & (feature_summary["sex"] == "F")
        & (feature_summary["age_bin"] == "<40")
    ].iloc[0]
    assert int(row["group_n"]) == 3
    assert int(row["n"]) == 2
    assert float(row["mean"]) == pytest.approx(70.0, abs=1e-9)
    assert float(row["p50"]) == pytest.approx(70.0, abs=1e-9)
    assert float(row["missing_rate"]) == pytest.approx(1 / 3, abs=1e-9)

    comp_row = group_compare[
        (group_compare["feature_name"] == "mean_hr")
        & (group_compare["group_a"] == "A")
        & (group_compare["group_b"] == "B")
    ].iloc[0]
    assert int(comp_row["n_a"]) == 2
    assert int(comp_row["n_b"]) == 3
    assert float(comp_row["mean_a"]) == pytest.approx(70.0, abs=1e-9)
    assert float(comp_row["mean_b"]) == pytest.approx(100.0, abs=1e-9)
    assert float(comp_row["diff_mean"]) == pytest.approx(-30.0, abs=1e-9)
    assert comp_row["test_method"] == "mannwhitney_u"
    assert comp_row["effect_name"] == "cohen_d"

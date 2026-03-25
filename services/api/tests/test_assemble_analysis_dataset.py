from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.assemble_analysis_dataset import assemble_analysis_dataset  # noqa: E402


def test_assemble_analysis_dataset_with_covariates(tmp_path: Path):
    run_id = "assemble-unit"
    artifacts_root = tmp_path
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cohort_df = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 11,
                "index_time": "2180-01-01 00:00:00",
                "cohort_label": "A",
            },
            {
                "subject_id": 2,
                "hadm_id": 22,
                "index_time": "2180-02-01 00:00:00",
                "cohort_label": "A",
            },
            {
                "subject_id": 2,
                "hadm_id": 23,
                "index_time": "2180-03-01 00:00:00",
                "cohort_label": "A",
            },
        ]
    )
    features_df = pd.DataFrame(
        [
            {"record_id": "r1", "mean_hr": 70.0, "rr_mean": 0.85, "rr_std": 0.05},
            {"record_id": "r2", "mean_hr": 90.0, "rr_mean": 0.67, "rr_std": 0.04},
            {"record_id": "r3", "mean_hr": 60.0, "rr_mean": 1.00, "rr_std": 0.03},
        ]
    )
    ecg_map_df = pd.DataFrame(
        [
            {"record_id": "r1", "subject_id": 1, "source": "mimic_ecg"},
            {"record_id": "r2", "subject_id": 2, "source": "ptbxl"},
            {"record_id": "r3", "subject_id": 3, "source": "mimic_ecg"},
        ]
    )
    cov_df = pd.DataFrame(
        [
            {"subject_id": 1, "age": 35, "sex": "F"},
            {"subject_id": 2, "age": 55, "sex": "M"},
            {"subject_id": 3, "age": 72, "sex": "M"},
        ]
    )

    cohort_path = run_dir / "cohort.parquet"
    features_path = run_dir / "ecg_features.parquet"
    ecg_map_path = run_dir / "ecg_map.parquet"
    cov_path = run_dir / "covariates.parquet"
    global_manifest_path = tmp_path / "global_manifest.parquet"

    cohort_df.to_parquet(cohort_path, index=False)
    features_df.to_parquet(features_path, index=False)
    ecg_map_df.to_parquet(ecg_map_path, index=False)
    cov_df.to_parquet(cov_path, index=False)
    ecg_map_df[["record_id", "subject_id", "source"]].to_parquet(global_manifest_path, index=False)

    out_path, summary = assemble_analysis_dataset(
        run_id=run_id,
        artifacts_root=artifacts_root,
        cohort_path=cohort_path,
        features_path=features_path,
        ecg_map_path=ecg_map_path,
        global_manifest_path=global_manifest_path,
        covariates_path=cov_path,
        age_bin_mode="fixed",
    )

    assert out_path.exists()
    out_df = pd.read_parquet(out_path)
    assert len(out_df) == len(features_df)

    required_cols = {"record_id", "subject_id", "sex", "age", "age_bin", "dataset_source"}
    assert required_cols.issubset(out_df.columns)

    age_bin_by_record = dict(
        zip(
            out_df["record_id"].astype(str),
            out_df["age_bin"].astype(str),
            strict=False,
        )
    )
    assert age_bin_by_record["r1"] == "<40"
    assert age_bin_by_record["r2"] == "40-60"
    assert age_bin_by_record["r3"] == ">60"

    source_by_record = dict(
        zip(
            out_df["record_id"].astype(str),
            out_df["dataset_source"].astype(str),
            strict=False,
        )
    )
    assert source_by_record["r1"] == "mimic_ecg"
    assert source_by_record["r2"] == "ptbxl"

    assert summary["rows"] == 3
    assert summary["rows_match_features"] is True
    assert summary["sex_counts"]["M"] == 2
    assert summary["sex_counts"]["F"] == 1



def test_assemble_analysis_dataset_with_window_map(tmp_path: Path):
    run_id = "assemble-window"
    artifacts_root = tmp_path
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cohort_df = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 11,
                "index_time": "2180-01-01 00:00:00",
                "cohort_label": "medication_exposure",
            }
        ]
    )
    features_df = pd.DataFrame(
        [
            {"record_id": "r_pre", "mean_hr": 70.0, "rr_mean": 0.85, "rr_std": 0.05},
            {"record_id": "r_post", "mean_hr": 90.0, "rr_mean": 0.67, "rr_std": 0.04},
        ]
    )
    ecg_map_df = pd.DataFrame(
        [
            {"record_id": "r_pre", "subject_id": 1, "source": "mimic_ecg"},
            {"record_id": "r_post", "subject_id": 1, "source": "mimic_ecg"},
        ]
    )
    window_map_df = pd.DataFrame(
        [
            {
                "record_id": "r_pre",
                "subject_id": 1,
                "hadm_id": 11,
                "index_time": "2180-01-01 00:00:00",
                "ecg_time": "2179-12-31 23:00:00",
                "delta_hours": -1.0,
                "window_group": "pre",
            },
            {
                "record_id": "r_post",
                "subject_id": 1,
                "hadm_id": 11,
                "index_time": "2180-01-01 00:00:00",
                "ecg_time": "2180-01-01 01:00:00",
                "delta_hours": 1.0,
                "window_group": "post",
            },
        ]
    )

    cohort_path = run_dir / "cohort.parquet"
    features_path = run_dir / "ecg_features.parquet"
    ecg_map_path = run_dir / "ecg_map.parquet"
    window_map_path = run_dir / "ecg_window_map.parquet"
    global_manifest_path = tmp_path / "global_manifest.parquet"

    cohort_df.to_parquet(cohort_path, index=False)
    features_df.to_parquet(features_path, index=False)
    ecg_map_df.to_parquet(ecg_map_path, index=False)
    window_map_df.to_parquet(window_map_path, index=False)
    ecg_map_df[["record_id", "subject_id", "source"]].to_parquet(global_manifest_path, index=False)

    out_path, summary = assemble_analysis_dataset(
        run_id=run_id,
        artifacts_root=artifacts_root,
        cohort_path=cohort_path,
        features_path=features_path,
        ecg_map_path=ecg_map_path,
        global_manifest_path=global_manifest_path,
        covariates_path=None,
        age_bin_mode="fixed",
        window_map_path=window_map_path,
    )

    out_df = pd.read_parquet(out_path)
    assert "window_group" in out_df.columns
    assert set(out_df["window_group"].astype(str).tolist()) == {"pre", "post"}
    assert summary["window_group_counts"]["pre"] == 1
    assert summary["window_group_counts"]["post"] == 1

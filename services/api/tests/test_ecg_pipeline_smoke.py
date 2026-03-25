"""
ECG pipeline smoke test on local demo data (20-50 records).

Checks:
- artifacts are generated
- row counts are consistent
- required feature fields are present
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.ecg_features import run_features  # noqa: E402
from pipelines.ecg_plots import generate_qc_feature_plots  # noqa: E402
from pipelines.ecg_qc import run_qc  # noqa: E402


def test_ecg_qc_features_smoke_demo_data(tmp_path: Path):
    manifest_path = REPO_ROOT / "storage" / "ecg_manifest.parquet"
    data_dir = (
        REPO_ROOT
        / "data"
        / "mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1"
    )
    if not manifest_path.exists() or not data_dir.exists():
        pytest.skip("demo ECG data/manifest not found")

    sample_n = 30  # requested range: 20-50
    all_ids = pd.read_parquet(manifest_path, columns=["record_id"])["record_id"].astype(str).tolist()
    record_ids = all_ids[:sample_n]
    if len(record_ids) < 20:
        pytest.skip("not enough demo ECG records for smoke test")

    run_id = "pytest-ecg-smoke"
    qc_path, qc_summary = run_qc(
        run_id=run_id,
        data_dir=data_dir,
        record_ids=record_ids,
        global_manifest_path=manifest_path,
        artifacts_root=tmp_path,
    )
    assert qc_path.exists()
    assert qc_summary["total"] == sample_n

    features_path, features_summary = run_features(
        run_id=run_id,
        data_dir=data_dir,
        global_manifest_path=manifest_path,
        artifacts_root=tmp_path,
        qc_path=qc_path,
        record_ids=record_ids,
        limit=0,
        feature_version="v1.0-test",
        thresholds={
            "hr_min_bpm": 1.0,
            "hr_max_bpm": 300.0,
            "rr_std_max_sec": 10.0,
            "p2p_mean_min_mv": 0.0,
            "p2p_mean_max_mv": 100.0,
            "min_detected_peaks": 2,
        },
    )
    assert features_path.exists()

    qc_df = pd.read_parquet(qc_path)
    features_df = pd.read_parquet(features_path)
    ecg_map_path = tmp_path / run_id / "ecg_map.parquet"
    assert ecg_map_path.exists()
    ecg_map_df = pd.read_parquet(ecg_map_path)
    assert {"record_id", "subject_id"}.issubset(set(ecg_map_df.columns))
    assert ecg_map_df["subject_id"].notna().all()
    assert set(qc_df["record_id"].astype(str)).issubset(set(ecg_map_df["record_id"].astype(str)))

    qc_pass_count = int(qc_df["qc_pass"].sum())
    assert len(qc_df) == sample_n
    assert len(features_df) == qc_pass_count
    assert features_summary["output_feature_count"] == len(features_df)
    assert features_summary["input_qc_pass_count"] == qc_pass_count

    required_cols = {
        "record_id",
        "mean_hr",
        "rr_mean",
        "rr_std",
        "lead_amplitude_p2p_mean",
        "lead_amplitude_p2p_std",
        "feature_version",
        "qc_version",
        "code_commit",
    }
    assert required_cols.issubset(set(features_df.columns))

    plots_summary = generate_qc_feature_plots(
        run_id=run_id,
        artifacts_root=tmp_path,
        data_dir=data_dir,
        global_manifest_path=manifest_path,
    )
    plots_dir = tmp_path / run_id / "plots"
    assert plots_dir.exists()
    assert (plots_dir / "plots_summary.json").exists()
    assert (plots_dir / "hr_distribution.png").exists()

    loaded_summary = json.loads((plots_dir / "plots_summary.json").read_text("utf-8"))
    assert loaded_summary["run_id"] == run_id
    assert loaded_summary["hr_distribution_plot"] == "plots/hr_distribution.png"
    assert loaded_summary["actual_qc_samples"]["pass"] >= 1
    assert plots_summary["run_id"] == run_id

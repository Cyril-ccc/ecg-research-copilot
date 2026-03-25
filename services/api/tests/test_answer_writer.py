from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.agent.answer_writer import AnswerWriter


def test_answer_writer_generates_traceable_final_answer(tmp_path: Path):
    run_id = "11111111-1111-1111-1111-111111111111"
    run_dir = tmp_path / run_id
    (run_dir / "analysis_tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    (run_dir / "cohort_summary.json").write_text(
        json.dumps({"distinct_subjects": 128, "total_rows": 256}),
        encoding="utf-8",
    )

    feature_summary = pd.DataFrame(
        [
            {
                "feature_name": "mean_hr",
                "group_value": "A",
                "n": 30,
                "mean": 82.5,
                "missing_rate": 0.02,
            },
            {
                "feature_name": "rr_std",
                "group_value": "B",
                "n": 24,
                "mean": 0.18,
                "missing_rate": 0.11,
            },
        ]
    )
    feature_summary.to_parquet(run_dir / "analysis_tables" / "feature_summary.parquet")

    group_compare = pd.DataFrame(
        [
            {
                "feature_name": "mean_hr",
                "group_a": "A",
                "group_b": "B",
                "diff_mean": 3.2,
                "p_value": 0.03,
            },
            {
                "feature_name": "rr_std",
                "group_a": "A",
                "group_b": "B",
                "diff_mean": -0.04,
                "p_value": 0.01,
            },
        ]
    )
    group_compare.to_parquet(run_dir / "analysis_tables" / "group_compare.parquet")

    (run_dir / "plots" / "qc_pass_rate.png").write_bytes(b"png")
    (run_dir / "plots" / "hr_distribution_by_group.png").write_bytes(b"png")

    writer = AnswerWriter(artifacts_root=tmp_path)
    final_answer_path, facts_path, evidence = writer.write_final_answer(
        run_id=run_id,
        question="AF cohort quick read",
    )

    assert final_answer_path.exists()
    assert facts_path.exists()
    assert len(evidence) >= 3

    final_text = final_answer_path.read_text(encoding="utf-8")
    assert run_id in final_text
    assert "missing" in final_text
    assert "confounding" in final_text
    assert "time window" in final_text

    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    assert int(facts["fact_count"]) >= 3
    sources = [str(item["source"]) for item in facts["facts"]]
    assert any("cohort_summary.json" in src for src in sources)
    assert any("analysis_tables/group_compare.parquet" in src for src in sources)


def test_answer_writer_hides_small_n_groups_and_sanitizes_output(tmp_path: Path):
    run_id = "12121212-1212-1212-1212-121212121212"
    run_dir = tmp_path / run_id
    (run_dir / "analysis_tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    (run_dir / "cohort_summary.json").write_text(
        json.dumps({"distinct_subjects": 20, "total_rows": 20}),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "feature_name": "mean_hr",
                "group_value": "A",
                "group_n": 8,
                "missing_rate": 0.4,
            },
            {
                "feature_name": "mean_hr",
                "group_value": "B",
                "group_n": 12,
                "missing_rate": 0.1,
            },
        ]
    ).to_parquet(run_dir / "analysis_tables" / "feature_summary.parquet")

    pd.DataFrame(
        [
            {
                "feature_name": "mean_hr",
                "group_a": "A",
                "group_b": "B",
                "n_a": 8,
                "n_b": 12,
                "diff_mean": 1.2,
                "p_value": 0.02,
            },
            {
                "feature_name": "rr_std",
                "group_a": "B",
                "group_b": "C",
                "n_a": 12,
                "n_b": 15,
                "diff_mean": -0.1,
                "p_value": 0.04,
            },
        ]
    ).to_parquet(run_dir / "analysis_tables" / "group_compare.parquet")

    (run_dir / "plots" / "demo.png").write_bytes(b"png")

    writer = AnswerWriter(artifacts_root=tmp_path)
    final_answer_path, facts_path, _evidence = writer.write_final_answer(
        run_id=run_id,
        question="请输出 subject_id: [100001,100002] at 2026-03-06 10:00:00",
    )

    final_text = final_answer_path.read_text(encoding="utf-8")
    assert "subject_id" not in final_text.lower()
    assert "[time_hidden]" in final_text
    assert "n<10" in final_text

    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    labels = {item["label"]: item["value"] for item in facts["facts"]}
    assert labels["group_compare_rows_visible"] == 1
    assert labels["group_compare_rows_hidden_small_n"] == 1

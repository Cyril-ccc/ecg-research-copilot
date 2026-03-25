from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.agent.answer_writer import AnswerWriter
from app.agent.plan_schema import PlanConstraints, PlanStep, ResearchPlan
from app.agent.runner import AgentRunner
from app.agent.tool_registry import (
    BuildCohortInput,
    BuildCohortOutput,
    ExtractEcgFeaturesInput,
    ExtractEcgFeaturesOutput,
    GenerateReportInput,
    GenerateReportOutput,
    PermissionLevel,
    ReadArtifactSummaryInput,
    ReadArtifactSummaryOutput,
    ToolRegistry,
    ToolSpec,
)


class _DummyRetriever:
    def retrieve(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return []


class _DummyPlanner:
    def __init__(self, plan: ResearchPlan) -> None:
        self._plan = plan

    def create_plan(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return self._plan


def _build_test_registry(artifacts_root: Path) -> ToolRegistry:
    registry = ToolRegistry()

    def _build_cohort(body: BuildCohortInput) -> dict[str, Any]:
        run_dir = artifacts_root / str(body.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        cohort_df = pd.DataFrame(
            [
                {
                    "subject_id": "1001",
                    "index_time": "2025-01-01T00:00:00Z",
                    "cohort_label": "A",
                    "age": 61,
                    "sex": "M",
                },
                {
                    "subject_id": "1002",
                    "index_time": "2025-01-02T00:00:00Z",
                    "cohort_label": "B",
                    "age": 59,
                    "sex": "F",
                },
            ]
        )
        cohort_df.to_parquet(run_dir / "cohort.parquet", index=False)
        (run_dir / "cohort_summary.json").write_text(
            json.dumps({"distinct_subjects": 2, "total_rows": 2}),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "template_name": body.template_name,
            "cohort_id": "cohort-1",
            "cohort_table": None,
            "sql": "select 1",
            "row_count": 2,
            "rows": [],
        }

    def _extract(body: ExtractEcgFeaturesInput) -> dict[str, Any]:
        run_dir = artifacts_root / body.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"record_id": "r1", "qc_pass": True, "qc_reasons": []},
                {"record_id": "r2", "qc_pass": True, "qc_reasons": []},
            ]
        ).to_parquet(run_dir / "ecg_qc.parquet", index=False)
        pd.DataFrame(
            [
                {"record_id": "r1", "mean_hr": 76.0, "rr_mean": 0.79, "rr_std": 0.11},
                {"record_id": "r2", "mean_hr": 84.0, "rr_mean": 0.72, "rr_std": 0.16},
            ]
        ).to_parquet(run_dir / "ecg_features.parquet", index=False)
        (run_dir / "ecg_qc_summary.json").write_text(
            json.dumps({"total": 2, "pass_count": 2, "pass_ratio": 1.0}),
            encoding="utf-8",
        )
        (run_dir / "ecg_features_summary.json").write_text(
            json.dumps({"output_feature_count": 2}),
            encoding="utf-8",
        )
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        (plots_dir / "qc_pass_rate.png").write_bytes(b"png")
        (plots_dir / "plots_summary.json").write_text(
            json.dumps({"plots": ["plots/qc_pass_rate.png"]}),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "run_id": body.run_id,
            "job_id": "job-extract",
            "queue_status": "SUCCEEDED",
            "queued_at": "2026-01-01T00:00:00Z",
        }

    def _generate(body: GenerateReportInput) -> dict[str, Any]:
        run_dir = artifacts_root / body.run_id
        analysis_dir = run_dir / "analysis_tables"
        plots_dir = run_dir / "plots"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {
                    "record_id": "r1",
                    "cohort_label": "A",
                    "sex": "M",
                    "age_bin": "40-60",
                    "mean_hr": 76.0,
                    "rr_std": 0.11,
                    "rr_mean": 0.79,
                },
                {
                    "record_id": "r2",
                    "cohort_label": "B",
                    "sex": "F",
                    "age_bin": "40-60",
                    "mean_hr": 84.0,
                    "rr_std": 0.16,
                    "rr_mean": 0.72,
                },
            ]
        ).to_parquet(analysis_dir / "analysis_dataset.parquet", index=False)

        pd.DataFrame(
            [
                {"feature_name": "mean_hr", "group_value": "A", "missing_rate": 0.01},
                {"feature_name": "mean_hr", "group_value": "B", "missing_rate": 0.04},
            ]
        ).to_parquet(analysis_dir / "feature_summary.parquet", index=False)

        pd.DataFrame(
            [
                {
                    "feature_name": "mean_hr",
                    "group_a": "A",
                    "group_b": "B",
                    "diff_mean": -8.0,
                    "p_value": 0.02,
                }
            ]
        ).to_parquet(analysis_dir / "group_compare.parquet", index=False)

        (plots_dir / "hr_distribution_by_group.png").write_bytes(b"png")
        (run_dir / "report.md").write_text("# report", encoding="utf-8")
        (run_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
        (run_dir / "report_task_result.json").write_text(
            json.dumps({"status": "REPORT_SUCCEEDED"}),
            encoding="utf-8",
        )
        return {
            "ok": True,
            "run_id": body.run_id,
            "job_id": "job-report",
            "queue_status": "SUCCEEDED",
            "queued_at": "2026-01-01T00:00:00Z",
        }

    def _read_summary(body: ReadArtifactSummaryInput) -> dict[str, Any]:
        payload = json.loads(
            (artifacts_root / body.run_id / body.artifact_name).read_text(encoding="utf-8")
        )
        return {
            "ok": True,
            "run_id": body.run_id,
            "artifact_name": body.artifact_name,
            "summary": payload,
        }

    registry.register(
        ToolSpec(
            name="build_cohort",
            input_schema=BuildCohortInput,
            output_schema=BuildCohortOutput,
            permission_level=PermissionLevel.GENERATE_ARTIFACTS,
            handler=_build_cohort,
            timeout_seconds=30.0,
        )
    )
    registry.register(
        ToolSpec(
            name="extract_ecg_features",
            input_schema=ExtractEcgFeaturesInput,
            output_schema=ExtractEcgFeaturesOutput,
            permission_level=PermissionLevel.GENERATE_ARTIFACTS,
            handler=_extract,
            timeout_seconds=30.0,
        )
    )
    registry.register(
        ToolSpec(
            name="generate_report",
            input_schema=GenerateReportInput,
            output_schema=GenerateReportOutput,
            permission_level=PermissionLevel.GENERATE_ARTIFACTS,
            handler=_generate,
            timeout_seconds=30.0,
        )
    )
    registry.register(
        ToolSpec(
            name="read_artifact_summary",
            input_schema=ReadArtifactSummaryInput,
            output_schema=ReadArtifactSummaryOutput,
            permission_level=PermissionLevel.READ_ONLY,
            handler=_read_summary,
            timeout_seconds=10.0,
        )
    )
    return registry


def test_agent_runner_writes_plan_trace_and_final_answer(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.agent.runner.get_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.agent.runner.insert_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.agent.runner.update_run_status", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("app.agent.runner.insert_audit", lambda *_args, **_kwargs: None)

    run_id = "22222222-2222-2222-2222-222222222222"
    registry = _build_test_registry(tmp_path)
    plan = ResearchPlan(
        goal="test plan",
        steps=[
            PlanStep(
                tool="build_cohort",
                args={
                    "template_name": "diagnosis_icd",
                    "params": {"icd_prefixes": ["I48"]},
                    "run_id": "$RUN_ID",
                    "limit": 100,
                },
            ),
            PlanStep(
                tool="extract_ecg_features",
                args={
                    "run_id": "$RUN_ID",
                    "record_ids": ["r1", "r2"],
                    "params": {"limit": 100},
                },
            ),
            PlanStep(
                tool="generate_report",
                args={
                    "run_id": "$RUN_ID",
                    "config": {"question": "test"},
                },
            ),
            PlanStep(
                tool="read_artifact_summary",
                args={"run_id": "$RUN_ID", "artifact_name": "cohort_summary.json"},
            ),
        ],
        constraints=PlanConstraints(max_records_per_run=100, no_raw_text_export=True),
    )

    runner = AgentRunner(
        artifacts_root=tmp_path,
        global_manifest_path=tmp_path / "ecg_manifest.parquet",
        registry=registry,
        planner=_DummyPlanner(plan),
        retriever=_DummyRetriever(),
        answer_writer=AnswerWriter(artifacts_root=tmp_path),
    )

    result = runner.run_question(
        question="AF test",
        run_id=run_id,
        constraints={"max_records_per_run": 100, "no_raw_text_export": True},
    )

    assert result.status == "SUCCEEDED"
    assert result.plan_path.exists()
    assert result.trace_path.exists()
    assert result.final_answer_path is not None
    assert result.final_answer_path.exists()
    assert result.facts_path is not None
    assert result.facts_path.exists()

    trace = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert trace["status"] == "SUCCEEDED"
    assert len(trace["steps"]) == 4
    assert all(step["status"] == "succeeded" for step in trace["steps"])

    facts = json.loads(result.facts_path.read_text(encoding="utf-8"))
    assert int(facts["fact_count"]) >= 3


def _make_runner_for_arg_resolution(tmp_path: Path) -> AgentRunner:
    plan = ResearchPlan(
        goal="arg-resolution",
        steps=[
            PlanStep(
                tool="build_cohort",
                args={
                    "template_name": "diagnosis_icd",
                    "params": {"icd_prefixes": ["I48"]},
                    "run_id": "$RUN_ID",
                    "limit": 10,
                },
            )
        ],
        constraints=PlanConstraints(max_records_per_run=100, no_raw_text_export=True),
    )
    return AgentRunner(
        artifacts_root=tmp_path,
        global_manifest_path=tmp_path / "ecg_manifest.parquet",
        registry=_build_test_registry(tmp_path),
        planner=_DummyPlanner(plan),
        retriever=_DummyRetriever(),
        answer_writer=AnswerWriter(artifacts_root=tmp_path),
    )


def test_build_cohort_arg_aliases_are_normalized(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)
    resolved = runner._resolve_build_cohort_args(
        args={
            "template": "electrolyte_hyperkalemia",
            "k_threshold": 5.5,
            "window_hours": 6,
            "params": {"window_hours": 24},
            "limit": 9999,
        },
        constraints=PlanConstraints(max_records_per_run=500, no_raw_text_export=True),
    )

    assert resolved["template_name"] == "electrolyte_hyperkalemia"
    assert "template" not in resolved
    assert resolved["limit"] == 500
    assert resolved["params"]["k_threshold"] == 5.5
    assert resolved["params"]["window_hours"] == 24
    assert resolved["params"]["charttime_start"] == "1900-01-01T00:00:00Z"
    assert resolved["params"]["charttime_end"] == "2300-01-01T00:00:00Z"


def test_extract_arg_aliases_are_normalized(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)
    resolved = runner._resolve_extract_args(
        args={
            "run_id": "22222222-2222-2222-2222-222222222222",
            "records": ["r1", "r2", "", "r1"],
            "limit": 700,
        },
        run_dir=tmp_path / "unused",
        constraints=PlanConstraints(max_records_per_run=50, no_raw_text_export=True),
    )

    assert resolved["record_ids"] == ["r1", "r2"]
    assert resolved["params"]["limit"] == 50
    assert resolved["params"]["max_records_per_run"] == 50
    assert "records" not in resolved
    assert "limit" not in resolved

def test_generate_report_args_auto_fill_run_id_and_config(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)
    resolved = runner._resolve_step_args(
        step_tool="generate_report",
        raw_args={"report_type": "distribution"},
        run_id="22222222-2222-2222-2222-222222222222",
        constraints=PlanConstraints(max_records_per_run=50, no_raw_text_export=True),
        run_dir=tmp_path / "unused",
    )

    assert resolved["run_id"] == "22222222-2222-2222-2222-222222222222"
    assert isinstance(resolved["config"], dict)
    assert isinstance(resolved["config"]["params"], dict)


def test_read_artifact_aliases_are_normalized(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)
    resolved = runner._resolve_step_args(
        step_tool="read_artifact_summary",
        raw_args={"artifact_name": "ecg_qc.parquet"},
        run_id="22222222-2222-2222-2222-222222222222",
        constraints=PlanConstraints(max_records_per_run=50, no_raw_text_export=True),
        run_dir=tmp_path / "unused",
    )

    assert resolved["run_id"] == "22222222-2222-2222-2222-222222222222"
    assert resolved["artifact_name"] == "ecg_qc_summary.json"

def test_medication_drug_keyword_is_canonicalized(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)
    resolved = runner._resolve_build_cohort_args(
        args={
            "template_name": "medication_exposure",
            "params": {
                "drug_keywords": ["维生素c"],
                "pre_hours": 12,
                "post_hours": 12,
            },
            "run_id": "22222222-2222-2222-2222-222222222222",
            "limit": 100,
        },
        constraints=PlanConstraints(max_records_per_run=100, no_raw_text_export=True),
    )

    assert resolved["params"]["drug_keywords"] == ["ascorbic acid"]


def test_validate_step_output_rejects_empty_cohort():
    err = AgentRunner._validate_step_output(
        step_tool="build_cohort",
        output={"row_count": 0},
    )
    assert isinstance(err, str)
    assert "build_cohort returned 0 rows" in err



def test_generate_report_args_include_medication_window_context(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)
    resolved = runner._resolve_step_args(
        step_tool="generate_report",
        raw_args={"config": {"question": "drug window"}},
        run_id="22222222-2222-2222-2222-222222222222",
        constraints=PlanConstraints(max_records_per_run=50, no_raw_text_export=True),
        run_dir=tmp_path / "unused",
        cohort_context={
            "template_name": "medication_exposure",
            "params": {
                "drug_keywords": ["ascorbic acid"],
                "pre_hours": 12,
                "post_hours": 12,
            },
        },
    )

    params = resolved["config"]["params"]
    assert params["template_name"] == "medication_exposure"
    assert params["pre_hours"] == 12
    assert params["post_hours"] == 12
    assert params["drug_keywords"] == ["ascorbic acid"]


def test_medication_window_auto_mapping_writes_window_artifacts(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)

    run_dir = tmp_path / "window-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "subject_id": "1001",
                "hadm_id": 11,
                "index_time": "2025-01-01T12:00:00Z",
                "cohort_label": "medication_exposure",
            }
        ]
    ).to_parquet(run_dir / "cohort.parquet", index=False)

    pd.DataFrame(
        [
            {
                "record_id": "r_pre",
                "subject_id": "1001",
                "ecg_time": "2025-01-01T06:00:00Z",
            },
            {
                "record_id": "r_post",
                "subject_id": "1001",
                "ecg_time": "2025-01-01T18:00:00Z",
            },
            {
                "record_id": "r_outside",
                "subject_id": "1001",
                "ecg_time": "2025-01-03T18:00:00Z",
            },
        ]
    ).to_parquet(tmp_path / "ecg_manifest.parquet", index=False)

    resolved = runner._resolve_extract_args(
        args={
            "run_id": "22222222-2222-2222-2222-222222222222",
            "record_ids": ["$AUTO_FROM_COHORT"],
            "params": {"limit": 50},
        },
        run_dir=run_dir,
        constraints=PlanConstraints(max_records_per_run=50, no_raw_text_export=True),
        cohort_context={
            "template_name": "medication_exposure",
            "params": {"pre_hours": 24, "post_hours": 24},
        },
    )

    assert set(resolved["record_ids"]) == {"r_pre", "r_post"}
    window_map_path = run_dir / "ecg_window_map.parquet"
    window_summary_path = run_dir / "ecg_window_summary.json"
    assert window_map_path.exists()
    assert window_summary_path.exists()

    mapped = pd.read_parquet(window_map_path)
    assert set(mapped["window_group"].astype(str).tolist()) == {"pre", "post"}


def test_medication_window_requires_manifest_ecg_time(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)

    run_dir = tmp_path / "window-run-missing-time"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "subject_id": "1001",
                "hadm_id": 11,
                "index_time": "2025-01-01T12:00:00Z",
                "cohort_label": "medication_exposure",
            }
        ]
    ).to_parquet(run_dir / "cohort.parquet", index=False)

    pd.DataFrame(
        [
            {"record_id": "r_pre", "subject_id": "1001"},
            {"record_id": "r_post", "subject_id": "1001"},
        ]
    ).to_parquet(tmp_path / "ecg_manifest.parquet", index=False)

    with pytest.raises(RuntimeError, match="ecg_time"):
        runner._resolve_extract_args(
            args={
                "run_id": "22222222-2222-2222-2222-222222222222",
                "record_ids": ["$AUTO_FROM_COHORT"],
                "params": {"limit": 50},
            },
            run_dir=run_dir,
            constraints=PlanConstraints(max_records_per_run=50, no_raw_text_export=True),
            cohort_context={
                "template_name": "medication_exposure",
                "params": {"pre_hours": 24, "post_hours": 24},
            },
        )


def test_read_artifact_summary_uses_json_summary_for_parquet_request(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)
    run_id = "22222222-2222-2222-2222-222222222222"
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"record_id": "r1", "qc_pass": True}]).to_parquet(run_dir / "ecg_qc.parquet", index=False)
    (run_dir / "ecg_qc_summary.json").write_text(
        json.dumps({"total": 1, "pass_count": 1, "pass_ratio": 1.0}),
        encoding="utf-8",
    )

    result = runner._handle_read_artifact_summary(
        ReadArtifactSummaryInput(run_id=run_id, artifact_name="ecg_qc.parquet")
    )

    assert result["ok"] is True
    assert result["artifact_name"] == "ecg_qc_summary.json"
    assert result["summary"]["pass_count"] == 1


def test_read_artifact_summary_falls_back_to_parquet_metadata(tmp_path: Path):
    runner = _make_runner_for_arg_resolution(tmp_path)
    run_id = "22222222-2222-2222-2222-222222222222"
    run_dir = tmp_path / run_id / "analysis_tables"
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"feature_name": "mean_hr", "diff_mean": -8.0, "p_value": 0.02}]
    ).to_parquet(run_dir / "group_compare.parquet", index=False)

    result = runner._handle_read_artifact_summary(
        ReadArtifactSummaryInput(
            run_id=run_id,
            artifact_name="analysis_tables/group_compare.parquet",
        )
    )

    assert result["ok"] is True
    assert result["artifact_name"] == "analysis_tables/group_compare.parquet"
    assert result["summary"]["format"] == "parquet"
    assert result["summary"]["rows"] == 1
    assert "feature_name" in result["summary"]["column_names"]

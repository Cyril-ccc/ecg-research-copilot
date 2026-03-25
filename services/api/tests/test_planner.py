import json

import pytest

from app.agent.plan_schema import ResearchPlan
from app.agent.planner import Planner
from app.agent.tool_registry import build_default_registry


def _build_registry():
    handlers = {
        "run_sql": lambda model: {
            "ok": True,
            "sql_sha256": "x",
            "rows": [],
            "row_count": 0,
        },
        "build_cohort": lambda model: {
            "ok": True,
            "template_name": model.template_name,
            "sql": "select 1",
            "row_count": 0,
            "rows": [],
        },
        "extract_ecg_features": lambda model: {
            "ok": True,
            "run_id": model.run_id,
            "job_id": "job",
            "queue_status": "QUEUED",
            "queued_at": "2026-01-01T00:00:00Z",
        },
        "generate_report": lambda model: {
            "ok": True,
            "run_id": model.run_id,
            "job_id": "job",
            "queue_status": "QUEUED",
            "queued_at": "2026-01-01T00:00:00Z",
        },
        "demo_report": lambda model: {
            "ok": True,
            "run_id": model.run_id or "00000000-0000-0000-0000-000000000000",
            "job_id": "job",
            "queue_status": "QUEUED",
            "queued_at": "2026-01-01T00:00:00Z",
        },
        "read_artifact_summary": lambda model: {
            "ok": True,
            "run_id": model.run_id,
            "artifact_name": model.artifact_name,
            "summary": {},
        },
    }
    return build_default_registry(handlers=handlers)


def _raise_llm(_prompt: str) -> str:
    raise RuntimeError("llm unavailable")


@pytest.mark.parametrize(
    "question,expected_template",
    [
        ("请评估入院6小时内高钾(K>=5.5)患者的ECG变化", "electrolyte_hyperkalemia"),
        (
            "入院 6 小时 K>5.5 的患者，ECG mean_hr 与 RR std 的总体分布是什么？",
            "electrolyte_hyperkalemia",
        ),
        ("AF患者住院期间的ECG风险分层", "diagnosis_icd"),
        ("比较胺碘酮用药前24小时和后24小时ECG特征", "medication_exposure"),
    ],
)
def test_planner_fallback_returns_valid_plan_for_typical_questions(
    question: str,
    expected_template: str,
):
    registry = _build_registry()
    planner = Planner(registry=registry, llm_generate=_raise_llm)

    plan = planner.create_plan(question=question, rag_snippets=["template notes from kb"])

    assert isinstance(plan, ResearchPlan)
    assert plan.steps[0].tool == "build_cohort"
    assert plan.steps[0].args["template_name"] == expected_template

    allowed_tools = {tool.name for tool in registry.list()}
    assert all(step.tool in allowed_tools for step in plan.steps)


def test_planner_retries_once_when_first_response_is_not_json():
    registry = _build_registry()
    calls = {"n": 0}

    def _llm(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not-json"
        return json.dumps(
            {
                "goal": "AF cohort ECG analysis",
                "steps": [
                    {
                        "tool": "build_cohort",
                        "args": {
                            "template_name": "diagnosis_icd",
                            "params": {"icd_prefixes": ["I48"], "window_hours": 24},
                            "run_id": "$RUN_ID",
                            "limit": 2000,
                        },
                    },
                    {
                        "tool": "generate_report",
                        "args": {
                            "run_id": "$RUN_ID",
                            "config": {"question": "AF cohort ECG analysis"},
                        },
                    },
                ],
                "constraints": {
                    "max_records_per_run": 2000,
                    "no_raw_text_export": True,
                },
            }
        )

    planner = Planner(registry=registry, llm_generate=_llm)
    plan = planner.create_plan(question="AF question", rag_snippets=["kb1"])

    assert calls["n"] == 2
    assert plan.goal == "AF cohort ECG analysis"
    allowed_tools = {tool.name for tool in registry.list()}
    assert all(step.tool in allowed_tools for step in plan.steps)


def test_planner_rejects_non_whitelisted_tools_and_falls_back():
    registry = _build_registry()

    def _llm(_prompt: str) -> str:
        return json.dumps(
            {
                "goal": "bad plan",
                "steps": [{"tool": "drop_all_tables", "args": {}}],
                "constraints": {
                    "max_records_per_run": 2000,
                    "no_raw_text_export": True,
                },
            }
        )

    planner = Planner(registry=registry, llm_generate=_llm)
    plan = planner.create_plan(question="高钾分析", rag_snippets=["kb1"])

    allowed_tools = {tool.name for tool in registry.list()}
    assert all(step.tool in allowed_tools for step in plan.steps)
    assert any(step.tool == "build_cohort" for step in plan.steps)


def test_planner_plan_repair_fills_required_fields_for_llm_plan():
    registry = _build_registry()

    def _llm(_prompt: str) -> str:
        return json.dumps(
            {
                "goal": "AF inpatient risk stratification",
                "steps": [
                    {
                        "tool": "build_cohort",
                        "args": {
                            "template": "diagnosis_icd",
                            "icd_prefixes": ["I48"],
                            "window_hours": 24,
                        },
                    },
                    {
                        "tool": "extract_ecg_features",
                        "args": {
                            "feature_names": ["mean_hr", "rr_std"],
                        },
                    },
                    {
                        "tool": "generate_report",
                        "args": {
                            "report_type": "risk_stratification",
                        },
                    },
                ],
                "constraints": {
                    "max_records_per_run": 500,
                    "no_raw_text_export": True,
                },
            }
        )

    planner = Planner(registry=registry, llm_generate=_llm)
    plan = planner.create_plan(question="AF患者住院期间的ECG风险分层", rag_snippets=["kb1"])

    steps = {step.tool: step.args for step in plan.steps}
    assert steps["build_cohort"]["template_name"] == "diagnosis_icd"
    assert steps["build_cohort"]["run_id"] == "$RUN_ID"
    assert steps["extract_ecg_features"]["run_id"] == "$RUN_ID"
    assert steps["extract_ecg_features"]["record_ids"] == ["$AUTO_FROM_COHORT"]
    assert int(steps["extract_ecg_features"]["params"]["limit"]) <= 500
    assert steps["generate_report"]["run_id"] == "$RUN_ID"
    assert "config" in steps["generate_report"]
    assert "read_artifact_summary" in steps


def test_planner_plan_repair_maps_template_tool_alias():
    registry = _build_registry()

    def _llm(_prompt: str) -> str:
        return json.dumps(
            {
                "goal": "drug pre post",
                "steps": [
                    {
                        "tool": "medication_exposure",
                        "args": {
                            "drug_keywords": ["胺碘酮"],
                            "pre_hours": 24,
                            "post_hours": 24,
                        },
                    }
                ],
                "constraints": {
                    "max_records_per_run": 200,
                    "no_raw_text_export": True,
                },
            }
        )

    planner = Planner(registry=registry, llm_generate=_llm)
    plan = planner.create_plan(question="比较胺碘酮用药前24小时和后24小时ECG特征", rag_snippets=["kb1"])

    assert plan.steps[0].tool == "build_cohort"
    assert plan.steps[0].args["template_name"] == "medication_exposure"
    assert plan.steps[0].args["run_id"] == "$RUN_ID"
    assert "drug_keywords" in plan.steps[0].args["params"]


def test_planner_plan_repair_maps_parquet_summary_artifacts():
    registry = _build_registry()

    def _llm(_prompt: str) -> str:
        return json.dumps(
            {
                "goal": "qc summary",
                "steps": [
                    {
                        "tool": "read_artifact_summary",
                        "args": {
                            "artifact_name": "ecg_qc.parquet",
                        },
                    }
                ],
                "constraints": {
                    "max_records_per_run": 200,
                    "no_raw_text_export": True,
                },
            }
        )

    planner = Planner(registry=registry, llm_generate=_llm)
    plan = planner.create_plan(question="查看 QC 概况", rag_snippets=["kb1"])

    step = next(step for step in plan.steps if step.tool == "read_artifact_summary")
    assert step.args["run_id"] == "$RUN_ID"
    assert step.args["artifact_name"] == "ecg_qc_summary.json"

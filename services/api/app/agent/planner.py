from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

import requests

from app.agent.plan_schema import PlanConstraints, PlanStep, ResearchPlan
from app.agent.tool_registry import ToolRegistry

LOGGER = logging.getLogger("api.agent.planner")

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
DEFAULT_FEATURE_VERSION = "v1.0-agent"
DEFAULT_ARTIFACT_SUMMARY = "cohort_summary.json"

WORKFLOW_TOOL_ORDER = [
    "build_cohort",
    "extract_ecg_features",
    "generate_report",
    "read_artifact_summary",
]

COHORT_TEMPLATE_NAMES = {
    "electrolyte_hyperkalemia",
    "diagnosis_icd",
    "medication_exposure",
}

MEDICATION_ALIAS_CANDIDATES: dict[str, list[str]] = {
    "amiodarone": ["amiodarone", "cordarone", "胺碘酮"],
    "metoprolol": ["metoprolol", "lopressor", "美托洛尔"],
    "furosemide": ["furosemide", "lasix", "呋塞米"],
    "potassium chloride": ["potassium chloride", "kcl", "氯化钾"],
    "ascorbic acid": ["ascorbic acid", "vitamin c", "vit c", "维生素c", "维生素 c"],
    "digoxin": ["digoxin", "lanoxin", "deslanoside", "cedilanid", "西地兰", "地高辛"],
}

DIAGNOSIS_ICD_RULES: list[tuple[list[str], list[str]]] = [
    (["nstemi"], ["I21.4", "410.7"]),
    (
        ["stemi", "ami", "myocardial infarction", "acute myocardial infarction", "心梗"],
        ["I21", "410"],
    ),
    (["af", "atrial fibrillation", "房颤", "心房颤动"], ["I48", "42731"]),
    (["heart failure", "chf", "心衰"], ["I50", "428"]),
    (["aki", "acute kidney injury", "急性肾损伤"], ["N17", "584"]),
]

READ_ARTIFACT_ALIASES = {
    "cohort": "cohort_summary.json",
    "cohort_summary": "cohort_summary.json",
    "cohort.parquet": "cohort_summary.json",
    "ecg_qc": "ecg_qc_summary.json",
    "ecg_qc.parquet": "ecg_qc_summary.json",
    "ecg_features": "ecg_features_summary.json",
    "ecg_features.parquet": "ecg_features_summary.json",
    "report_plots": "plots/report_plots_summary.json",
    "plots/report_plots_summary.json": "plots/report_plots_summary.json",
    "plots/plots_summary.json": "plots/plots_summary.json",
    "analysis_tables/analysis_dataset.parquet": "analysis_tables/analysis_dataset_summary.json",
    "analysis_tables/analysis_dataset_all.parquet": "analysis_tables/analysis_dataset_summary_all.json",
}


class Planner:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        request_timeout_sec: float = 30.0,
        llm_generate: Callable[[str], str] | None = None,
    ) -> None:
        self.registry = registry
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.ollama_model = ollama_model
        self.request_timeout_sec = max(1.0, float(request_timeout_sec))
        self._llm_generate = llm_generate

    def create_plan(
        self,
        *,
        question: str,
        rag_snippets: list[str],
        constraints: dict[str, Any] | PlanConstraints | None = None,
    ) -> ResearchPlan:
        question_txt = str(question or "").strip()
        if not question_txt:
            raise ValueError("question cannot be empty")

        constraints_obj = self._parse_constraints(constraints)
        allowed_tools = {spec.name for spec in self.registry.list()}
        if not allowed_tools:
            raise ValueError("tool registry is empty")

        base_prompt = self._build_prompt(
            question=question_txt,
            rag_snippets=rag_snippets,
            constraints=constraints_obj,
            allowed_tools=allowed_tools,
        )

        last_err = ""
        for attempt in range(2):
            try:
                prompt = (
                    base_prompt
                    if attempt == 0
                    else self._build_retry_prompt(base_prompt=base_prompt, error=last_err)
                )
                response_text = self._generate(prompt)
                plan_obj = self._parse_json_only(response_text)
                repaired_obj = self._repair_plan_payload(
                    plan_obj=plan_obj,
                    question=question_txt,
                    constraints=constraints_obj,
                    allowed_tools=allowed_tools,
                )
                plan = ResearchPlan.model_validate(repaired_obj)
                plan.ensure_tools_whitelisted(allowed_tools)
                return plan
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                LOGGER.warning("planner_attempt_failed attempt=%d err=%s", attempt + 1, exc)

        LOGGER.warning("planner_fallback_to_template_mode reason=%s", last_err)
        return self._fallback_plan(
            question=question_txt,
            constraints=constraints_obj,
            allowed_tools=allowed_tools,
        )

    def create_plan_json(
        self,
        *,
        question: str,
        rag_snippets: list[str],
        constraints: dict[str, Any] | PlanConstraints | None = None,
    ) -> dict[str, Any]:
        return self.create_plan(
            question=question,
            rag_snippets=rag_snippets,
            constraints=constraints,
        ).to_plan_json()

    @staticmethod
    def _parse_constraints(value: dict[str, Any] | PlanConstraints | None) -> PlanConstraints:
        if isinstance(value, PlanConstraints):
            return value
        if value is None:
            return PlanConstraints()
        return PlanConstraints.model_validate(value)

    def _generate(self, prompt: str) -> str:
        if self._llm_generate is not None:
            return str(self._llm_generate(prompt))

        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        response = requests.post(url, json=payload, timeout=self.request_timeout_sec)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or "response" not in body:
            raise ValueError("ollama response missing 'response' field")
        return str(body["response"])

    @staticmethod
    def _parse_json_only(text: str) -> dict[str, Any]:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\\s*```$", "", raw)

        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
            raise ValueError("planner output must be a JSON object")
        except Exception as exc:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("planner output is not valid JSON") from exc
            obj = json.loads(raw[start : end + 1])
            if not isinstance(obj, dict):
                raise ValueError("planner output must be a JSON object") from exc
            return obj

    @staticmethod
    def _build_prompt(
        *,
        question: str,
        rag_snippets: list[str],
        constraints: PlanConstraints,
        allowed_tools: set[str],
    ) -> str:
        snippets = [str(x).strip() for x in rag_snippets if str(x).strip()]
        kb_block = "\n".join(f"- {snippet}" for snippet in snippets) or "- (none)"
        tool_list = ", ".join(sorted(allowed_tools))
        max_records = min(5000, int(constraints.max_records_per_run))

        return f"""You are a clinical research planner.
Return ONLY a single JSON object. No markdown, no explanation.
Use ONLY the provided knowledge snippets and available tools.

Required JSON schema (no extra keys):
{{
  \"goal\": \"...\",
  \"steps\": [
    {{\"tool\": \"...\", \"args\": {{...}}}}
  ],
  \"constraints\": {{
    \"max_records_per_run\": <int>,
    \"no_raw_text_export\": <bool>
  }}
}}

Available tools (whitelist): [{tool_list}]
Do not invent tool names.
For each tool, required args are:
- build_cohort: template_name, params, run_id, limit
- extract_ecg_features: run_id, record_ids, params
- generate_report: run_id, config
- read_artifact_summary: run_id, artifact_name

Token rules:
- Use run_id as literal token \"$RUN_ID\".
- For extract_ecg_features.record_ids, use [\"$AUTO_FROM_COHORT\"] unless explicit IDs are provided.
- Keep max_records_per_run consistent with constraints and set per-step limits <= max_records_per_run.
- Use canonical field names: template_name, record_ids, config, artifact_name.

Output should follow this executable example:
{{
  \"goal\": \"Analyze AF inpatient ECG profile\",
  \"steps\": [
    {{\"tool\": \"build_cohort\", \"args\": {{\"template_name\": \"diagnosis_icd\", \"params\": {{\"icd_prefixes\": [\"I48\"], \"window_hours\": 24}}, \"run_id\": \"$RUN_ID\", \"limit\": {max_records}}}}},
    {{\"tool\": \"extract_ecg_features\", \"args\": {{\"run_id\": \"$RUN_ID\", \"record_ids\": [\"$AUTO_FROM_COHORT\"], \"params\": {{\"limit\": {max_records}, \"feature_version\": \"{DEFAULT_FEATURE_VERSION}\"}}}}}},
    {{\"tool\": \"generate_report\", \"args\": {{\"run_id\": \"$RUN_ID\", \"config\": {{\"question\": \"Analyze AF inpatient ECG profile\", \"params\": {{\"planner_mode\": \"llm\", \"template_name\": \"diagnosis_icd\"}}}}}}}},
    {{\"tool\": \"read_artifact_summary\", \"args\": {{\"run_id\": \"$RUN_ID\", \"artifact_name\": \"cohort_summary.json\"}}}}
  ],
  \"constraints\": {{\"max_records_per_run\": {constraints.max_records_per_run}, \"no_raw_text_export\": {str(constraints.no_raw_text_export).lower()}}}
}}

User question:
{question}

Knowledge snippets (only source):
{kb_block}
"""

    @staticmethod
    def _build_retry_prompt(*, base_prompt: str, error: str) -> str:
        return (
            f"{base_prompt}\n"
            "Your previous answer failed JSON/schema validation.\n"
            f"Validation error: {error}\n"
            "Regenerate and return ONLY corrected JSON.\n"
        )

    def _repair_plan_payload(
        self,
        *,
        plan_obj: dict[str, Any],
        question: str,
        constraints: PlanConstraints,
        allowed_tools: set[str],
    ) -> dict[str, Any]:
        if not isinstance(plan_obj, dict):
            raise ValueError("planner output must be a JSON object")

        goal = str(plan_obj.get("goal", "")).strip() or question
        safe_constraints = self._repair_constraints(
            raw_constraints=plan_obj.get("constraints"),
            fallback=constraints,
        )

        raw_steps = plan_obj.get("steps")
        steps_in = raw_steps if isinstance(raw_steps, list) else []
        repaired_steps = self._repair_steps(
            steps=steps_in,
            question=question,
            constraints=safe_constraints,
            allowed_tools=allowed_tools,
        )
        if not repaired_steps:
            raise ValueError("plan has no repairable steps")

        return {
            "goal": goal,
            "steps": repaired_steps,
            "constraints": safe_constraints.model_dump(mode="python"),
        }

    def _repair_constraints(
        self,
        *,
        raw_constraints: Any,
        fallback: PlanConstraints,
    ) -> PlanConstraints:
        if not isinstance(raw_constraints, dict):
            return fallback

        merged = {
            "max_records_per_run": raw_constraints.get(
                "max_records_per_run",
                fallback.max_records_per_run,
            ),
            "no_raw_text_export": raw_constraints.get(
                "no_raw_text_export",
                fallback.no_raw_text_export,
            ),
        }
        try:
            return PlanConstraints.model_validate(merged)
        except Exception:  # noqa: BLE001
            return fallback

    def _repair_steps(
        self,
        *,
        steps: list[Any],
        question: str,
        constraints: PlanConstraints,
        allowed_tools: set[str],
    ) -> list[dict[str, Any]]:
        intent_locked_template = self._detect_locked_template(question)
        fallback_template, fallback_params = self._fallback_template_params(question)
        max_records = min(5000, int(constraints.max_records_per_run))

        staged: dict[str, dict[str, Any]] = {}
        for raw_step in steps:
            if not isinstance(raw_step, dict):
                continue
            tool = str(raw_step.get("tool", "")).strip()
            args = raw_step.get("args") if isinstance(raw_step.get("args"), dict) else {}

            if tool in COHORT_TEMPLATE_NAMES:
                args = dict(args)
                args.setdefault("template_name", tool)
                tool = "build_cohort"

            if tool not in allowed_tools:
                continue

            repaired_args = self._repair_step_args(
                tool=tool,
                args=args,
                question=question,
                max_records=max_records,
                fallback_template=fallback_template,
                fallback_params=fallback_params,
                intent_locked_template=intent_locked_template,
            )
            if tool not in staged:
                staged[tool] = {"tool": tool, "args": repaired_args}

        if "build_cohort" in allowed_tools and "build_cohort" not in staged:
            staged["build_cohort"] = {
                "tool": "build_cohort",
                "args": self._default_build_args(
                    template_name=fallback_template,
                    params=fallback_params,
                    max_records=max_records,
                ),
            }

        template_name = fallback_template
        if "build_cohort" in staged:
            template_name = str(staged["build_cohort"]["args"].get("template_name", fallback_template))
        if intent_locked_template in COHORT_TEMPLATE_NAMES:
            template_name = intent_locked_template

        if "generate_report" in staged:
            staged["generate_report"]["args"] = self._repair_generate_report_args(
                args=staged["generate_report"].get("args", {}),
                question=question,
                template_name=template_name,
                force_template_name=True,
            )

        if (
            "extract_ecg_features" in allowed_tools
            and "build_cohort" in staged
            and "extract_ecg_features" not in staged
        ):
            staged["extract_ecg_features"] = {
                "tool": "extract_ecg_features",
                "args": self._default_extract_args(max_records=max_records),
            }

        if "generate_report" in allowed_tools and "build_cohort" in staged and "generate_report" not in staged:
            staged["generate_report"] = {
                "tool": "generate_report",
                "args": self._default_generate_report_args(
                    question=question,
                    template_name=template_name,
                ),
            }

        if "read_artifact_summary" in allowed_tools and "read_artifact_summary" not in staged:
            staged["read_artifact_summary"] = {
                "tool": "read_artifact_summary",
                "args": self._default_read_artifact_args(),
            }

        ordered: list[dict[str, Any]] = []
        for tool in WORKFLOW_TOOL_ORDER:
            if tool in staged:
                ordered.append(staged[tool])

        for tool, step in staged.items():
            if tool not in WORKFLOW_TOOL_ORDER:
                ordered.append(step)

        return ordered

    def _repair_step_args(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        question: str,
        max_records: int,
        fallback_template: str,
        fallback_params: dict[str, Any],
        intent_locked_template: str | None,
    ) -> dict[str, Any]:
        if tool == "build_cohort":
            return self._repair_build_cohort_args(
                args=args,
                max_records=max_records,
                fallback_template=fallback_template,
                fallback_params=fallback_params,
                intent_locked_template=intent_locked_template,
            )
        if tool == "extract_ecg_features":
            return self._repair_extract_args(args=args, max_records=max_records)
        if tool == "generate_report":
            return self._repair_generate_report_args(
                args=args,
                question=question,
                template_name=fallback_template,
            )
        if tool == "read_artifact_summary":
            return self._repair_read_artifact_args(args=args)
        return dict(args)

    def _repair_build_cohort_args(
        self,
        *,
        args: dict[str, Any],
        max_records: int,
        fallback_template: str,
        fallback_params: dict[str, Any],
        intent_locked_template: str | None,
    ) -> dict[str, Any]:
        out = dict(args)

        template_name = ""
        for key in ("template_name", "template", "cohort_template"):
            value = str(out.get(key, "")).strip()
            if value:
                template_name = value
                break
        if intent_locked_template in COHORT_TEMPLATE_NAMES:
            template_name = intent_locked_template
        elif template_name not in COHORT_TEMPLATE_NAMES:
            template_name = fallback_template

        raw_params = out.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}

        top_level_param_keys = [
            "window_hours",
            "k_threshold",
            "label_keyword",
            "icd_prefixes",
            "icd_codes",
            "icd_version",
            "drug_keywords",
            "drug_names",
            "source",
            "pre_hours",
            "post_hours",
            "charttime_start",
            "charttime_end",
            "admittime_start",
            "admittime_end",
            "starttime_start",
            "starttime_end",
        ]
        for key in top_level_param_keys:
            if key in out and key not in params:
                params[key] = out[key]

        for list_key in ("icd_prefixes", "icd_codes", "drug_keywords", "drug_names"):
            values = self._as_non_empty_string_list(params.get(list_key))
            if values:
                params[list_key] = values
            else:
                params.pop(list_key, None)

        if template_name == "diagnosis_icd":
            if "icd_prefixes" not in params and "icd_codes" not in params:
                fallback_prefixes = self._as_non_empty_string_list(fallback_params.get("icd_prefixes"))
                params["icd_prefixes"] = fallback_prefixes or ["I48"]
            params.setdefault("window_hours", 24)

        elif template_name == "electrolyte_hyperkalemia":
            params.setdefault("k_threshold", 5.5)
            params.setdefault("label_keyword", "potassium")
            params.setdefault("window_hours", 6)

        elif template_name == "medication_exposure":
            params.setdefault("source", "prescriptions")
            if "drug_keywords" not in params and "drug_names" not in params:
                fallback_drugs = self._as_non_empty_string_list(fallback_params.get("drug_keywords"))
                params["drug_keywords"] = fallback_drugs or ["amiodarone"]
            params.setdefault("pre_hours", 24)
            params.setdefault("post_hours", 24)

        run_id = str(out.get("run_id", "")).strip() or "$RUN_ID"
        limit = self._to_int(out.get("limit"), default=max_records)
        limit = max(1, min(5000, limit))

        return {
            "template_name": template_name,
            "params": params,
            "run_id": run_id,
            "limit": limit,
        }

    def _repair_extract_args(self, *, args: dict[str, Any], max_records: int) -> dict[str, Any]:
        out = dict(args)
        run_id = str(out.get("run_id", "")).strip() or "$RUN_ID"

        raw_record_ids = out.get("record_ids")
        if not isinstance(raw_record_ids, list):
            raw_record_ids = out.get("records")
        if not isinstance(raw_record_ids, list):
            raw_record_ids = out.get("record_id_list")

        record_ids = self._as_non_empty_string_list(raw_record_ids)
        if not record_ids:
            record_ids = ["$AUTO_FROM_COHORT"]

        raw_params = out.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}

        if "limit" in out and "limit" not in params:
            params["limit"] = out.get("limit")

        for key in ("feature_names", "feature_types", "features"):
            values = self._as_non_empty_string_list(out.get(key))
            if values and "requested_features" not in params:
                params["requested_features"] = values

        feature_set = str(out.get("feature_set", "")).strip().lower()
        if feature_set and "requested_features" not in params:
            if feature_set == "qt_qtc":
                params["requested_features"] = ["qt_ms", "qtc_ms", "rr_mean", "rr_std"]
            else:
                params["requested_features"] = [feature_set]

        limit = self._to_int(params.get("limit"), default=max_records)
        limit = max(1, min(max_records, limit))
        params["limit"] = limit
        params.setdefault("feature_version", DEFAULT_FEATURE_VERSION)

        return {
            "run_id": run_id,
            "record_ids": record_ids,
            "params": params,
        }

    def _repair_generate_report_args(
        self,
        *,
        args: dict[str, Any],
        question: str,
        template_name: str,
        force_template_name: bool = False,
    ) -> dict[str, Any]:
        out = dict(args)
        run_id = str(out.get("run_id", "")).strip() or "$RUN_ID"

        raw_config = out.get("config")
        config = dict(raw_config) if isinstance(raw_config, dict) else {}
        config.setdefault("question", question)

        raw_params = config.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        params.setdefault("planner_mode", "plan_repair")
        if template_name:
            if force_template_name:
                params["template_name"] = template_name
            else:
                params.setdefault("template_name", template_name)

        if str(params.get("template_name", "")).strip() != "medication_exposure":
            for key in (
                "source",
                "drug_names",
                "drug_keywords",
                "pre_hours",
                "post_hours",
                "starttime_start",
                "starttime_end",
            ):
                params.pop(key, None)

        for key, value in out.items():
            if key in {"run_id", "config"}:
                continue
            params.setdefault(key, value)

        config["params"] = params
        return {
            "run_id": run_id,
            "config": config,
        }

    def _repair_read_artifact_args(self, *, args: dict[str, Any]) -> dict[str, Any]:
        out = dict(args)
        run_id = str(out.get("run_id", "")).strip() or "$RUN_ID"

        artifact_name = str(out.get("artifact_name", "")).strip().replace("\\", "/")
        if not artifact_name:
            artifact_name = str(out.get("artifact", "")).strip().replace("\\", "/")
        if not artifact_name:
            artifact_type = str(out.get("artifact_type", "")).strip().lower()
            artifact_name = READ_ARTIFACT_ALIASES.get(artifact_type, "")
        if artifact_name:
            artifact_name = READ_ARTIFACT_ALIASES.get(artifact_name.lower(), artifact_name)
        if not artifact_name:
            artifact_name = DEFAULT_ARTIFACT_SUMMARY

        return {
            "run_id": run_id,
            "artifact_name": artifact_name,
        }
    def _default_build_args(
        self,
        *,
        template_name: str,
        params: dict[str, Any],
        max_records: int,
    ) -> dict[str, Any]:
        return {
            "template_name": template_name,
            "params": dict(params),
            "run_id": "$RUN_ID",
            "limit": max_records,
        }

    def _default_extract_args(self, *, max_records: int) -> dict[str, Any]:
        return {
            "run_id": "$RUN_ID",
            "record_ids": ["$AUTO_FROM_COHORT"],
            "params": {
                "limit": max_records,
                "feature_version": DEFAULT_FEATURE_VERSION,
            },
        }

    @staticmethod
    def _default_generate_report_args(*, question: str, template_name: str) -> dict[str, Any]:
        return {
            "run_id": "$RUN_ID",
            "config": {
                "question": question,
                "params": {
                    "planner_mode": "plan_repair",
                    "template_name": template_name,
                },
            },
        }

    @staticmethod
    def _default_read_artifact_args() -> dict[str, Any]:
        return {
            "run_id": "$RUN_ID",
            "artifact_name": DEFAULT_ARTIFACT_SUMMARY,
        }

    @staticmethod
    def _to_int(value: Any, *, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _as_non_empty_string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            txt = value.strip()
            return [txt] if txt else []
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            txt = str(item).strip()
            if txt:
                out.append(txt)
        return out

    def _fallback_plan(
        self,
        *,
        question: str,
        constraints: PlanConstraints,
        allowed_tools: set[str],
    ) -> ResearchPlan:
        template_name, params = self._fallback_template_params(question)

        steps: list[PlanStep] = []
        if "build_cohort" in allowed_tools:
            steps.append(
                PlanStep(
                    tool="build_cohort",
                    args={
                        "template_name": template_name,
                        "params": params,
                        "run_id": "$RUN_ID",
                        "limit": min(5000, constraints.max_records_per_run),
                    },
                )
            )
        if "extract_ecg_features" in allowed_tools:
            steps.append(
                PlanStep(
                    tool="extract_ecg_features",
                    args={
                        "run_id": "$RUN_ID",
                        "record_ids": ["$AUTO_FROM_COHORT"],
                        "params": {
                            "limit": min(5000, constraints.max_records_per_run),
                            "feature_version": "v1.0-agent",
                        },
                    },
                )
            )
        if "generate_report" in allowed_tools:
            steps.append(
                PlanStep(
                    tool="generate_report",
                    args={
                        "run_id": "$RUN_ID",
                        "config": {
                            "question": question,
                            "params": {
                                "planner_mode": "template_fallback",
                                "template_name": template_name,
                            },
                        },
                    },
                )
            )
        if "read_artifact_summary" in allowed_tools:
            steps.append(
                PlanStep(
                    tool="read_artifact_summary",
                    args={
                        "run_id": "$RUN_ID",
                        "artifact_name": "cohort_summary.json",
                    },
                )
            )

        if not steps:
            raise ValueError("no allowed tools available for fallback planning")

        plan = ResearchPlan(goal=question, steps=steps, constraints=constraints)
        plan.ensure_tools_whitelisted(allowed_tools)
        return plan

    def _fallback_template_params(self, question: str) -> tuple[str, dict[str, Any]]:
        q_raw = str(question)
        locked_template = self._detect_locked_template(question)

        if locked_template == "electrolyte_hyperkalemia":
            return (
                "electrolyte_hyperkalemia",
                {
                    "k_threshold": 5.5,
                    "label_keyword": "potassium",
                    "window_hours": 6,
                },
            )

        if locked_template == "diagnosis_icd":
            return (
                "diagnosis_icd",
                {
                    "icd_prefixes": self._extract_icd_prefixes(q_raw),
                    "window_hours": 24,
                },
            )

        if locked_template == "medication_exposure":
            drug = self._extract_drug_keyword(q_raw) or "amiodarone"
            pre_hours = self._extract_hours(q_raw, kind="pre", default=24)
            post_hours = self._extract_hours(q_raw, kind="post", default=24)
            return (
                "medication_exposure",
                {
                    "source": "prescriptions",
                    "drug_keywords": [drug],
                    "pre_hours": pre_hours,
                    "post_hours": post_hours,
                },
            )

        return (
            "diagnosis_icd",
            {
                "icd_prefixes": ["I48"],
                "window_hours": 24,
            },
        )

    def _detect_locked_template(self, question: str) -> str | None:
        q_raw = str(question or "")
        q_lower = q_raw.lower()

        has_hyperkalemia = self._contains_any(q_lower, ["hyperkal", "high k", "k>=", "k >", "k>"])
        has_hyperkalemia = has_hyperkalemia or self._contains_any(q_raw, ["高钾", "高血钾"])
        if has_hyperkalemia:
            return "electrolyte_hyperkalemia"

        has_known_drug = bool(self._extract_drug_keyword(q_raw))
        has_medication_keywords = self._contains_any(
            q_lower,
            ["medication", "drug", "pre", "post", "before", "after", "exposure", "prescription"],
        )
        has_medication_keywords = has_medication_keywords or self._contains_any(
            q_raw,
            ["用药", "药物", "前后", "暴露", "给药", "处方"],
        )

        has_diagnosis_keywords = self._contains_any(
            q_lower,
            [
                "diagnosis",
                "icd",
                "atrial fibrillation",
                "stemi",
                "nstemi",
                "heart failure",
                "acute kidney injury",
                "risk stratification",
            ],
        )
        has_diagnosis_keywords = has_diagnosis_keywords or self._contains_any(
            q_raw,
            ["诊断", "房颤", "心房颤动", "心梗", "心衰", "急性肾损伤", "风险分层"],
        )
        if re.search(r"\baf\b", q_lower, flags=re.IGNORECASE):
            has_diagnosis_keywords = True

        if has_medication_keywords and (has_known_drug or not has_diagnosis_keywords):
            return "medication_exposure"
        if has_diagnosis_keywords:
            return "diagnosis_icd"
        return None

    @staticmethod
    def _contains_any(text: str, needles: list[str]) -> bool:
        return any(n in text for n in needles)

    @staticmethod
    def _extract_icd_prefixes(question: str) -> list[str]:
        q_raw = str(question or "")
        q_lower = q_raw.lower()
        for aliases, icd_prefixes in DIAGNOSIS_ICD_RULES:
            for alias in aliases:
                if alias in q_lower or alias in q_raw:
                    if alias == "af" and re.search(r"\baf\b", q_lower, flags=re.IGNORECASE) is None:
                        continue
                    return list(icd_prefixes)
        return ["I48"]

    @staticmethod
    def _extract_drug_keyword(question: str) -> str:
        q_lower = question.lower()
        for canonical, aliases in MEDICATION_ALIAS_CANDIDATES.items():
            for alias in aliases:
                if alias.lower() in q_lower or alias in question:
                    return canonical
        return ""

    @staticmethod
    def _extract_hours(question: str, *, kind: str, default: int) -> int:
        text = question.lower()
        patterns = {
            "pre": [
                r"pre\\s*[-_:]?\\s*(\\d{1,3})",
                r"before\\s*(\\d{1,3})\\s*(?:h|hr|hrs|hour|hours)",
                r"前\\s*(\\d{1,3})\\s*小时",
                r"前\\s*(\\d{1,3})",
            ],
            "post": [
                r"post\\s*[-_:]?\\s*(\\d{1,3})",
                r"after\\s*(\\d{1,3})\\s*(?:h|hr|hrs|hour|hours)",
                r"后\\s*(\\d{1,3})\\s*小时",
                r"后\\s*(\\d{1,3})",
            ],
        }
        for pattern in patterns.get(kind, []):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = int(match.group(1))
                return max(1, min(168, value))
        return default


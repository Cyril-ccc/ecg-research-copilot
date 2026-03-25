from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from fastapi import HTTPException

from app.agent.answer_writer import AnswerWriter
from app.agent.knowledge_base import (
    ALLOWED_DOC_TYPES,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    KnowledgeBaseRetriever,
    OllamaEmbeddingClient,
    format_snippets_for_prompt,
)
from app.agent.plan_schema import PlanConstraints, ResearchPlan
from app.agent.planner import DEFAULT_OLLAMA_MODEL, Planner
from app.agent.tool_executor import ToolExecutor
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
from app.core.config import ARTIFACTS_DIR, DEMO_DATA_DIR, DEMO_MANIFEST_PATH
from app.db.models import get_run, insert_audit, insert_run, update_run_status

LOGGER = logging.getLogger("api.agent.runner")

MAX_COHORT_LIMIT = 5000
DEFAULT_COHORT_WINDOW_HOURS = 24
MAX_COHORT_WINDOW_HOURS = 168
DEFAULT_TIME_WINDOW_START = "1900-01-01T00:00:00Z"
DEFAULT_TIME_WINDOW_END = "2300-01-01T00:00:00Z"
MEDICATION_WINDOW_MAP_FILE = "ecg_window_map.parquet"
MEDICATION_WINDOW_SUMMARY_FILE = "ecg_window_summary.json"

ARTIFACT_SUMMARY_NAME_MAP: dict[str, str] = {
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

DRUG_KEYWORD_CANONICAL_MAP: dict[str, str] = {
    "胺碘酮": "amiodarone",
    "amiodarone": "amiodarone",
    "美托洛尔": "metoprolol",
    "metoprolol": "metoprolol",
    "呋塞米": "furosemide",
    "furosemide": "furosemide",
    "氯化钾": "potassium chloride",
    "kcl": "potassium chloride",
    "potassium chloride": "potassium chloride",
    "维生素c": "ascorbic acid",
    "维生素 c": "ascorbic acid",
    "vitamin c": "ascorbic acid",
    "vit c": "ascorbic acid",
    "ascorbic acid": "ascorbic acid",
    "西地兰": "digoxin",
    "地高辛": "digoxin",
    "digoxin": "digoxin",
    "lanoxin": "digoxin",
    "deslanoside": "digoxin",
    "cedilanid": "digoxin",
}

RAG_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "override_instructions": re.compile(
        r"(ignore|bypass).{0,30}(instruction|rule|policy)|忽略.{0,20}(指令|规则|策略)",
        flags=re.IGNORECASE,
    ),
    "execute_commands": re.compile(
        r"(execute|run).{0,30}(command|shell|bash|powershell|sql)|(执行|运行).{0,20}(命令|脚本|sql)",
        flags=re.IGNORECASE,
    ),
    "data_export": re.compile(
        r"(export|dump|print|list).{0,40}(subject_id|patient|raw data|raw text)|"
        r"(导出|打印|列出).{0,40}(subject_id|患者|原始数据|原始文本)",
        flags=re.IGNORECASE,
    ),
}

QUESTION_POLICY_PATTERNS: dict[str, re.Pattern[str]] = {
    "destructive_sql": re.compile(
        r"\bdrop\s+table\b|\btruncate\s+table\b|\bdelete\s+from\b|\balter\s+table\b|"
        r"\bupdate\s+\w+\s+set\b|删库|删表|删除表|清空表",
        flags=re.IGNORECASE,
    ),
    "patient_exfiltration": re.compile(
        r"subject_id.{0,20}(全部|all)|"
        r"(打印|列出|导出|export|dump).{0,30}(subject_id|患者|patient)|"
        r"patient[- ]level|raw\s*text\s*export|导出原始文本",
        flags=re.IGNORECASE,
    ),
    "policy_override": re.compile(
        r"(ignore|bypass).{0,20}(rule|policy|instruction)|忽略.{0,20}(规则|策略|指令)",
        flags=re.IGNORECASE,
    ),
}


class PolicyRejectedError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    status: str
    plan_path: Path
    trace_path: Path
    final_answer_path: Path | None
    facts_path: Path | None
    error: str | None
    steps: list[dict[str, Any]]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_doc_types(doc_types: list[str] | None) -> list[str]:
    if not doc_types:
        return sorted(ALLOWED_DOC_TYPES)
    out = sorted({str(x).strip() for x in doc_types if str(x).strip()})
    illegal = sorted(set(out) - set(ALLOWED_DOC_TYPES))
    if illegal:
        raise ValueError(f"unsupported doc_type filter: {illegal}")
    return out


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        txt = str(value).strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def _model_dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="python"))
    if hasattr(value, "dict"):
        return dict(value.dict())
    raise TypeError(f"unsupported response type: {type(value)}")


class AgentRunner:
    def __init__(
        self,
        *,
        artifacts_root: Path = ARTIFACTS_DIR,
        data_dir: Path = DEMO_DATA_DIR,
        global_manifest_path: Path = DEMO_MANIFEST_PATH,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        embedding_model: str | None = None,
        registry: ToolRegistry | None = None,
        planner: Planner | None = None,
        retriever: KnowledgeBaseRetriever | None = None,
        tool_executor: ToolExecutor | None = None,
        answer_writer: AnswerWriter | None = None,
    ) -> None:
        self.artifacts_root = Path(artifacts_root).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.global_manifest_path = Path(global_manifest_path).resolve()

        ollama_base_url = str(
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        )
        ollama_model = str(
            ollama_model
            or os.getenv("OLLAMA_MODEL")
            or DEFAULT_OLLAMA_MODEL
        )
        embedding_model = str(
            embedding_model
            or os.getenv("OLLAMA_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )

        self.registry = registry or self._build_registry()
        self.tool_executor = tool_executor or ToolExecutor(registry=self.registry)
        self.planner = planner or Planner(
            registry=self.registry,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
        )

        if retriever is None:
            embedding_client = OllamaEmbeddingClient(
                model=embedding_model,
                base_url=ollama_base_url,
            )
            retriever = KnowledgeBaseRetriever(embedding_client=embedding_client)
        self.retriever = retriever

        self.answer_writer = answer_writer or AnswerWriter(artifacts_root=self.artifacts_root)

    def run_question(
        self,
        *,
        question: str,
        run_id: str | None = None,
        actor: str = "agent",
        constraints: dict[str, Any] | PlanConstraints | None = None,
        kb_top_k: int = 5,
        kb_doc_types: list[str] | None = None,
    ) -> AgentRunResult:
        question_txt = str(question or "").strip()
        if not question_txt:
            raise ValueError("question cannot be empty")

        constraints_obj = self._parse_constraints(constraints)
        run_id_txt, run_uuid, run_dir = self._ensure_run(
            run_id=run_id,
            question=question_txt,
            params={
                "source": "agent_runner",
                "constraints": constraints_obj.model_dump(mode="python"),
            },
        )

        plan_path = run_dir / "plan.json"
        trace_path = run_dir / "agent_trace.json"
        final_answer_path: Path | None = None
        facts_path: Path | None = None

        started = perf_counter()
        started_at = _utc_now_iso()
        status = "SUCCEEDED"
        err_msg: str | None = None
        step_traces: list[dict[str, Any]] = []
        rag_filter_meta: dict[str, Any] = {
            "retrieved": 0,
            "kept": 0,
            "rejected_count": 0,
            "rejected": [],
        }
        cohort_context: dict[str, Any] = {}

        insert_audit(
            run_uuid,
            actor,
            "agent_run_started",
            {
                "question": question_txt,
                "constraints": constraints_obj.model_dump(mode="python"),
            },
        )
        update_run_status(run_uuid, "AGENT_RUNNING")

        try:
            self._enforce_question_policy(question=question_txt)

            rag_snippets, rag_filter_meta = self._retrieve_snippets(
                question=question_txt,
                top_k=max(1, int(kb_top_k)),
                doc_types=kb_doc_types,
            )
            insert_audit(
                run_uuid,
                actor,
                "agent_rag_filter",
                rag_filter_meta,
            )

            plan: ResearchPlan = self.planner.create_plan(
                question=question_txt,
                rag_snippets=rag_snippets,
                constraints=constraints_obj,
            )
            plan_json = plan.to_plan_json()
            plan_path.write_text(
                json.dumps(plan_json, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
            insert_audit(
                run_uuid,
                actor,
                "agent_plan_created",
                {
                    "plan_path": str(plan_path),
                    "step_count": len(plan.steps),
                },
            )

            for idx, step in enumerate(plan.steps, start=1):
                step_started = perf_counter()
                resolved_args = self._resolve_step_args(
                    step_tool=step.tool,
                    raw_args=step.args,
                    run_id=run_id_txt,
                    constraints=plan.constraints,
                    run_dir=run_dir,
                    cohort_context=cohort_context,
                )
                trace_item: dict[str, Any] = {
                    "index": idx,
                    "tool": step.tool,
                    "input": step.args,
                    "validated_args": resolved_args,
                    "status": "running",
                    "started_at": _utc_now_iso(),
                    "duration_seconds": 0.0,
                    "error": None,
                    "output": None,
                    "output_paths": [],
                }
                try:
                    output = self.tool_executor.execute(
                        tool_name=step.tool,
                        raw_args=resolved_args,
                        actor=actor,
                        run_id=run_id_txt,
                    )
                    trace_item["output"] = output
                    step_guard_error = self._validate_step_output(
                        step_tool=step.tool,
                        output=output,
                    )
                    if step_guard_error is None:
                        trace_item["status"] = "succeeded"
                    else:
                        status = "FAILED"
                        err_msg = step_guard_error
                        trace_item["status"] = "failed"
                        trace_item["error"] = step_guard_error
                except Exception as exc:  # noqa: BLE001
                    status = "FAILED"
                    err_msg = self._format_error(exc)
                    trace_item["status"] = "failed"
                    trace_item["error"] = err_msg
                finally:
                    trace_item["duration_seconds"] = round(
                        perf_counter() - step_started,
                        6,
                    )
                    trace_item["output_paths"] = self._collect_output_paths(
                        step_tool=step.tool,
                        run_dir=run_dir,
                        validated_args=resolved_args,
                    )
                    step_traces.append(trace_item)

                if trace_item["status"] == "succeeded" and step.tool == "build_cohort":
                    cohort_context = {
                        "template_name": str(resolved_args.get("template_name", "")).strip(),
                        "params": dict(resolved_args.get("params", {}))
                        if isinstance(resolved_args.get("params"), dict)
                        else {},
                    }

                if trace_item["status"] != "succeeded":
                    break

            if status == "SUCCEEDED":
                final_answer_path, facts_path, evidence = self.answer_writer.write_final_answer(
                    run_id=run_id_txt,
                    question=question_txt,
                )
                insert_audit(
                    run_uuid,
                    actor,
                    "agent_answer_written",
                    {
                        "final_answer_path": str(final_answer_path),
                        "facts_path": str(facts_path),
                        "fact_count": len(evidence),
                    },
                )
        except PolicyRejectedError as exc:
            status = "REJECTED"
            err_msg = exc.message
            insert_audit(
                run_uuid,
                actor,
                "agent_policy_reject",
                {
                    "code": exc.code,
                    "message": exc.message,
                    "question": question_txt,
                },
            )
        except Exception as exc:  # noqa: BLE001
            status = "FAILED"
            err_msg = self._format_error(exc)
            LOGGER.exception("agent_run_failed run_id=%s err=%s", run_id_txt, err_msg)

        finished_at = _utc_now_iso()
        duration_seconds = round(perf_counter() - started, 6)
        trace_payload = {
            "run_id": run_id_txt,
            "question": question_txt,
            "status": status,
            "error": err_msg,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "plan_path": str(plan_path),
            "final_answer_path": str(final_answer_path) if final_answer_path else None,
            "facts_path": str(facts_path) if facts_path else None,
            "rag_filter": rag_filter_meta,
            "steps": step_traces,
        }
        trace_path.write_text(
            json.dumps(trace_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        if status == "SUCCEEDED":
            run_status = "AGENT_SUCCEEDED"
        elif status == "REJECTED":
            run_status = "AGENT_REJECTED"
        else:
            run_status = "AGENT_FAILED"
        update_run_status(run_uuid, run_status)
        insert_audit(
            run_uuid,
            actor,
            "agent_run_finished",
            {
                "status": status,
                "error": err_msg,
                "trace_path": str(trace_path),
                "duration_seconds": duration_seconds,
            },
        )

        return AgentRunResult(
            run_id=run_id_txt,
            status=status,
            plan_path=plan_path,
            trace_path=trace_path,
            final_answer_path=final_answer_path,
            facts_path=facts_path,
            error=err_msg,
            steps=step_traces,
        )

    @staticmethod
    def _parse_constraints(
        value: dict[str, Any] | PlanConstraints | None,
    ) -> PlanConstraints:
        if isinstance(value, PlanConstraints):
            return value
        if value is None:
            return PlanConstraints()
        return PlanConstraints.model_validate(value)

    def _ensure_run(
        self,
        *,
        run_id: str | None,
        question: str,
        params: dict[str, Any],
    ) -> tuple[str, uuid.UUID, Path]:
        run_uuid = uuid.UUID(str(run_id)) if run_id else uuid.uuid4()
        run_id_txt = str(run_uuid)
        run_dir = self.artifacts_root / run_id_txt
        run_dir.mkdir(parents=True, exist_ok=True)

        row = get_run(run_uuid)
        if row is None:
            insert_run(run_uuid, question, params, "AGENT_CREATED", str(run_dir))

        params_path = run_dir / "params.json"
        params_payload = {
            "run_id": run_id_txt,
            "question": question,
            "params": params,
            "status": "AGENT_CREATED",
            "updated_at": _utc_now_iso(),
            "artifacts_path": str(run_dir),
        }
        params_path.write_text(
            json.dumps(params_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return run_id_txt, run_uuid, run_dir

    def _enforce_question_policy(self, *, question: str) -> None:
        text = str(question or "").strip()
        if not text:
            raise PolicyRejectedError(
                code="empty_question",
                message="question cannot be empty",
            )

        if QUESTION_POLICY_PATTERNS["destructive_sql"].search(text):
            raise PolicyRejectedError(
                code="destructive_sql_request",
                message="request rejected: destructive SQL intent is not allowed",
            )
        if QUESTION_POLICY_PATTERNS["patient_exfiltration"].search(text):
            raise PolicyRejectedError(
                code="patient_level_export_request",
                message="request rejected: patient-level export is not allowed",
            )
        if QUESTION_POLICY_PATTERNS["policy_override"].search(text):
            raise PolicyRejectedError(
                code="policy_override_request",
                message="request rejected: cannot bypass safety rules",
            )

    @staticmethod
    def _detect_snippet_injection_reason(text: str) -> str | None:
        raw = str(text or "")
        for reason, pattern in RAG_INJECTION_PATTERNS.items():
            if pattern.search(raw):
                return reason
        return None

    def _retrieve_snippets(
        self,
        *,
        question: str,
        top_k: int,
        doc_types: list[str] | None,
    ) -> tuple[list[str], dict[str, Any]]:
        try:
            raw_snippets = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                doc_types=set(_normalize_doc_types(doc_types)),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("kb_retrieve_failed fallback_empty_snippets err=%s", exc)
            return [], {
                "retrieved": 0,
                "kept": 0,
                "rejected_count": 0,
                "rejected": [],
                "error": str(exc),
            }

        retrieved_count = len(raw_snippets)
        rejected: list[dict[str, Any]] = []

        if raw_snippets and isinstance(raw_snippets[0], str):
            kept_strings: list[str] = []
            for idx, snippet_text in enumerate(raw_snippets):
                reason = self._detect_snippet_injection_reason(str(snippet_text))
                if reason is not None:
                    rejected.append(
                        {
                            "index": idx,
                            "reason": reason,
                        }
                    )
                    continue
                kept_strings.append(str(snippet_text))
            return kept_strings, {
                "retrieved": retrieved_count,
                "kept": len(kept_strings),
                "rejected_count": len(rejected),
                "rejected": rejected,
            }

        kept_snippets: list[Any] = []
        for snippet in raw_snippets:
            content = str(getattr(snippet, "content", ""))
            reason = self._detect_snippet_injection_reason(content)
            if reason is not None:
                rejected.append(
                    {
                        "doc_name": str(getattr(snippet, "doc_name", "unknown")),
                        "chunk_idx": int(getattr(snippet, "chunk_idx", -1)),
                        "reason": reason,
                    }
                )
                continue
            kept_snippets.append(snippet)

        return format_snippets_for_prompt(kept_snippets), {
            "retrieved": retrieved_count,
            "kept": len(kept_snippets),
            "rejected_count": len(rejected),
            "rejected": rejected,
        }

    def _resolve_step_args(
        self,
        *,
        step_tool: str,
        raw_args: dict[str, Any],
        run_id: str,
        constraints: PlanConstraints,
        run_dir: Path,
        cohort_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = self._replace_tokens(value=raw_args, run_id=run_id)
        if step_tool in {"build_cohort", "extract_ecg_features", "generate_report", "read_artifact_summary"}:
            if not str(resolved.get("run_id", "")).strip():
                resolved["run_id"] = run_id

        if step_tool == "build_cohort":
            return self._resolve_build_cohort_args(args=resolved, constraints=constraints)
        if step_tool == "extract_ecg_features":
            return self._resolve_extract_args(
                args=resolved,
                run_dir=run_dir,
                constraints=constraints,
                cohort_context=cohort_context,
            )
        if step_tool == "generate_report":
            return self._resolve_generate_report_args(
                args=resolved,
                cohort_context=cohort_context,
            )
        if step_tool == "read_artifact_summary":
            return self._resolve_read_artifact_summary_args(args=resolved)
        return resolved

    def _resolve_build_cohort_args(
        self,
        *,
        args: dict[str, Any],
        constraints: PlanConstraints,
    ) -> dict[str, Any]:
        out = self._normalize_build_cohort_args(args)
        template_name = str(out.get("template_name", "")).strip()
        params = out.get("params") if isinstance(out.get("params"), dict) else {}
        guarded_params, _guard_meta = self._apply_cohort_guardrails(
            template_name=template_name,
            params=params,
        )
        out["params"] = guarded_params

        max_n = min(MAX_COHORT_LIMIT, int(constraints.max_records_per_run))
        limit = _safe_int(out.get("limit"), default=max_n)
        if limit <= 0:
            limit = max_n
        out["limit"] = min(max(1, limit), max_n)
        return out

    @staticmethod
    def _normalize_build_cohort_args(args: dict[str, Any]) -> dict[str, Any]:
        out = dict(args)

        template_name = ""
        for key in ("template_name", "template", "cohort_template"):
            value = str(out.get(key, "")).strip()
            if value:
                template_name = value
                break
        if template_name:
            out["template_name"] = template_name
        out.pop("template", None)
        out.pop("cohort_template", None)

        params = out.get("params")
        normalized_params = dict(params) if isinstance(params, dict) else {}
        top_level_param_keys = [
            "window_hours",
            "k_threshold",
            "label_keyword",
            "icd_prefixes",
            "icd_codes",
            "icd_version",
            "drug_names",
            "drug_keywords",
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
            if key in out:
                if key not in normalized_params:
                    normalized_params[key] = out[key]
                out.pop(key, None)
        out["params"] = normalized_params
        return out

    def _apply_cohort_guardrails(
        self,
        *,
        template_name: str,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        out = dict(params)
        changes: dict[str, Any] = {}

        key_map = {
            "electrolyte_hyperkalemia": ("charttime_start", "charttime_end"),
            "diagnosis_icd": ("admittime_start", "admittime_end"),
            "medication_exposure": ("starttime_start", "starttime_end"),
        }
        start_key, end_key = key_map.get(template_name, (None, None))
        if start_key is not None and not str(out.get(start_key, "")).strip():
            out[start_key] = DEFAULT_TIME_WINDOW_START
            changes[start_key] = DEFAULT_TIME_WINDOW_START
        if end_key is not None and not str(out.get(end_key, "")).strip():
            out[end_key] = DEFAULT_TIME_WINDOW_END
            changes[end_key] = DEFAULT_TIME_WINDOW_END

        if template_name == "medication_exposure":
            canonical_keywords = self._canonicalize_drug_keywords(out.get("drug_keywords"))
            if canonical_keywords:
                if out.get("drug_keywords") != canonical_keywords:
                    changes["drug_keywords"] = canonical_keywords
                out["drug_keywords"] = canonical_keywords
            pre_hours = self._normalize_hour_value(
                out.get("pre_hours"),
                default=DEFAULT_COHORT_WINDOW_HOURS,
            )
            post_hours = self._normalize_hour_value(
                out.get("post_hours"),
                default=DEFAULT_COHORT_WINDOW_HOURS,
            )
            if out.get("pre_hours") != pre_hours:
                changes["pre_hours"] = pre_hours
            if out.get("post_hours") != post_hours:
                changes["post_hours"] = post_hours
            out["pre_hours"] = pre_hours
            out["post_hours"] = post_hours
        else:
            window_hours = self._normalize_hour_value(
                out.get("window_hours"),
                default=DEFAULT_COHORT_WINDOW_HOURS,
            )
            if out.get("window_hours") != window_hours:
                changes["window_hours"] = window_hours
            out["window_hours"] = window_hours

        return out, changes

    @staticmethod
    def _normalize_hour_value(value: Any, *, default: int) -> int:
        hours = _safe_int(value, default=default)
        if hours <= 0:
            hours = default
        return max(1, min(MAX_COHORT_WINDOW_HOURS, hours))

    @staticmethod
    def _canonicalize_drug_keyword(value: str) -> str:
        key = re.sub(r"\s+", " ", str(value or "").strip().lower())
        if not key:
            return ""
        return DRUG_KEYWORD_CANONICAL_MAP.get(key, key)

    @classmethod
    def _canonicalize_drug_keywords(cls, raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            keyword = cls._canonicalize_drug_keyword(str(item))
            if not keyword or keyword in seen:
                continue
            seen.add(keyword)
            out.append(keyword)
        return out


    def _resolve_extract_args(
        self,
        *,
        args: dict[str, Any],
        run_dir: Path,
        constraints: PlanConstraints,
        cohort_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = dict(args)
        params = out.get("params") if isinstance(out.get("params"), dict) else {}
        template_name = str((cohort_context or {}).get("template_name", "")).strip()
        cohort_params = (
            dict((cohort_context or {}).get("params", {}))
            if isinstance((cohort_context or {}).get("params"), dict)
            else {}
        )
        has_med_window = self._is_medication_window_mode(
            template_name=template_name,
            params=cohort_params,
        )
        manifest_path = Path(
            str(params.get("global_manifest", self.global_manifest_path))
        ).resolve()

        raw_record_ids = out.get("record_ids")
        if not isinstance(raw_record_ids, list) and isinstance(out.get("records"), list):
            raw_record_ids = out.get("records")
        cleaned = (
            [str(x).strip() for x in raw_record_ids if str(x).strip()]
            if isinstance(raw_record_ids, list)
            else []
        )
        needs_auto = (not cleaned) or any(x == "$AUTO_FROM_COHORT" for x in cleaned)
        max_n = int(constraints.max_records_per_run)
        if needs_auto:
            if has_med_window:
                cleaned = self._load_medication_window_record_ids(
                    cohort_path=run_dir / "cohort.parquet",
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    pre_hours=_safe_int(cohort_params.get("pre_hours"), default=DEFAULT_COHORT_WINDOW_HOURS),
                    post_hours=_safe_int(cohort_params.get("post_hours"), default=DEFAULT_COHORT_WINDOW_HOURS),
                    max_records=max_n,
                    candidate_record_ids=None,
                )
            else:
                cleaned = self._load_record_ids_from_cohort(
                    cohort_path=run_dir / "cohort.parquet",
                    max_records=max_n,
                    manifest_path=manifest_path,
                )
        else:
            cleaned = _dedupe_keep_order(cleaned)
            cleaned = cleaned[:max_n]
            if has_med_window:
                cleaned = self._load_medication_window_record_ids(
                    cohort_path=run_dir / "cohort.parquet",
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    pre_hours=_safe_int(cohort_params.get("pre_hours"), default=DEFAULT_COHORT_WINDOW_HOURS),
                    post_hours=_safe_int(cohort_params.get("post_hours"), default=DEFAULT_COHORT_WINDOW_HOURS),
                    max_records=max_n,
                    candidate_record_ids=cleaned,
                )

        if not has_med_window:
            for rel in (MEDICATION_WINDOW_MAP_FILE, MEDICATION_WINDOW_SUMMARY_FILE):
                stale = run_dir / rel
                if stale.exists():
                    stale.unlink()

        if not cleaned:
            raise RuntimeError("extract_ecg_features has empty record_ids after resolution")

        out["record_ids"] = cleaned
        if "limit" in out and "limit" not in params:
            params["limit"] = out.get("limit")
        step_limit = _safe_int(params.get("limit"), default=max_n)
        if step_limit <= 0:
            step_limit = max_n
        params["limit"] = min(step_limit, max_n)
        params["max_records_per_run"] = max_n
        out["params"] = params
        out.pop("records", None)
        out.pop("limit", None)
        return out

    def _resolve_generate_report_args(
        self,
        *,
        args: dict[str, Any],
        cohort_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out = dict(args)
        config = out.get("config") if isinstance(out.get("config"), dict) else {}
        params = config.get("params") if isinstance(config.get("params"), dict) else {}
        template_name = str((cohort_context or {}).get("template_name", "")).strip()
        cohort_params = (
            dict((cohort_context or {}).get("params", {}))
            if isinstance((cohort_context or {}).get("params"), dict)
            else {}
        )

        if template_name:
            params["template_name"] = template_name

        medication_keys = {
            "source",
            "drug_names",
            "drug_keywords",
            "pre_hours",
            "post_hours",
            "starttime_start",
            "starttime_end",
        }
        diagnosis_keys = {
            "window_hours",
            "icd_prefixes",
            "icd_codes",
            "icd_version",
            "admittime_start",
            "admittime_end",
        }
        electrolyte_keys = {
            "window_hours",
            "k_threshold",
            "label_keyword",
            "charttime_start",
            "charttime_end",
        }

        if template_name == "medication_exposure":
            pass_through_keys = medication_keys
        elif template_name == "diagnosis_icd":
            pass_through_keys = diagnosis_keys
        elif template_name == "electrolyte_hyperkalemia":
            pass_through_keys = electrolyte_keys
        else:
            pass_through_keys = set()

        if template_name != "medication_exposure":
            for key in medication_keys:
                params.pop(key, None)

        for key in pass_through_keys:
            if key in cohort_params and key not in params:
                params[key] = cohort_params[key]

        config["params"] = params
        out["config"] = config
        return out

    def _resolve_read_artifact_summary_args(self, *, args: dict[str, Any]) -> dict[str, Any]:
        out = dict(args)
        artifact_name = self._canonical_summary_artifact_name(str(out.get("artifact_name", "")))
        if artifact_name == "cohort_summary.json":
            artifact_name = self._canonical_summary_artifact_name(str(out.get("artifact", "")))
        if artifact_name == "cohort_summary.json":
            artifact_type = str(out.get("artifact_type", "")).strip().lower()
            artifact_name = self._canonical_summary_artifact_name(artifact_type)
        out["artifact_name"] = artifact_name
        out.pop("artifact", None)
        out.pop("artifact_type", None)
        return out

    @staticmethod
    def _normalize_artifact_name(value: str) -> str:
        return str(value or "").strip().replace("\\", "/")

    @classmethod
    def _canonical_summary_artifact_name(cls, artifact_name: str) -> str:
        normalized = cls._normalize_artifact_name(artifact_name)
        if not normalized:
            return "cohort_summary.json"
        return ARTIFACT_SUMMARY_NAME_MAP.get(normalized.lower(), normalized)

    @classmethod
    def _summarize_parquet_artifact(cls, artifact_path: Path) -> dict[str, Any]:
        df = pd.read_parquet(artifact_path)
        return {
            "format": "parquet",
            "path": cls._normalize_artifact_name(str(artifact_path.name)),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": [str(col) for col in df.columns.tolist()],
        }
    @staticmethod
    def _replace_tokens(*, value: Any, run_id: str) -> Any:
        if isinstance(value, str):
            return run_id if value == "$RUN_ID" else value
        if isinstance(value, list):
            return [AgentRunner._replace_tokens(value=x, run_id=run_id) for x in value]
        if isinstance(value, dict):
            return {
                str(k): AgentRunner._replace_tokens(value=v, run_id=run_id)
                for k, v in value.items()
            }
        return value


    def _load_record_ids_from_cohort(
        self,
        *,
        cohort_path: Path,
        max_records: int,
        manifest_path: Path | None = None,
    ) -> list[str]:
        if not cohort_path.exists():
            raise FileNotFoundError(f"cohort not found: {cohort_path}")
        effective_manifest_path = manifest_path or self.global_manifest_path
        if not effective_manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {effective_manifest_path}")

        cohort_df = pd.read_parquet(cohort_path)
        if "subject_id" not in cohort_df.columns:
            raise RuntimeError("cohort.parquet missing subject_id")

        manifest_df = pd.read_parquet(
            effective_manifest_path,
            columns=["record_id", "subject_id"],
        )

        cohort_subjects = cohort_df[["subject_id"]].copy()
        cohort_subjects["subject_id"] = cohort_subjects["subject_id"].astype("string").str.strip()
        cohort_subjects = cohort_subjects[cohort_subjects["subject_id"].notna()]
        cohort_subjects["_ord"] = range(len(cohort_subjects))

        manifest_df["subject_id"] = manifest_df["subject_id"].astype("string").str.strip()
        manifest_df["record_id"] = manifest_df["record_id"].astype("string").str.strip()
        manifest_df = manifest_df[
            manifest_df["subject_id"].notna() & manifest_df["record_id"].notna()
        ].copy()

        joined = cohort_subjects.merge(manifest_df, on="subject_id", how="left")
        joined = joined[joined["record_id"].notna()].copy()
        joined = joined.sort_values(["_ord", "record_id"]).drop_duplicates(
            subset=["record_id"],
            keep="first",
        )

        record_ids = [str(v).strip() for v in joined["record_id"].tolist() if str(v).strip()]
        if max_records > 0:
            record_ids = record_ids[:max_records]
        return record_ids

    @staticmethod
    def _is_medication_window_mode(*, template_name: str, params: dict[str, Any]) -> bool:
        if str(template_name).strip() != "medication_exposure":
            return False
        pre_hours = _safe_int(params.get("pre_hours"), default=0)
        post_hours = _safe_int(params.get("post_hours"), default=0)
        return pre_hours > 0 or post_hours > 0

    def _load_medication_window_record_ids(
        self,
        *,
        cohort_path: Path,
        run_dir: Path,
        manifest_path: Path,
        pre_hours: int,
        post_hours: int,
        max_records: int,
        candidate_record_ids: list[str] | None,
    ) -> list[str]:
        if not cohort_path.exists():
            raise FileNotFoundError(f"cohort not found: {cohort_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")

        cohort_df = pd.read_parquet(cohort_path).copy()
        if "subject_id" not in cohort_df.columns or "index_time" not in cohort_df.columns:
            raise RuntimeError(
                "cohort.parquet must contain subject_id/index_time for medication pre/post analysis"
            )
        cohort_df["subject_id"] = cohort_df["subject_id"].astype("string").str.strip()
        cohort_df["index_time"] = pd.to_datetime(
            cohort_df["index_time"],
            errors="coerce",
            utc=True,
        )
        cohort_df = cohort_df[
            cohort_df["subject_id"].notna() & cohort_df["index_time"].notna()
        ].copy()
        if cohort_df.empty:
            raise RuntimeError("cohort has no valid subject_id/index_time rows")
        cohort_df["_cohort_ord"] = range(len(cohort_df))
        cohort_df["_event_id"] = cohort_df["_cohort_ord"].astype(int)

        manifest_df = pd.read_parquet(manifest_path).copy()
        required_manifest_cols = {"record_id", "subject_id", "ecg_time"}
        missing_manifest_cols = sorted(required_manifest_cols - set(manifest_df.columns))
        if missing_manifest_cols:
            raise RuntimeError(
                "medication pre/post analysis requires manifest columns "
                f"{sorted(required_manifest_cols)}; missing {missing_manifest_cols}. "
                "Please regenerate storage/ecg_manifest.parquet with ecg_time."
            )
        manifest_df = manifest_df[["record_id", "subject_id", "ecg_time"]].copy()
        manifest_df["record_id"] = manifest_df["record_id"].astype("string").str.strip()
        manifest_df["subject_id"] = manifest_df["subject_id"].astype("string").str.strip()
        manifest_df["ecg_time"] = pd.to_datetime(
            manifest_df["ecg_time"],
            errors="coerce",
            utc=True,
        )
        manifest_df = manifest_df[
            manifest_df["record_id"].notna()
            & manifest_df["subject_id"].notna()
            & manifest_df["ecg_time"].notna()
        ].copy()
        manifest_df = manifest_df.drop_duplicates(subset=["record_id"], keep="first")
        if candidate_record_ids:
            wanted = set(_dedupe_keep_order(candidate_record_ids))
            manifest_df = manifest_df[manifest_df["record_id"].isin(wanted)].copy()
        if manifest_df.empty:
            raise RuntimeError("manifest has no valid ECG rows for medication window selection")

        cohort_cols = ["subject_id", "index_time", "_cohort_ord", "_event_id"]
        if "hadm_id" in cohort_df.columns:
            cohort_cols.append("hadm_id")
        if "cohort_label" in cohort_df.columns:
            cohort_cols.append("cohort_label")
        joined = manifest_df.merge(
            cohort_df[cohort_cols],
            on="subject_id",
            how="inner",
        )
        if joined.empty:
            raise RuntimeError("no ECG records matched cohort subject_id for medication window analysis")

        joined["delta_hours"] = (
            (joined["ecg_time"] - joined["index_time"]).dt.total_seconds() / 3600.0
        )
        joined = joined[
            (joined["delta_hours"] >= -float(pre_hours))
            & (joined["delta_hours"] <= float(post_hours))
        ].copy()
        if joined.empty:
            raise RuntimeError(
                f"no ECG records in medication window [-{pre_hours}h, +{post_hours}h] around index_time"
            )

        joined["abs_delta_hours"] = joined["delta_hours"].abs()
        joined["window_group"] = joined["delta_hours"].apply(
            lambda value: "pre" if float(value) < 0.0 else "post"
        )
        joined = joined.sort_values(
            ["_event_id", "window_group", "abs_delta_hours", "ecg_time", "record_id"],
            ascending=[True, True, True, True, True],
        )

        # keep one nearest ECG per exposure event per window side (pre/post)
        per_event_group = joined.drop_duplicates(
            subset=["_event_id", "window_group"],
            keep="first",
        ).copy()
        if per_event_group.empty:
            raise RuntimeError("medication window selection produced no usable event-group rows")

        event_group_matrix = per_event_group.pivot_table(
            index="_event_id",
            columns="window_group",
            values="record_id",
            aggfunc="count",
            fill_value=0,
        )
        for col in ("pre", "post"):
            if col not in event_group_matrix.columns:
                event_group_matrix[col] = 0

        def _pair_status(row: pd.Series) -> str:
            has_pre = int(row.get("pre", 0)) > 0
            has_post = int(row.get("post", 0)) > 0
            if has_pre and has_post:
                return "both"
            if has_pre:
                return "pre_only"
            if has_post:
                return "post_only"
            return "none"

        event_group_matrix["pair_status"] = event_group_matrix.apply(_pair_status, axis=1)
        status_map = {
            int(event_id): str(status)
            for event_id, status in event_group_matrix["pair_status"].to_dict().items()
        }

        per_event_group["pair_status"] = (
            per_event_group["_event_id"].map(status_map).fillna("unknown")
        )
        per_event_group = per_event_group[per_event_group["pair_status"] != "none"].copy()
        if per_event_group.empty:
            raise RuntimeError("medication window selection produced no pre/post candidate rows")

        per_event_group["pair_event_id"] = per_event_group["_event_id"].astype(int).astype(str)
        status_rank = {"both": 0, "pre_only": 1, "post_only": 2, "unknown": 3}
        per_event_group["_status_rank"] = per_event_group["pair_status"].map(status_rank).fillna(9)
        per_event_group = per_event_group.sort_values(
            ["_status_rank", "_event_id", "window_group", "abs_delta_hours", "record_id"],
            ascending=[True, True, True, True, True],
        )

        event_counts_all = {
            "both": int((event_group_matrix["pair_status"] == "both").sum()),
            "pre_only": int((event_group_matrix["pair_status"] == "pre_only").sum()),
            "post_only": int((event_group_matrix["pair_status"] == "post_only").sum()),
        }
        paired_event_count_all = int(event_counts_all.get("both", 0))
        no_paired_fallback = bool(pre_hours > 0 and post_hours > 0 and paired_event_count_all <= 0)
        if no_paired_fallback:
            LOGGER.warning(
                "medication_window_no_pairs_fallback_unpaired "
                "pre_hours=%s post_hours=%s event_counts=%s",
                pre_hours,
                post_hours,
                event_counts_all,
            )

        selected = per_event_group.copy()
        if max_records > 0 and len(selected) > max_records:
            chunks: list[pd.DataFrame] = []
            used_record_ids: set[str] = set()

            if max_records >= 2:
                max_pair_events = max_records // 2
                if max_pair_events > 0:
                    keep_pair_events = (
                        selected[selected["pair_status"] == "both"]["_event_id"]
                        .drop_duplicates()
                        .head(max_pair_events)
                        .tolist()
                    )
                    if keep_pair_events:
                        paired_rows = selected[selected["_event_id"].isin(set(keep_pair_events))].copy()
                        chunks.append(paired_rows)
                        used_record_ids = {
                            str(x)
                            for x in paired_rows["record_id"].astype(str).tolist()
                            if str(x).strip()
                        }

            kept = sum(len(chunk) for chunk in chunks)
            remaining = max(0, max_records - kept)
            if remaining > 0:
                remainder = selected[
                    ~selected["record_id"].astype(str).isin(used_record_ids)
                ].copy()
                remainder = remainder.sort_values(
                    ["_status_rank", "abs_delta_hours", "_event_id", "record_id"],
                    ascending=[True, True, True, True],
                )
                if not remainder.empty:
                    chunks.append(remainder.head(remaining))

            if chunks:
                selected = pd.concat(chunks, ignore_index=True)
            else:
                selected = selected.head(max_records).copy()

        selected = selected.sort_values(
            ["_status_rank", "_event_id", "window_group", "abs_delta_hours", "record_id"],
            ascending=[True, True, True, True, True],
        )

        window_counts_all = per_event_group["window_group"].value_counts().to_dict()
        window_counts_selected = selected["window_group"].value_counts().to_dict()

        selected_event_status = selected[["_event_id", "pair_status"]].drop_duplicates(subset=["_event_id"])
        selected_event_counts = {
            "both": int((selected_event_status["pair_status"] == "both").sum()),
            "pre_only": int((selected_event_status["pair_status"] == "pre_only").sum()),
            "post_only": int((selected_event_status["pair_status"] == "post_only").sum()),
        }

        selected_subject_status = selected[["subject_id", "_event_id", "pair_status"]].drop_duplicates(
            subset=["_event_id"]
        )
        subject_counts = {
            "both": int(
                selected_subject_status[selected_subject_status["pair_status"] == "both"][
                    "subject_id"
                ].nunique()
            ),
            "pre_only": int(
                selected_subject_status[selected_subject_status["pair_status"] == "pre_only"][
                    "subject_id"
                ].nunique()
            ),
            "post_only": int(
                selected_subject_status[selected_subject_status["pair_status"] == "post_only"][
                    "subject_id"
                ].nunique()
            ),
        }

        map_cols = [
            "record_id",
            "subject_id",
            "ecg_time",
            "index_time",
            "delta_hours",
            "abs_delta_hours",
            "window_group",
            "pair_event_id",
            "pair_status",
        ]
        if "hadm_id" in selected.columns:
            map_cols.append("hadm_id")
        if "cohort_label" in selected.columns:
            map_cols.append("cohort_label")
        window_map_df = selected[map_cols].copy()
        window_map_path = run_dir / MEDICATION_WINDOW_MAP_FILE
        window_map_df.to_parquet(window_map_path, index=False)

        summary = {
            "selected_records": int(len(window_map_df)),
            "window_group_counts": {str(k): int(v) for k, v in window_counts_selected.items()},
            "window_group_counts_all": {str(k): int(v) for k, v in window_counts_all.items()},
            "event_counts": {str(k): int(v) for k, v in event_counts_all.items()},
            "selected_event_counts": {str(k): int(v) for k, v in selected_event_counts.items()},
            "subject_counts": {str(k): int(v) for k, v in subject_counts.items()},
            "paired_event_count": paired_event_count_all,
            "paired_record_count": int((window_map_df["pair_status"] == "both").sum()),
            "main_analysis_mode": (
                "paired_pre_post_per_exposure"
                if (pre_hours > 0 and post_hours > 0 and paired_event_count_all > 0)
                else "unpaired_window_comparison"
            ),
            "no_paired_fallback": no_paired_fallback,
            "pre_hours": int(pre_hours),
            "post_hours": int(post_hours),
            "manifest_path": str(manifest_path),
            "candidate_restricted": bool(candidate_record_ids),
        }
        (run_dir / MEDICATION_WINDOW_SUMMARY_FILE).write_text(
            json.dumps(summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        record_ids = [
            str(value).strip()
            for value in window_map_df["record_id"].astype(str).tolist()
            if str(value).strip()
        ]
        return _dedupe_keep_order(record_ids)

    def _collect_output_paths(
        self,
        *,
        step_tool: str,
        run_dir: Path,
        validated_args: dict[str, Any],
    ) -> list[str]:
        candidates: list[str] = []
        if step_tool == "build_cohort":
            candidates.extend(["cohort.parquet", "cohort_summary.json"])
        elif step_tool == "extract_ecg_features":
            candidates.extend(
                [
                    "ecg_qc.parquet",
                    "ecg_qc_summary.json",
                    "ecg_features.parquet",
                    "ecg_features_summary.json",
                    MEDICATION_WINDOW_MAP_FILE,
                    MEDICATION_WINDOW_SUMMARY_FILE,
                    "plots/plots_summary.json",
                ]
            )
        elif step_tool == "generate_report":
            candidates.extend(
                [
                    "analysis_tables/analysis_dataset.parquet",
                    "analysis_tables/analysis_dataset_all.parquet",
                    "analysis_tables/analysis_dataset_summary_all.json",
                    "analysis_tables/feature_summary.parquet",
                    "analysis_tables/group_compare.parquet",
                    "analysis_tables/feature_summary_sensitivity_all.parquet",
                    "analysis_tables/group_compare_sensitivity_all.parquet",
                    "analysis_tables/analysis_tables_summary_sensitivity_all.json",
                    "plots/report_plots_summary.json",
                    "report.md",
                    "run_metadata.json",
                    "report_task_result.json",
                ]
            )
        elif step_tool == "read_artifact_summary":
            artifact_name = str(validated_args.get("artifact_name", "cohort_summary.json"))
            candidates.append(artifact_name)

        out: list[str] = []
        for rel in candidates:
            if (run_dir / rel).exists():
                out.append(rel)
        return out

    @staticmethod
    def _validate_step_output(*, step_tool: str, output: Any) -> str | None:
        if step_tool != "build_cohort":
            return None
        payload = output if isinstance(output, dict) else {}
        row_count = _safe_int(payload.get("row_count"), default=-1)
        if row_count == 0:
            return (
                "build_cohort returned 0 rows; no cohort matched the current query "
                "(try canonical diagnosis/drug mapping or broaden the time window)"
            )
        return None

    @staticmethod
    def _format_error(exc: Exception) -> str:
        if isinstance(exc, HTTPException):
            return f"HTTP {exc.status_code}: {exc.detail}"
        return str(exc)

    @staticmethod
    def _ensure_pipeline_import_path() -> None:
        file_path = Path(__file__).resolve()
        repo_root: Path | None = None
        for parent in file_path.parents:
            if (parent / "pipelines").exists():
                repo_root = parent
                break
        if repo_root is None:
            return
        txt = str(repo_root)
        if txt not in sys.path:
            sys.path.insert(0, txt)

    def _build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="build_cohort",
                input_schema=BuildCohortInput,
                output_schema=BuildCohortOutput,
                permission_level=PermissionLevel.GENERATE_ARTIFACTS,
                handler=self._handle_build_cohort,
                timeout_seconds=60.0,
                retry_count=0,
            )
        )
        registry.register(
            ToolSpec(
                name="extract_ecg_features",
                input_schema=ExtractEcgFeaturesInput,
                output_schema=ExtractEcgFeaturesOutput,
                permission_level=PermissionLevel.GENERATE_ARTIFACTS,
                handler=self._handle_extract_ecg_features,
                timeout_seconds=600.0,
                retry_count=0,
            )
        )
        registry.register(
            ToolSpec(
                name="generate_report",
                input_schema=GenerateReportInput,
                output_schema=GenerateReportOutput,
                permission_level=PermissionLevel.GENERATE_ARTIFACTS,
                handler=self._handle_generate_report,
                timeout_seconds=600.0,
                retry_count=0,
            )
        )
        registry.register(
            ToolSpec(
                name="read_artifact_summary",
                input_schema=ReadArtifactSummaryInput,
                output_schema=ReadArtifactSummaryOutput,
                permission_level=PermissionLevel.READ_ONLY,
                handler=self._handle_read_artifact_summary,
                timeout_seconds=10.0,
                retry_count=0,
            )
        )
        return registry

    def _handle_build_cohort(self, body: BuildCohortInput) -> dict[str, Any]:
        from app.routes.tools import BuildCohortRequest, build_cohort

        guarded_params, guard_changes = self._apply_cohort_guardrails(
            template_name=body.template_name,
            params=body.params,
        )
        safe_limit = min(MAX_COHORT_LIMIT, max(1, _safe_int(body.limit, default=1000)))

        response = build_cohort(
            BuildCohortRequest(
                template_name=body.template_name,
                params=guarded_params,
                run_id=body.run_id,
                limit=safe_limit,
            ),
            x_actor="agent",
        )

        if guard_changes and body.run_id:
            try:
                run_uuid = uuid.UUID(str(body.run_id))
            except ValueError:
                run_uuid = None
            if run_uuid is not None:
                insert_audit(
                    run_uuid,
                    "agent",
                    "agent_guardrail_build_cohort",
                    {
                        "template_name": body.template_name,
                        "changes": guard_changes,
                        "limit": safe_limit,
                    },
                )
        return _model_dump(response)

    def _handle_extract_ecg_features(self, body: ExtractEcgFeaturesInput) -> dict[str, Any]:
        self._ensure_pipeline_import_path()
        from pipelines.ecg_features import FEATURE_VERSION, run_features
        from pipelines.ecg_plots import generate_qc_feature_plots
        from pipelines.ecg_qc import run_qc

        run_id = body.run_id
        params = body.params if isinstance(body.params, dict) else {}
        run_dir = self.artifacts_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        data_dir = Path(str(params.get("data_dir", self.data_dir))).resolve()
        manifest_path = Path(
            str(params.get("global_manifest", self.global_manifest_path))
        ).resolve()
        qc_thresholds = (
            params.get("qc_thresholds")
            if isinstance(params.get("qc_thresholds"), dict)
            else None
        )
        feature_thresholds = (
            params.get("feature_thresholds")
            if isinstance(params.get("feature_thresholds"), dict)
            else None
        )
        feature_version = str(params.get("feature_version", FEATURE_VERSION))

        max_records = _safe_int(params.get("max_records_per_run"), default=len(body.record_ids))
        if max_records <= 0:
            max_records = len(body.record_ids)
        capped_record_ids = body.record_ids[:max_records]

        limit = _safe_int(params.get("limit"), default=max_records)
        if limit <= 0:
            limit = max_records
        limit = min(limit, max_records)

        if len(capped_record_ids) < len(body.record_ids):
            insert_audit(
                uuid.UUID(run_id),
                "agent",
                "agent_guardrail_extract_records",
                {
                    "requested_records": len(body.record_ids),
                    "capped_records": len(capped_record_ids),
                    "max_records_per_run": max_records,
                },
            )

        job_id = uuid.uuid4().hex
        queued_at = _utc_now_iso()
        qc_path, qc_summary = run_qc(
            run_id=run_id,
            data_dir=data_dir,
            record_ids=capped_record_ids,
            global_manifest_path=manifest_path,
            artifacts_root=self.artifacts_root,
            thresholds=qc_thresholds,
        )
        features_path, features_summary = run_features(
            run_id=run_id,
            data_dir=data_dir,
            global_manifest_path=manifest_path,
            artifacts_root=self.artifacts_root,
            qc_path=qc_path,
            record_ids=capped_record_ids,
            limit=limit,
            feature_version=feature_version,
            thresholds=feature_thresholds,
        )
        plots_summary = generate_qc_feature_plots(
            run_id=run_id,
            artifacts_root=self.artifacts_root,
            data_dir=data_dir,
            global_manifest_path=manifest_path,
        )

        task_result = {
            "job_id": job_id,
            "run_id": run_id,
            "status": "SUCCEEDED",
            "queued_at": queued_at,
            "finished_at": _utc_now_iso(),
            "record_count": len(capped_record_ids),
            "qc_path": str(qc_path),
            "features_path": str(features_path),
            "qc_summary": qc_summary,
            "features_summary": features_summary,
            "plots_summary": plots_summary,
        }
        (run_dir / "ecg_feature_task_result.json").write_text(
            json.dumps(task_result, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        return {
            "ok": True,
            "run_id": run_id,
            "job_id": job_id,
            "queue_status": "SUCCEEDED",
            "queued_at": queued_at,
        }
    def _handle_generate_report(self, body: GenerateReportInput) -> dict[str, Any]:
        self._ensure_pipeline_import_path()
        from pipelines.assemble_analysis_dataset import assemble_analysis_dataset
        from pipelines.build_analysis_tables import build_analysis_tables
        from pipelines.build_report_plots import build_report_plots
        from pipelines.generate_report import generate_report

        run_id = body.run_id
        config = body.config if isinstance(body.config, dict) else {}
        run_dir = self.artifacts_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        queued_at = _utc_now_iso()
        manifest_path = Path(
            str(config.get("global_manifest", self.global_manifest_path))
        ).resolve()

        analysis_dataset_path, _analysis_summary = assemble_analysis_dataset(
            run_id=run_id,
            artifacts_root=self.artifacts_root,
            cohort_path=run_dir / "cohort.parquet",
            features_path=run_dir / "ecg_features.parquet",
            ecg_map_path=run_dir / "ecg_map.parquet",
            global_manifest_path=manifest_path,
            covariates_path=None,
            age_bin_mode="fixed",
            window_map_path=run_dir / MEDICATION_WINDOW_MAP_FILE,
        )

        analysis_dir = run_dir / "analysis_tables"
        analysis_summary_path = analysis_dir / "analysis_dataset_summary.json"
        analysis_dataset_all_path = analysis_dir / "analysis_dataset_all.parquet"
        analysis_summary_all_path = analysis_dir / "analysis_dataset_summary_all.json"

        def _count_values(df: pd.DataFrame, col: str, *, default: str = "Unknown") -> dict[str, int]:
            if col not in df.columns:
                return {}
            values = (
                df[col]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
                .fillna(default)
            )
            counts = values.value_counts(dropna=False)
            return {str(k): int(v) for k, v in counts.items()}

        def _write_json(path: Path, payload: dict[str, Any]) -> None:
            path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )

        paired_main_applied = False
        analysis_df = pd.read_parquet(analysis_dataset_path)
        if "pair_status" in analysis_df.columns:
            analysis_df["pair_status"] = (
                analysis_df["pair_status"]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "<na>": pd.NA})
                .fillna("unknown")
            )
        if "window_group" in analysis_df.columns:
            analysis_df["window_group"] = (
                analysis_df["window_group"]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "<na>": pd.NA})
                .fillna("unknown")
            )

        window_counts_all = _count_values(analysis_df, "window_group", default="unknown")
        pair_status_counts_all = _count_values(analysis_df, "pair_status", default="unknown")

        if "pair_status" in analysis_df.columns:
            analysis_df.to_parquet(analysis_dataset_all_path, index=False)
            summary_all = dict(_analysis_summary)
            summary_all.update(
                {
                    "analysis_mode": "all_window_records",
                    "rows": int(len(analysis_df)),
                    "rows_all": int(len(analysis_df)),
                    "window_group_counts": window_counts_all,
                    "window_group_counts_all": window_counts_all,
                    "pair_status_counts": pair_status_counts_all,
                    "pair_status_counts_all": pair_status_counts_all,
                }
            )
            _write_json(analysis_summary_all_path, summary_all)

            main_df = analysis_df[analysis_df["pair_status"] == "both"].copy()
            if {"pair_event_id", "window_group"}.issubset(set(main_df.columns)):
                pair_group_n = (
                    main_df.groupby("pair_event_id", dropna=False)["window_group"]
                    .nunique()
                    .astype(int)
                )
                valid_pair_ids = {
                    str(idx)
                    for idx, n in pair_group_n.items()
                    if int(n) >= 2
                }
                main_df = main_df[
                    main_df["pair_event_id"].astype("string").isin(valid_pair_ids)
                ].copy()

            main_window_counts = _count_values(main_df, "window_group", default="unknown")
            has_main_pre = int(main_window_counts.get("pre", 0)) > 0
            has_main_post = int(main_window_counts.get("post", 0)) > 0

            if not main_df.empty and has_main_pre and has_main_post:
                paired_main_applied = True
                main_df.to_parquet(analysis_dataset_path, index=False)

                feature_rows_total = int(_analysis_summary.get("feature_rows", len(analysis_df)))
                main_summary = dict(_analysis_summary)
                main_summary.update(
                    {
                        "analysis_mode": "paired_window_group_main",
                        "rows": int(len(main_df)),
                        "rows_all": int(len(analysis_df)),
                        "feature_rows": feature_rows_total,
                        "rows_match_features": bool(len(main_df) == feature_rows_total),
                        "missing_subject_id_count": int(
                            main_df["subject_id"].isna().sum()
                        )
                        if "subject_id" in main_df.columns
                        else 0,
                        "sex_counts": _count_values(main_df, "sex"),
                        "age_bin_counts": _count_values(main_df, "age_bin"),
                        "window_group_counts": main_window_counts,
                        "window_group_counts_all": window_counts_all,
                        "pair_status_counts": _count_values(main_df, "pair_status", default="unknown"),
                        "pair_status_counts_all": pair_status_counts_all,
                    }
                )
                _write_json(analysis_summary_path, main_summary)
                _analysis_summary = main_summary

        compare_by = "cohort_label"
        group_cols = ["cohort_label", "sex", "age_bin"]
        window_counts = _analysis_summary.get("window_group_counts", {})
        has_pre = int(window_counts.get("pre", 0)) > 0
        has_post = int(window_counts.get("post", 0)) > 0

        cfg_params = config.get("params") if isinstance(config.get("params"), dict) else {}
        if has_pre and has_post:
            compare_by = "window_group"
            group_cols = ["window_group", "cohort_label", "sex", "age_bin"]
            cfg_params.setdefault("compare_by", "window_group")
            if paired_main_applied:
                cfg_params.setdefault("analysis_mode", "paired_pre_post_per_exposure")
                cfg_params.setdefault(
                    "sensitivity_dataset_path",
                    "analysis_tables/analysis_dataset_all.parquet",
                )
                cfg_params.setdefault(
                    "sensitivity_compare_path",
                    "analysis_tables/group_compare_sensitivity_all.parquet",
                )
        if pair_status_counts_all:
            cfg_params.setdefault("pair_status_counts_all", pair_status_counts_all)
        config["params"] = cfg_params

        _feature_summary_path, _group_compare_path, _tables_summary = build_analysis_tables(
            run_id=run_id,
            artifacts_root=self.artifacts_root,
            analysis_dataset_path=analysis_dataset_path,
            group_cols=group_cols,
            compare_by=compare_by,
            compare_features=["mean_hr", "rr_std", "rr_mean"],
            output_suffix="",
        )

        if paired_main_applied and analysis_dataset_all_path.exists():
            sensitivity_compare_by = "cohort_label"
            sensitivity_group_cols = ["cohort_label", "sex", "age_bin"]
            has_all_pre = int(window_counts_all.get("pre", 0)) > 0
            has_all_post = int(window_counts_all.get("post", 0)) > 0
            if has_all_pre and has_all_post:
                sensitivity_compare_by = "window_group"
                sensitivity_group_cols = ["window_group", "cohort_label", "sex", "age_bin"]

            build_analysis_tables(
                run_id=run_id,
                artifacts_root=self.artifacts_root,
                analysis_dataset_path=analysis_dataset_all_path,
                group_cols=sensitivity_group_cols,
                compare_by=sensitivity_compare_by,
                compare_features=["mean_hr", "rr_std", "rr_mean"],
                output_suffix="sensitivity_all",
            )

        _plots_summary = build_report_plots(
            run_id=run_id,
            artifacts_root=self.artifacts_root,
            analysis_dataset_path=analysis_dataset_path,
            qc_path=run_dir / "ecg_qc.parquet",
            fail_top_n=8,
            feature_preferred=["rr_std", "qtc", "qtc_ms"],
            group_col=compare_by,
        )

        question_arg = config.get("question") if isinstance(config.get("question"), str) else None
        params_json_arg: str | None = None
        if isinstance(config.get("params"), dict):
            params_json_arg = json.dumps(config["params"], ensure_ascii=True)
        elif isinstance(config.get("params_json"), str):
            params_json_arg = config.get("params_json")

        report_path, metadata_path, _meta = generate_report(
            run_id=run_id,
            artifacts_root=self.artifacts_root,
            question_arg=question_arg,
            params_json_arg=params_json_arg,
            qc_path_override=None,
        )

        task_result = {
            "job_id": job_id,
            "run_id": run_id,
            "status": "REPORT_SUCCEEDED",
            "queued_at": queued_at,
            "finished_at": _utc_now_iso(),
            "report_path": str(report_path),
            "metadata_path": str(metadata_path),
        }
        (run_dir / "report_task_result.json").write_text(
            json.dumps(task_result, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        return {
            "ok": True,
            "run_id": run_id,
            "job_id": job_id,
            "queue_status": "SUCCEEDED",
            "queued_at": queued_at,
        }

    def _handle_read_artifact_summary(self, body: ReadArtifactSummaryInput) -> dict[str, Any]:
        run_dir = self.artifacts_root / body.run_id
        requested_name = self._normalize_artifact_name(body.artifact_name)
        artifact_name = self._canonical_summary_artifact_name(requested_name)
        artifact_path = run_dir / artifact_name
        if artifact_path.exists() and artifact_path.is_file():
            if artifact_path.suffix.lower() == ".json":
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
                summary = payload if isinstance(payload, dict) else {"value": payload}
                return {
                    "ok": True,
                    "run_id": body.run_id,
                    "artifact_name": artifact_name,
                    "summary": summary,
                }
            if artifact_path.suffix.lower() == ".parquet":
                return {
                    "ok": True,
                    "run_id": body.run_id,
                    "artifact_name": artifact_name,
                    "summary": self._summarize_parquet_artifact(artifact_path),
                }

        requested_path = run_dir / requested_name
        if requested_path.exists() and requested_path.is_file() and requested_path.suffix.lower() == ".parquet":
            return {
                "ok": True,
                "run_id": body.run_id,
                "artifact_name": requested_name,
                "summary": self._summarize_parquet_artifact(requested_path),
            }

        return {
            "ok": False,
            "run_id": body.run_id,
            "artifact_name": artifact_name,
            "summary": {},
        }






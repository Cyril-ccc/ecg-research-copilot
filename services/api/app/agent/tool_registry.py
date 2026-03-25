from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PermissionLevel(StrEnum):
    READ_ONLY = "read_only"
    GENERATE_ARTIFACTS = "generate_artifacts"


class RunSqlInput(BaseModel):
    sql: str
    limit: int = Field(default=1000, ge=1, le=5000)
    run_id: str | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        uuid.UUID(value)
        return value


class RunSqlOutput(BaseModel):
    ok: bool
    sql_sha256: str
    limited_sql: str | None = None
    row_count: int = 0
    rows: list[dict[str, Any]] = Field(default_factory=list)
    rejected_reason: str | None = None


class BuildCohortInput(BaseModel):
    template_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    limit: int = Field(default=1000, ge=1, le=5000)


class BuildCohortOutput(BaseModel):
    ok: bool
    template_name: str
    cohort_id: str | None = None
    cohort_table: str | None = None
    sql: str
    row_count: int
    rows: list[dict[str, Any]] = Field(default_factory=list)


class ExtractEcgFeaturesInput(BaseModel):
    run_id: str
    record_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        uuid.UUID(value)
        return value

    @field_validator("record_ids")
    @classmethod
    def validate_record_ids(cls, values: list[str]) -> list[str]:
        out = [str(v).strip() for v in values if str(v).strip()]
        if not out:
            raise ValueError("record_ids cannot be empty")
        return list(dict.fromkeys(out))


class ExtractEcgFeaturesOutput(BaseModel):
    ok: bool
    run_id: str
    job_id: str
    queue_status: str
    queued_at: str


class GenerateReportInput(BaseModel):
    run_id: str
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        uuid.UUID(value)
        return value


class GenerateReportOutput(BaseModel):
    ok: bool
    run_id: str
    job_id: str
    queue_status: str
    queued_at: str


class DemoReportInput(BaseModel):
    run_id: str | None = None
    sample_n: int = Field(default=10, ge=1, le=200)
    question: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        uuid.UUID(str(value).strip())
        return str(value).strip()


class DemoReportOutput(BaseModel):
    ok: bool
    run_id: str
    job_id: str
    queue_status: str
    queued_at: str


class ReadArtifactSummaryInput(BaseModel):
    run_id: str
    artifact_name: str = "cohort_summary.json"

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        uuid.UUID(value)
        return value


class ReadArtifactSummaryOutput(BaseModel):
    ok: bool
    run_id: str
    artifact_name: str
    summary: dict[str, Any] = Field(default_factory=dict)


ToolHandler = Callable[[BaseModel], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel] | None
    permission_level: PermissionLevel
    handler: ToolHandler
    timeout_seconds: float = 30.0
    retry_count: int = 0


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        key = spec.name.strip()
        if not key:
            raise ValueError("tool name cannot be empty")
        if key in self._tools:
            raise ValueError(f"tool already registered: {key}")
        self._tools[key] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(str(name).strip())

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())


def build_default_registry(*, handlers: dict[str, ToolHandler]) -> ToolRegistry:
    required_names = [
        "run_sql",
        "build_cohort",
        "extract_ecg_features",
        "generate_report",
        "demo_report",
    ]
    missing = [name for name in required_names if name not in handlers]
    if missing:
        raise ValueError(f"missing tool handlers: {missing}")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="run_sql",
            input_schema=RunSqlInput,
            output_schema=RunSqlOutput,
            permission_level=PermissionLevel.READ_ONLY,
            handler=handlers["run_sql"],
            timeout_seconds=10.0,
            retry_count=0,
        )
    )
    registry.register(
        ToolSpec(
            name="build_cohort",
            input_schema=BuildCohortInput,
            output_schema=BuildCohortOutput,
            permission_level=PermissionLevel.GENERATE_ARTIFACTS,
            handler=handlers["build_cohort"],
            timeout_seconds=30.0,
            retry_count=0,
        )
    )
    registry.register(
        ToolSpec(
            name="extract_ecg_features",
            input_schema=ExtractEcgFeaturesInput,
            output_schema=ExtractEcgFeaturesOutput,
            permission_level=PermissionLevel.GENERATE_ARTIFACTS,
            handler=handlers["extract_ecg_features"],
            timeout_seconds=30.0,
            retry_count=0,
        )
    )
    registry.register(
        ToolSpec(
            name="generate_report",
            input_schema=GenerateReportInput,
            output_schema=GenerateReportOutput,
            permission_level=PermissionLevel.GENERATE_ARTIFACTS,
            handler=handlers["generate_report"],
            timeout_seconds=30.0,
            retry_count=0,
        )
    )
    registry.register(
        ToolSpec(
            name="demo_report",
            input_schema=DemoReportInput,
            output_schema=DemoReportOutput,
            permission_level=PermissionLevel.GENERATE_ARTIFACTS,
            handler=handlers["demo_report"],
            timeout_seconds=60.0,
            retry_count=0,
        )
    )

    read_summary_handler = handlers.get("read_artifact_summary")
    if read_summary_handler is not None:
        registry.register(
            ToolSpec(
                name="read_artifact_summary",
                input_schema=ReadArtifactSummaryInput,
                output_schema=ReadArtifactSummaryOutput,
                permission_level=PermissionLevel.READ_ONLY,
                handler=read_summary_handler,
                timeout_seconds=10.0,
                retry_count=0,
            )
        )
    return registry

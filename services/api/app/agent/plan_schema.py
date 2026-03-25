from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class PlanConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_records_per_run: int = Field(default=2000, ge=1, le=50000)
    no_raw_text_export: bool = True


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    steps: list[PlanStep] = Field(min_length=1)
    constraints: PlanConstraints = Field(default_factory=PlanConstraints)

    @field_validator("steps")
    @classmethod
    def validate_steps_not_empty(cls, value: list[PlanStep]) -> list[PlanStep]:
        if not value:
            raise ValueError("steps cannot be empty")
        return value

    def ensure_tools_whitelisted(self, allowed_tools: set[str]) -> None:
        unknown = sorted({step.tool for step in self.steps if step.tool not in allowed_tools})
        if unknown:
            raise ValueError(f"plan contains non-whitelisted tools: {unknown}")

    def to_plan_json(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

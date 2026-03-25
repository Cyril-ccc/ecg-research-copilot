from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.agent.runner import AgentRunner

router = APIRouter(prefix="/agent")


class AgentAskRequest(BaseModel):
    question: str = Field(min_length=1)
    run_id: str | None = None
    constraints: dict[str, Any] | None = None
    kb_top_k: int = Field(default=5, ge=1, le=20)
    kb_doc_types: list[str] | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        uuid.UUID(str(value).strip())
        return str(value).strip()


class AgentAskResponse(BaseModel):
    ok: bool
    run_id: str
    status: str
    plan_path: str
    trace_path: str
    final_answer_path: str | None = None
    facts_path: str | None = None
    error: str | None = None
    step_count: int = 0


@router.post("/ask", response_model=AgentAskResponse)
def ask_agent(body: AgentAskRequest, x_actor: str | None = Header(default=None)):
    runner = AgentRunner()
    result = runner.run_question(
        question=body.question,
        run_id=body.run_id,
        actor=x_actor or "agent",
        constraints=body.constraints,
        kb_top_k=body.kb_top_k,
        kb_doc_types=body.kb_doc_types,
    )
    response = AgentAskResponse(
        ok=result.status == "SUCCEEDED",
        run_id=result.run_id,
        status=result.status,
        plan_path=str(result.plan_path),
        trace_path=str(result.trace_path),
        final_answer_path=str(result.final_answer_path) if result.final_answer_path else None,
        facts_path=str(result.facts_path) if result.facts_path else None,
        error=result.error,
        step_count=len(result.steps),
    )
    if result.status == "REJECTED":
        raise HTTPException(
            status_code=403,
            detail={
                "message": result.error or "request rejected by policy",
                "run_id": result.run_id,
                "trace_path": str(result.trace_path),
            },
        )
    return response


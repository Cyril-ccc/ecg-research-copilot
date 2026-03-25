import time
import uuid
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from app.agent.tool_executor import ToolExecutor
from app.agent.tool_registry import PermissionLevel, ToolRegistry, ToolSpec


class EchoInput(BaseModel):
    text: str
    run_id: str | None = None


class EchoOutput(BaseModel):
    echoed: str


def _build_executor(
    handler,
    *,
    timeout_seconds: float = 1.0,
) -> tuple[ToolExecutor, list[dict[str, Any]]]:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="echo",
            input_schema=EchoInput,
            output_schema=EchoOutput,
            permission_level=PermissionLevel.READ_ONLY,
            handler=handler,
            timeout_seconds=timeout_seconds,
            retry_count=0,
        )
    )

    audit_records: list[dict[str, Any]] = []

    def _audit(run_id: uuid.UUID | None, actor: str, action: str, payload: dict[str, Any]) -> None:
        audit_records.append(
            {
                "run_id": run_id,
                "actor": actor,
                "action": action,
                "payload": payload,
            }
        )

    return ToolExecutor(registry=registry, audit_writer=_audit), audit_records


def test_tool_name_not_in_registry_is_rejected_with_403():
    executor, audit_records = _build_executor(lambda m: {"echoed": m.text})

    with pytest.raises(HTTPException) as exc_info:
        executor.execute(tool_name="unknown_tool", raw_args={"text": "hi"}, actor="tester")

    assert exc_info.value.status_code == 403
    assert len(audit_records) == 1
    assert audit_records[0]["action"] == "tool_call"
    assert audit_records[0]["payload"]["tool_name"] == "unknown_tool"
    assert audit_records[0]["payload"]["status"] == "rejected_forbidden"


def test_invalid_args_are_rejected_before_handler_execution():
    calls = {"count": 0}

    def _handler(model: EchoInput) -> dict[str, Any]:
        calls["count"] += 1
        return {"echoed": model.text}

    executor, audit_records = _build_executor(_handler)

    with pytest.raises(HTTPException) as exc_info:
        executor.execute(tool_name="echo", raw_args={"missing_text": "x"}, actor="tester")

    assert exc_info.value.status_code == 422
    assert calls["count"] == 0
    assert len(audit_records) == 1
    assert audit_records[0]["payload"]["status"] == "rejected_invalid_args"


def test_timeout_is_recorded_and_returns_504():
    def _slow_handler(model: EchoInput) -> dict[str, Any]:
        time.sleep(0.2)
        return {"echoed": model.text}

    executor, audit_records = _build_executor(_slow_handler, timeout_seconds=0.05)

    with pytest.raises(HTTPException) as exc_info:
        executor.execute(tool_name="echo", raw_args={"text": "slow"}, actor="tester")

    assert exc_info.value.status_code == 504
    assert len(audit_records) == 1
    assert audit_records[0]["payload"]["status"] == "timeout"


def test_success_writes_audit_with_validated_args_and_duration():
    executor, audit_records = _build_executor(lambda m: {"echoed": m.text})

    result = executor.execute(tool_name="echo", raw_args={"text": "ok"}, actor="tester")

    assert result == {"echoed": "ok"}
    assert len(audit_records) == 1
    payload = audit_records[0]["payload"]
    assert payload["status"] == "succeeded"
    assert payload["validated_args"]["text"] == "ok"
    assert payload["duration"] >= 0

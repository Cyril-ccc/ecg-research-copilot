from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from time import perf_counter
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ValidationError

from app.agent.tool_registry import ToolRegistry, ToolSpec
from app.db.models import insert_audit

LOGGER = logging.getLogger("api.agent.tool_executor")

AuditWriter = Callable[[uuid.UUID | None, str, str, dict[str, Any]], None]


class ToolExecutor:
    def __init__(self, *, registry: ToolRegistry, audit_writer: AuditWriter | None = None) -> None:
        self.registry = registry
        self.audit_writer = audit_writer or insert_audit

    def execute(
        self,
        *,
        tool_name: str,
        raw_args: dict[str, Any],
        actor: str = "local",
        run_id: str | uuid.UUID | None = None,
    ) -> Any:
        start = perf_counter()
        spec = self.registry.get(tool_name)
        if spec is None:
            duration = perf_counter() - start
            self._audit(
                run_id=self._coerce_uuid(run_id),
                actor=actor,
                payload={
                    "tool_name": str(tool_name),
                    "validated_args": {},
                    "status": "rejected_forbidden",
                    "duration": duration,
                    "error": "tool not in registry",
                },
            )
            raise HTTPException(status_code=403, detail=f"tool '{tool_name}' is not allowed")

        try:
            validated_obj = spec.input_schema.model_validate(raw_args)
        except ValidationError as exc:
            duration = perf_counter() - start
            self._audit(
                run_id=self._coerce_uuid(run_id),
                actor=actor,
                payload={
                    "tool_name": spec.name,
                    "validated_args": {},
                    "status": "rejected_invalid_args",
                    "duration": duration,
                    "error": exc.errors(),
                },
            )
            raise HTTPException(
                status_code=422,
                detail={"message": "invalid tool args", "errors": exc.errors()},
            ) from exc

        validated_args = validated_obj.model_dump(mode="python")
        audit_run_id = self._coerce_uuid(run_id) or self._coerce_uuid(validated_args.get("run_id"))

        attempts = max(1, int(spec.retry_count) + 1)
        last_exc: Exception | None = None
        last_status = "failed"

        for attempt_idx in range(1, attempts + 1):
            try:
                output = self._call_with_timeout(spec=spec, validated_obj=validated_obj)
                normalized_output = self._validate_output(spec=spec, output=output)
                duration = perf_counter() - start
                self._audit(
                    run_id=audit_run_id,
                    actor=actor,
                    payload={
                        "tool_name": spec.name,
                        "validated_args": validated_args,
                        "status": "succeeded",
                        "duration": duration,
                        "attempt": attempt_idx,
                        "permission_level": spec.permission_level.value,
                    },
                )
                return normalized_output
            except TimeoutError as exc:
                last_exc = exc
                last_status = "timeout"
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                last_status = "failed"
                if attempt_idx < attempts:
                    continue
                break

        duration = perf_counter() - start
        self._audit(
            run_id=audit_run_id,
            actor=actor,
            payload={
                "tool_name": spec.name,
                "validated_args": validated_args,
                "status": last_status,
                "duration": duration,
                "attempt": attempts,
                "permission_level": spec.permission_level.value,
                "error": str(last_exc) if last_exc else "unknown",
            },
        )

        if last_status == "timeout":
            raise HTTPException(
                status_code=504,
                detail=f"tool '{spec.name}' timed out after {spec.timeout_seconds:.2f}s",
            )
        if isinstance(last_exc, HTTPException):
            raise last_exc
        raise HTTPException(status_code=500, detail=f"tool '{spec.name}' failed: {last_exc}")

    def _call_with_timeout(self, *, spec: ToolSpec, validated_obj: BaseModel) -> Any:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(spec.handler, validated_obj)
            try:
                if spec.timeout_seconds <= 0:
                    return future.result()
                return future.result(timeout=float(spec.timeout_seconds))
            except FutureTimeoutError as exc:
                future.cancel()
                raise TimeoutError("tool execution timeout") from exc

    @staticmethod
    def _validate_output(*, spec: ToolSpec, output: Any) -> Any:
        if spec.output_schema is None:
            return output
        validated = spec.output_schema.model_validate(output)
        return validated.model_dump(mode="python")

    def _audit(self, *, run_id: uuid.UUID | None, actor: str, payload: dict[str, Any]) -> None:
        try:
            self.audit_writer(run_id, actor, "tool_call", payload)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("tool_call_audit_failed err=%s", exc)

    @staticmethod
    def _coerce_uuid(value: Any) -> uuid.UUID | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        txt = str(value).strip()
        if not txt:
            return None
        try:
            return uuid.UUID(txt)
        except ValueError:
            return None

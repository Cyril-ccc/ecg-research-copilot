import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

from app.core.config import LOG_LEVEL

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "-"
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "run_id": getattr(record, "run_id", "-"),
        }

        # 可选字段（middleware/route 里如果 extra 带了就会出现）
        for k in ("method", "path", "status_code", "client"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(LOG_LEVEL)
    handler.addFilter(RequestContextFilter())
    handler.setFormatter(JsonFormatter())

    root.addHandler(handler)

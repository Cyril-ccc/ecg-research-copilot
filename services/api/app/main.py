import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import APP_INIT_DB_ON_STARTUP
from app.core.logging import configure_logging, request_id_var
from app.core.schema_whitelist import load_whitelist, whitelist_stats
from app.db.models import init_db
from app.routes.agent import router as agent_router
from app.routes.runs import router as runs_router
from app.routes.tools import router as tools_router

configure_logging()
logger = logging.getLogger("api")

app = FastAPI(title="ECG Research Copilot")
app.include_router(runs_router)
app.include_router(tools_router)
app.include_router(agent_router)

# Serve static UI pages at /ui/
_static = Path(__file__).parent / "static"


@app.get("/ui/agent", include_in_schema=False)
def ui_agent_page():
    return FileResponse(_static / "agent" / "index.html")


app.mount("/ui", StaticFiles(directory=str(_static), html=True), name="ui")


@app.on_event("startup")
def _startup():
    if APP_INIT_DB_ON_STARTUP:
        init_db()
    else:
        logger.info("startup_init_db_skipped")
    logger.info("startup_complete")
    wl = load_whitelist()
    tables, fields = whitelist_stats(wl)
    logger.info(f"schema_whitelist_loaded tables={tables} fields={fields}")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        logger.info(
            "request_finished",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "client": request.client.host if request.client else None,
            },
        )
        return response
    finally:
        request_id_var.reset(token)


@app.get("/health")
def health():
    return {"ok": True}

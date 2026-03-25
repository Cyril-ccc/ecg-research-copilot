import os
import re
from pathlib import Path


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./storage/artifacts")).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ecg:ecg@localhost:5432/ecg")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
META_DATABASE_URL = os.getenv(
    "META_DATABASE_URL",
    os.getenv("DATABASE_URL", "postgresql://ecg:ecg@localhost:5432/ecg"),
)
DATA_DATABASE_URL = os.getenv("DATA_DATABASE_URL", META_DATABASE_URL)
SCHEMA_WHITELIST_PATH = os.getenv("SCHEMA_WHITELIST_PATH", "config/schema_whitelist.yaml")
DATA_SCHEMA = os.getenv("DATA_SCHEMA", "mimiciv")
APP_INIT_DB_ON_STARTUP = _env_flag("APP_INIT_DB_ON_STARTUP", True)
DEMO_DATA_DIR = Path(
    os.getenv(
        "DEMO_DATA_DIR",
        "/data/mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1",
    )
).resolve()
DEMO_MANIFEST_PATH = Path(
    os.getenv("DEMO_MANIFEST_PATH", "/storage/ecg_manifest.parquet")
).resolve()

if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", DATA_SCHEMA):
    raise ValueError("DATA_SCHEMA must be a valid SQL identifier")

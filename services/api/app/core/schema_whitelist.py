import os
from pathlib import Path
from typing import Any

import yaml


def _resolve_whitelist_path() -> Path:
    env_path = os.getenv("SCHEMA_WHITELIST_PATH")
    if env_path:
        return Path(env_path)

    # Fallback for local runs: walk up from this file and look for
    # config/schema_whitelist.yaml at any ancestor.
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidate = parent / "config" / "schema_whitelist.yaml"
        if candidate.exists():
            return candidate

    # Fallback for docker-compose volume mount.
    return Path("/config/schema_whitelist.yaml")


def load_whitelist() -> dict[str, Any]:
    path = _resolve_whitelist_path()
    if not path.exists():
        raise FileNotFoundError(
            f"schema whitelist file not found at '{path}'. "
            "Set SCHEMA_WHITELIST_PATH to a valid file path."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("whitelist yaml must be a mapping")
    return data


def whitelist_stats(whitelist: dict[str, Any]) -> tuple[int, int]:
    allow = whitelist.get("allow") or {}
    table_count = 0
    field_count = 0
    for _schema, tables in allow.items():
        if not isinstance(tables, dict):
            continue
        table_count += len(tables)
        for _t, cols in tables.items():
            if isinstance(cols, list):
                field_count += len(cols)
    return table_count, field_count

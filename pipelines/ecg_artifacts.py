from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("ecg_artifacts")


def write_run_ecg_map(
    *,
    run_dir: Path,
    global_manifest_path: Path,
    record_ids: Sequence[str],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "ecg_map.parquet"

    unique_ids = list(dict.fromkeys(str(v).strip() for v in record_ids if str(v).strip()))
    if not unique_ids:
        pd.DataFrame(
            columns=["record_id", "subject_id", "source"]
        ).to_parquet(out_path, index=False)
        return out_path

    manifest_df = pd.read_parquet(global_manifest_path)
    if "record_id" not in manifest_df.columns:
        raise RuntimeError(
            f"Global manifest missing required column: record_id ({global_manifest_path})"
        )

    extra_cols = [c for c in ("subject_id", "ecg_time", "source") if c in manifest_df.columns]
    lookup = manifest_df[["record_id", *extra_cols]].copy()
    lookup["record_id"] = lookup["record_id"].astype(str)
    lookup = lookup.drop_duplicates(subset=["record_id"], keep="first")

    req = pd.DataFrame({"record_id": unique_ids})
    out_df = req.merge(lookup, on="record_id", how="left")
    if "subject_id" not in out_df.columns:
        out_df["subject_id"] = pd.NA
        LOGGER.warning("Global manifest has no subject_id column: %s", global_manifest_path)

    ordered_cols = ["record_id", "subject_id"] + [
        c for c in ("ecg_time", "source") if c in out_df.columns
    ]
    out_df = out_df[ordered_cols]
    out_df.to_parquet(out_path, index=False)
    return out_path

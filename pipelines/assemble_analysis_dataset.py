"""
Assemble run-level analysis dataset for reporting.

Inputs (from artifacts/<run_id>/):
- cohort.parquet
- ecg_features.parquet
- ecg_map.parquet (preferred; auto-generated from global manifest if missing)
- covariates.parquet (optional)

Output:
- analysis_tables/analysis_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from pipelines.ecg_artifacts import write_run_ecg_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("assemble_analysis_dataset")


def _normalize_id(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.str.replace(r"\.0+$", "", regex=True)
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return out


def _first_non_null(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    out = df[existing[0]].copy()
    for col in existing[1:]:
        out = out.where(out.notna(), df[col])
    return out


def _normalize_sex(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    low = raw.str.lower()
    mapped = pd.Series(pd.NA, index=series.index, dtype="string")
    mapped = mapped.mask(low.isin({"m", "male", "man", "boy"}), "M")
    mapped = mapped.mask(low.isin({"f", "female", "woman", "girl"}), "F")
    mapped = mapped.mask(low.isin({"unknown", "unk", "na", "n/a"}), "Unknown")
    mapped = mapped.where(mapped.notna(), raw)
    return mapped.fillna("Unknown")


def _build_age_bin(age: pd.Series, mode: str) -> pd.Series:
    age_num = pd.to_numeric(age, errors="coerce")
    out = pd.Series(["Unknown"] * len(age_num), index=age_num.index, dtype="string")

    if mode == "quartile":
        valid = age_num.dropna()
        if len(valid) >= 4 and valid.nunique() >= 4:
            try:
                labels = ["Q1", "Q2", "Q3", "Q4"]
                q = pd.qcut(valid, q=4, labels=labels, duplicates="drop")
                out.loc[q.index] = q.astype("string")
                return out
            except ValueError:
                pass

    out = out.mask(age_num < 40, "<40")
    out = out.mask((age_num >= 40) & (age_num <= 60), "40-60")
    out = out.mask(age_num > 60, ">60")
    return out


def _read_parquet_required(path: Path, required_cols: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_parquet(path)
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise RuntimeError(f"File {path} missing required columns: {missing}")
    return df


def _read_parquet_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def assemble_analysis_dataset(
    *,
    run_id: str,
    artifacts_root: Path,
    cohort_path: Path,
    features_path: Path,
    ecg_map_path: Path,
    global_manifest_path: Path,
    covariates_path: Path | None = None,
    age_bin_mode: str = "fixed",
    window_map_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_dir = artifacts_root / run_id
    analysis_dir = run_dir / "analysis_tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    cohort_df = _read_parquet_required(
        cohort_path,
        required_cols={"subject_id", "index_time", "cohort_label"},
    ).copy()
    features_df = _read_parquet_required(features_path, required_cols={"record_id"}).copy()

    features_df["record_id"] = _normalize_id(features_df["record_id"])
    features_df = features_df.dropna(subset=["record_id"]).copy()

    if not ecg_map_path.exists():
        LOGGER.warning(
            "ecg_map.parquet missing; freezing mapping from global manifest: %s",
            ecg_map_path,
        )
        write_run_ecg_map(
            run_dir=run_dir,
            global_manifest_path=global_manifest_path,
            record_ids=features_df["record_id"].astype(str).tolist(),
        )

    ecg_map_df = _read_parquet_required(
        ecg_map_path,
        required_cols={"record_id", "subject_id"},
    ).copy()
    ecg_map_df["record_id"] = _normalize_id(ecg_map_df["record_id"])
    ecg_map_df["subject_id"] = _normalize_id(ecg_map_df["subject_id"])
    ecg_map_df = ecg_map_df.dropna(subset=["record_id"]).drop_duplicates(
        subset=["record_id"],
        keep="first",
    )
    ecg_map_df = ecg_map_df.rename(columns={"source": "source_map"})

    merged = features_df.merge(
        ecg_map_df,
        on="record_id",
        how="left",
        validate="m:1",
    )

    if "subject_id_x" in merged.columns and "subject_id_y" in merged.columns:
        subj_map = _normalize_id(merged["subject_id_y"])
        subj_feature = _normalize_id(merged["subject_id_x"])
        merged["subject_id"] = subj_map.where(subj_map.notna(), subj_feature)
        merged = merged.drop(columns=["subject_id_y"])
        merged = merged.drop(columns=["subject_id_x"])
    elif "subject_id_y" in merged.columns:
        merged["subject_id"] = _normalize_id(merged["subject_id_y"])
        merged = merged.drop(columns=["subject_id_y"])
    else:
        merged["subject_id"] = _normalize_id(merged["subject_id"])

    cohort_df["subject_id"] = _normalize_id(cohort_df["subject_id"])
    cohort_df["_index_time_ts"] = pd.to_datetime(cohort_df["index_time"], errors="coerce")
    cohort_df = cohort_df.sort_values(["subject_id", "_index_time_ts"], na_position="last")
    cohort_df = cohort_df.drop_duplicates(subset=["subject_id"], keep="first")

    cohort_cols = ["subject_id", "index_time", "cohort_label"]
    for col in ("age", "sex", "gender"):
        if col in cohort_df.columns:
            cohort_cols.append(col)
    cohort_df = cohort_df[cohort_cols]

    analysis_df = merged.merge(
        cohort_df,
        on="subject_id",
        how="left",
        validate="m:1",
    )

    cov_df = _read_parquet_optional(covariates_path) if covariates_path else None
    if cov_df is not None:
        if "record_id" in cov_df.columns:
            cov_df = cov_df.copy()
            cov_df["record_id"] = _normalize_id(cov_df["record_id"])
            cov_df = cov_df.drop_duplicates(subset=["record_id"], keep="first")
            analysis_df = analysis_df.merge(
                cov_df,
                on="record_id",
                how="left",
                suffixes=("", "_cov"),
            )
        elif "subject_id" in cov_df.columns:
            cov_df = cov_df.copy()
            cov_df["subject_id"] = _normalize_id(cov_df["subject_id"])
            cov_df = cov_df.drop_duplicates(subset=["subject_id"], keep="first")
            analysis_df = analysis_df.merge(
                cov_df,
                on="subject_id",
                how="left",
                suffixes=("", "_cov"),
            )
        else:
            LOGGER.warning("covariates.parquet has no record_id/subject_id key and will be ignored")

    window_df = _read_parquet_optional(window_map_path) if window_map_path else None
    if window_df is not None:
        if "record_id" not in window_df.columns:
            LOGGER.warning("window_map.parquet missing record_id and will be ignored")
        else:
            window_df = window_df.copy()
            window_df["record_id"] = _normalize_id(window_df["record_id"])
            window_df = window_df.dropna(subset=["record_id"]).drop_duplicates(
                subset=["record_id"],
                keep="first",
            )
            rename_map: dict[str, str] = {}
            if "index_time" in window_df.columns:
                rename_map["index_time"] = "index_time_window"
            if "hadm_id" in window_df.columns:
                rename_map["hadm_id"] = "hadm_id_window"
            if rename_map:
                window_df = window_df.rename(columns=rename_map)

            analysis_df = analysis_df.merge(
                window_df,
                on="record_id",
                how="left",
                suffixes=("", "_window"),
            )

            if "index_time_window" in analysis_df.columns:
                analysis_df["index_time"] = _first_non_null(
                    analysis_df,
                    ["index_time_window", "index_time"],
                )
            if "hadm_id_window" in analysis_df.columns:
                analysis_df["hadm_id"] = _first_non_null(
                    analysis_df,
                    ["hadm_id_window", "hadm_id"],
                )
            if "window_group" in analysis_df.columns:
                analysis_df["window_group"] = (
                    analysis_df["window_group"]
                    .astype("string")
                    .str.strip()
                    .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
                    .fillna("Unknown")
                )

    analysis_df["dataset_source"] = _first_non_null(
        analysis_df,
        ["source_map", "source"],
    ).fillna("unknown")
    analysis_df["sex"] = _normalize_sex(
        _first_non_null(analysis_df, ["sex", "gender", "sex_cov", "gender_cov"])
    )
    analysis_df["age"] = pd.to_numeric(
        _first_non_null(analysis_df, ["age", "age_cov"]),
        errors="coerce",
    )
    analysis_df["age_bin"] = _build_age_bin(analysis_df["age"], mode=age_bin_mode)

    out_path = analysis_dir / "analysis_dataset.parquet"
    analysis_df.to_parquet(out_path, index=False)

    sex_counts = (
        analysis_df.groupby("sex", dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "sex"], ascending=[False, True])
    )
    age_bin_counts = (
        analysis_df.groupby("age_bin", dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "age_bin"], ascending=[False, True])
    )
    window_group_counts: dict[str, int] = {}
    if "window_group" in analysis_df.columns:
        wg_counts = (
            analysis_df.groupby("window_group", dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["n", "window_group"], ascending=[False, True])
        )
        window_group_counts = {
            str(r["window_group"]): int(r["n"]) for _, r in wg_counts.iterrows()
        }

    summary = {
        "run_id": run_id,
        "rows": int(len(analysis_df)),
        "feature_rows": int(len(features_df)),
        "rows_match_features": bool(len(analysis_df) == len(features_df)),
        "missing_subject_id_count": int(analysis_df["subject_id"].isna().sum()),
        "sex_counts": {str(r["sex"]): int(r["n"]) for _, r in sex_counts.iterrows()},
        "age_bin_counts": {str(r["age_bin"]): int(r["n"]) for _, r in age_bin_counts.iterrows()},
        "window_group_counts": window_group_counts,
    }
    (analysis_dir / "analysis_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return out_path, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble run-level analysis dataset parquet")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--artifacts-root",
        default=str(PROJECT_ROOT / "storage" / "artifacts"),
    )
    parser.add_argument("--cohort-parquet", default=None)
    parser.add_argument("--features-parquet", default=None)
    parser.add_argument("--ecg-map-parquet", default=None)
    parser.add_argument(
        "--global-manifest",
        default=str(PROJECT_ROOT / "storage" / "ecg_manifest.parquet"),
    )
    parser.add_argument("--covariates-parquet", default=None)
    parser.add_argument("--window-map-parquet", default=None)
    parser.add_argument("--age-bin-mode", choices=["fixed", "quartile"], default="fixed")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    artifacts_root = Path(args.artifacts_root)
    run_dir = artifacts_root / args.run_id

    cohort_path = (
        Path(args.cohort_parquet) if args.cohort_parquet else run_dir / "cohort.parquet"
    )
    features_path = (
        Path(args.features_parquet) if args.features_parquet else run_dir / "ecg_features.parquet"
    )
    ecg_map_path = (
        Path(args.ecg_map_parquet) if args.ecg_map_parquet else run_dir / "ecg_map.parquet"
    )
    covariates_path = (
        Path(args.covariates_parquet) if args.covariates_parquet else run_dir / "covariates.parquet"
    )
    window_map_path = (
        Path(args.window_map_parquet) if args.window_map_parquet else run_dir / "ecg_window_map.parquet"
    )

    try:
        out_path, summary = assemble_analysis_dataset(
            run_id=args.run_id,
            artifacts_root=artifacts_root,
            cohort_path=cohort_path,
            features_path=features_path,
            ecg_map_path=ecg_map_path,
            global_manifest_path=Path(args.global_manifest),
            covariates_path=covariates_path,
            age_bin_mode=args.age_bin_mode,
            window_map_path=window_map_path,
        )
    except Exception as exc:
        LOGGER.error("assemble_analysis_dataset failed: %s", exc)
        return 1

    LOGGER.info("Wrote analysis dataset: %s", out_path)
    LOGGER.info("rows=%d feature_rows=%d", summary["rows"], summary["feature_rows"])
    LOGGER.info("sex_counts=%s", summary["sex_counts"])
    LOGGER.info("age_bin_counts=%s", summary["age_bin_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

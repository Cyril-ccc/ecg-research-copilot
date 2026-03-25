"""
Build descriptive and group-comparison tables from analysis_dataset.parquet.

Outputs (under artifacts/<run_id>/analysis_tables):
- feature_summary*.parquet
- group_compare*.parquet
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("build_analysis_tables")

DEFAULT_GROUP_COLS = ["cohort_label", "sex", "age_bin"]
DEFAULT_COMPARE_FEATURES = ["mean_hr", "rr_std", "rr_mean"]
RESERVED_NON_FEATURE_COLS = {
    "record_id",
    "subject_id",
    "hadm_id",
    "index_time",
    "dataset_source",
    "source",
    "source_map",
    "age",
    "cohort_label",
    "sex",
    "age_bin",
    "window_group",
    "delta_hours",
    "ecg_time",
}


def _normalize_group_col(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return out.fillna("Unknown")


def _ensure_group_columns(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in group_cols:
        if col not in out.columns:
            out[col] = "Unknown"
        out[col] = _normalize_group_col(out[col])
    return out


def _infer_feature_columns(df: pd.DataFrame, group_cols: list[str]) -> list[str]:
    excluded = RESERVED_NON_FEATURE_COLS | set(group_cols)
    feature_cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            feature_cols.append(col)
    return feature_cols


def _safe_float(value: float | np.floating[Any] | None) -> float:
    if value is None:
        return float("nan")
    val = float(value)
    if np.isfinite(val):
        return val
    return float("nan")


def _cohens_d(x_a: pd.Series, x_b: pd.Series) -> float:
    n_a = len(x_a)
    n_b = len(x_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(x_a.var(ddof=1))
    var_b = float(x_b.var(ddof=1))
    pooled_num = (n_a - 1) * var_a + (n_b - 1) * var_b
    pooled_den = n_a + n_b - 2
    if pooled_den <= 0:
        return float("nan")
    pooled_sd = float(np.sqrt(pooled_num / pooled_den))
    if not np.isfinite(pooled_sd) or pooled_sd == 0.0:
        return float("nan")
    return float((x_a.mean() - x_b.mean()) / pooled_sd)


def _build_feature_summary(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    group_var = " x ".join(group_cols)

    for group_key, gdf in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_map = dict(zip(group_cols, group_key, strict=False))
        group_value = "|".join(f"{c}={group_map[c]}" for c in group_cols)
        total_n = int(len(gdf))

        for feature in feature_cols:
            values = pd.to_numeric(gdf[feature], errors="coerce")
            valid = values.dropna().astype(float)
            n = int(valid.size)
            missing_rate = float(1.0 - (n / total_n)) if total_n else float("nan")

            rows.append(
                {
                    **group_map,
                    "feature_name": feature,
                    "group_var": group_var,
                    "group_value": group_value,
                    "group_n": total_n,
                    "n": n,
                    "mean": _safe_float(valid.mean()) if n else float("nan"),
                    "var": _safe_float(valid.var(ddof=1)) if n >= 2 else float("nan"),
                    "std": _safe_float(valid.std(ddof=1)) if n >= 2 else float("nan"),
                    "p25": _safe_float(valid.quantile(0.25)) if n else float("nan"),
                    "p50": _safe_float(valid.quantile(0.50)) if n else float("nan"),
                    "p75": _safe_float(valid.quantile(0.75)) if n else float("nan"),
                    "min": _safe_float(valid.min()) if n else float("nan"),
                    "max": _safe_float(valid.max()) if n else float("nan"),
                    "missing_rate": missing_rate,
                }
            )

    return pd.DataFrame(rows)


def _build_group_compare(
    df: pd.DataFrame,
    *,
    compare_by: str,
    compare_features: list[str],
) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    groups = sorted(df[compare_by].astype("string").dropna().unique().tolist())
    if len(groups) < 2:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "group_var",
                "group_a",
                "group_b",
                "n_a",
                "n_b",
                "mean_a",
                "mean_b",
                "median_a",
                "median_b",
                "diff_mean",
                "diff_median",
                "test_method",
                "p_value",
                "effect_name",
                "effect_size",
            ]
        )

    for feature in compare_features:
        values = pd.to_numeric(df[feature], errors="coerce")
        for group_a, group_b in itertools.combinations(groups, 2):
            x_a = values[df[compare_by] == group_a].dropna().astype(float)
            x_b = values[df[compare_by] == group_b].dropna().astype(float)
            n_a = int(len(x_a))
            n_b = int(len(x_b))

            mean_a = _safe_float(x_a.mean()) if n_a else float("nan")
            mean_b = _safe_float(x_b.mean()) if n_b else float("nan")
            median_a = _safe_float(x_a.median()) if n_a else float("nan")
            median_b = _safe_float(x_b.median()) if n_b else float("nan")

            if n_a > 0 and n_b > 0:
                _, p_value = stats.mannwhitneyu(x_a, x_b, alternative="two-sided")
                p_out = _safe_float(p_value)
            else:
                p_out = float("nan")

            out_rows.append(
                {
                    "feature_name": feature,
                    "group_var": compare_by,
                    "group_a": str(group_a),
                    "group_b": str(group_b),
                    "n_a": n_a,
                    "n_b": n_b,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "median_a": median_a,
                    "median_b": median_b,
                    "diff_mean": _safe_float(mean_a - mean_b) if n_a and n_b else float("nan"),
                    "diff_median": _safe_float(median_a - median_b) if n_a and n_b else float("nan"),
                    "test_method": "mannwhitney_u",
                    "p_value": p_out,
                    "effect_name": "cohen_d",
                    "effect_size": _cohens_d(x_a, x_b),
                }
            )
    return pd.DataFrame(out_rows)


def build_analysis_tables(
    *,
    run_id: str,
    artifacts_root: Path,
    analysis_dataset_path: Path,
    group_cols: list[str] | None = None,
    compare_by: str = "cohort_label",
    compare_features: list[str] | None = None,
    output_suffix: str = "",
) -> tuple[Path, Path, dict[str, Any]]:
    if not analysis_dataset_path.exists():
        raise FileNotFoundError(f"analysis dataset not found: {analysis_dataset_path}")

    group_cols = group_cols or DEFAULT_GROUP_COLS
    compare_features = compare_features or DEFAULT_COMPARE_FEATURES

    run_dir = artifacts_root / run_id
    analysis_dir = run_dir / "analysis_tables"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(analysis_dataset_path)
    group_cols_with_compare = list(group_cols)
    if compare_by not in group_cols_with_compare:
        group_cols_with_compare.append(compare_by)
    df = _ensure_group_columns(df, group_cols=group_cols_with_compare)

    feature_cols = _infer_feature_columns(df, group_cols=group_cols)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found in analysis_dataset.parquet")

    compare_features = [f for f in compare_features if f in feature_cols]
    if not compare_features:
        compare_features = feature_cols[: min(3, len(feature_cols))]

    feature_summary_df = _build_feature_summary(
        df,
        group_cols=group_cols,
        feature_cols=feature_cols,
    )
    group_compare_df = _build_group_compare(
        df,
        compare_by=compare_by,
        compare_features=compare_features,
    )

    safe_suffix = str(output_suffix or "").strip().replace(" ", "_")
    suffix_txt = f"_{safe_suffix}" if safe_suffix else ""
    feature_summary_path = analysis_dir / f"feature_summary{suffix_txt}.parquet"
    group_compare_path = analysis_dir / f"group_compare{suffix_txt}.parquet"
    feature_summary_df.to_parquet(feature_summary_path, index=False)
    group_compare_df.to_parquet(group_compare_path, index=False)

    summary = {
        "run_id": run_id,
        "output_suffix": safe_suffix,
        "analysis_dataset_rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "group_cols": group_cols,
        "compare_by": compare_by,
        "compare_features": compare_features,
        "feature_summary_rows": int(len(feature_summary_df)),
        "group_compare_rows": int(len(group_compare_df)),
    }
    summary_name = "analysis_tables_summary.json"
    if safe_suffix:
        summary_name = f"analysis_tables_summary_{safe_suffix}.json"
    (analysis_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return feature_summary_path, group_compare_path, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature summary and group-compare tables")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--artifacts-root",
        default=str(PROJECT_ROOT / "storage" / "artifacts"),
    )
    parser.add_argument("--analysis-dataset-parquet", default=None)
    parser.add_argument(
        "--group-cols",
        default="cohort_label,sex,age_bin",
        help="Comma-separated columns for feature summary grouping",
    )
    parser.add_argument("--compare-by", default="cohort_label")
    parser.add_argument(
        "--compare-features",
        default="mean_hr,rr_std,rr_mean",
        help="Comma-separated features for group comparison",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional output suffix, e.g. sensitivity_all",
    )
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
    analysis_dataset_path = (
        Path(args.analysis_dataset_parquet)
        if args.analysis_dataset_parquet
        else run_dir / "analysis_tables" / "analysis_dataset.parquet"
    )
    group_cols = [x.strip() for x in args.group_cols.split(",") if x.strip()]
    compare_features = [x.strip() for x in args.compare_features.split(",") if x.strip()]

    try:
        feature_summary_path, group_compare_path, summary = build_analysis_tables(
            run_id=args.run_id,
            artifacts_root=artifacts_root,
            analysis_dataset_path=analysis_dataset_path,
            group_cols=group_cols,
            compare_by=args.compare_by,
            compare_features=compare_features,
            output_suffix=str(args.output_suffix),
        )
    except Exception as exc:
        LOGGER.error("build_analysis_tables failed: %s", exc)
        return 1

    LOGGER.info("Wrote feature summary: %s", feature_summary_path)
    LOGGER.info("Wrote group compare: %s", group_compare_path)
    LOGGER.info("summary=%s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


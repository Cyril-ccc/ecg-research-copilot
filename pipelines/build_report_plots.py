"""
Build report-ready plots for one run.

Required outputs under artifacts/<run_id>/plots/:
- qc_pass_rate.png
- hr_distribution_by_group.png
- feature_boxplot_qtc_or_rrstd.png
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("build_report_plots")


def _normalize_group(series: pd.Series, default: str = "Other") -> pd.Series:
    out = series.astype("string").str.strip()
    low = out.str.lower()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    out = out.mask(low.isin({"unknown", "unk", "na", "n/a"}), default)
    return out.fillna(default)


def _parse_reason_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except (ValueError, SyntaxError):
                pass
        if ";" in text:
            parts = [p.strip() for p in text.split(";")]
            return [p for p in parts if p]
        if "," in text:
            parts = [p.strip() for p in text.split(",")]
            return [p for p in parts if p]
        return [text]
    return [str(value).strip()]


def _annotate_bars(ax: plt.Axes, bars: list[Any], values: list[float]) -> None:
    ymax = max(values) if values else 0.0
    offset = max(ymax * 0.02, 0.4)
    for rect, v in zip(bars, values, strict=False):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + offset,
            f"n={int(v)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_qc_pass_rate(
    *,
    qc_df: pd.DataFrame,
    out_path: Path,
    top_n: int = 8,
) -> dict[str, Any]:
    df = qc_df.copy()
    if "qc_pass" not in df.columns:
        raise RuntimeError("ecg_qc.parquet missing qc_pass column")

    pass_counts = (
        df["qc_pass"]
        .astype("boolean")
        .fillna(False)
        .map({True: "Pass", False: "Fail"})
        .value_counts()
        .reindex(["Pass", "Fail"], fill_value=0)
    )
    pass_values = [int(pass_counts["Pass"]), int(pass_counts["Fail"])]

    fail_df = df[df["qc_pass"] == False].copy()  # noqa: E712
    reason_counter: dict[str, int] = {}
    if "qc_reasons" in fail_df.columns:
        for raw in fail_df["qc_reasons"].tolist():
            reasons = _parse_reason_cell(raw)
            if not reasons:
                reasons = ["unspecified"]
            for r in reasons:
                reason_counter[r] = reason_counter.get(r, 0) + 1
    top_items = sorted(reason_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not top_items:
        top_items = [("unspecified", int(pass_values[1]))]

    reason_labels = [x[0] for x in top_items]
    reason_values = [int(x[1]) for x in top_items]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax0, ax1 = axes

    bars0 = ax0.bar(["Pass", "Fail"], pass_values, color=["#16a34a", "#dc2626"], alpha=0.92)
    ax0.set_title("QC Pass/Fail")
    ax0.set_xlabel("QC Result")
    ax0.set_ylabel("Record Count")
    ax0.grid(axis="y", alpha=0.25)
    _annotate_bars(ax0, list(bars0), pass_values)

    bars1 = ax1.barh(reason_labels, reason_values, color="#f59e0b", alpha=0.92)
    ax1.invert_yaxis()
    ax1.set_title(f"Top {min(top_n, len(reason_labels))} Fail Reasons")
    ax1.set_xlabel("Count")
    ax1.set_ylabel("Fail Reason")
    ax1.grid(axis="x", alpha=0.25)
    for rect, v in zip(bars1, reason_values, strict=False):
        ax1.text(rect.get_width() + max(max(reason_values) * 0.02, 0.4), rect.get_y() + rect.get_height() / 2, f"n={int(v)}", va="center", fontsize=9)

    fig.suptitle(f"QC Summary (Total n={len(df)})", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "total_n": int(len(df)),
        "pass_n": pass_values[0],
        "fail_n": pass_values[1],
        "top_fail_reasons": [{"reason": k, "n": v} for k, v in top_items],
    }


def plot_hr_distribution_by_group(
    *,
    analysis_df: pd.DataFrame,
    out_path: Path,
    group_col: str = "cohort_label",
    sex_col: str = "sex",
) -> dict[str, Any]:
    df = analysis_df.copy()
    if "mean_hr" not in df.columns:
        raise RuntimeError("analysis_dataset.parquet missing mean_hr column")

    df[group_col] = _normalize_group(df[group_col] if group_col in df.columns else pd.Series(["Unknown"] * len(df)))
    df[sex_col] = _normalize_group(df[sex_col] if sex_col in df.columns else pd.Series(["Unknown"] * len(df)))
    df["mean_hr"] = pd.to_numeric(df["mean_hr"], errors="coerce")
    df = df.dropna(subset=["mean_hr"]).copy()
    if df.empty:
        raise RuntimeError("No valid mean_hr values for plotting")

    sexes = sorted(df[sex_col].astype(str).unique().tolist())
    cohorts = sorted(df[group_col].astype(str).unique().tolist())
    informative_sexes = [s for s in sexes if s.lower() != "other"]
    use_sex_facets = len(informative_sexes) >= 2

    bins = min(18, max(8, int(np.sqrt(len(df)))))
    palette = ["#2563eb", "#ef4444", "#10b981", "#f59e0b", "#6b7280"]

    group_sizes: dict[str, int] = {}
    if use_sex_facets:
        n_rows = len(sexes)
        fig, axes = plt.subplots(n_rows, 1, figsize=(11, 4.2 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        for i, sex in enumerate(sexes):
            ax = axes[i]
            sdf = df[df[sex_col].astype(str) == sex]
            ax.set_title(f"{sex_col}={sex} (n={len(sdf)})")
            for j, cohort in enumerate(cohorts):
                g = sdf[sdf[group_col].astype(str) == cohort]["mean_hr"].dropna().astype(float)
                if g.empty:
                    continue
                label = f"{cohort} (n={len(g)})"
                group_sizes[f"{sex}|{cohort}"] = int(len(g))
                ax.hist(
                    g.values,
                    bins=bins,
                    alpha=0.45,
                    label=label,
                    color=palette[j % len(palette)],
                    edgecolor="white",
                    linewidth=0.4,
                )
            ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=9)
        axes[-1].set_xlabel("Mean Heart Rate (bpm)")
        title = f"Heart Rate Distribution by {group_col} x {sex_col} (n={len(df)})"
        display_mode = "cohort_x_sex"
    else:
        fig, ax = plt.subplots(1, 1, figsize=(11, 5.0), sharex=True)
        for j, cohort in enumerate(cohorts):
            g = df[df[group_col].astype(str) == cohort]["mean_hr"].dropna().astype(float)
            if g.empty:
                continue
            label = f"{cohort} (n={len(g)})"
            group_sizes[f"{cohort}"] = int(len(g))
            ax.hist(
                g.values,
                bins=bins,
                alpha=0.45,
                label=label,
                color=palette[j % len(palette)],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xlabel("Mean Heart Rate (bpm)")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=9)
        title = f"Heart Rate Distribution by {group_col} (n={len(df)})"
        display_mode = "cohort_only"

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "total_n": int(len(df)),
        "display_mode": display_mode,
        "title": title,
        "sex_levels": sexes,
        "cohort_levels": cohorts,
        "group_sizes": group_sizes,
    }


def _pick_feature_for_boxplot(df: pd.DataFrame, preferred: list[str]) -> str:
    for name in preferred:
        if name in df.columns and pd.to_numeric(df[name], errors="coerce").notna().any():
            return name
    numeric_cols: list[str] = []
    for col in df.columns:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().any():
            numeric_cols.append(col)
    if not numeric_cols:
        raise RuntimeError("No numeric feature columns available for boxplot")
    return numeric_cols[0]


def plot_feature_boxplot(
    *,
    analysis_df: pd.DataFrame,
    out_path: Path,
    group_col: str = "cohort_label",
    feature_preferred: list[str] | None = None,
) -> dict[str, Any]:
    df = analysis_df.copy()
    feature_preferred = feature_preferred or ["rr_std", "qtc", "qtc_ms"]
    feature_name = _pick_feature_for_boxplot(df, feature_preferred)

    df[group_col] = _normalize_group(df[group_col] if group_col in df.columns else pd.Series(["Unknown"] * len(df)))
    df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")
    df = df.dropna(subset=[feature_name]).copy()
    if df.empty:
        raise RuntimeError(f"No valid values for feature {feature_name}")

    groups = sorted(df[group_col].astype(str).unique().tolist())
    data = [df[df[group_col].astype(str) == g][feature_name].astype(float).values for g in groups]
    counts = [len(x) for x in data]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bp = ax.boxplot(
        data,
        tick_labels=groups,
        patch_artist=True,
        showmeans=True,
        meanline=False,
    )
    colors = ["#60a5fa", "#fca5a5", "#86efac", "#fde68a", "#c4b5fd"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.5)

    ymax = max(float(np.nanmax(x)) for x in data if len(x) > 0)
    ypad = max(abs(ymax) * 0.03, 0.02)
    for i, n in enumerate(counts, start=1):
        ax.text(i, ymax + ypad, f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"{feature_name} by {group_col} (n={len(df)})")
    ax.set_xlabel(group_col)
    ax.set_ylabel(feature_name)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "feature_name": feature_name,
        "total_n": int(len(df)),
        "group_sizes": {g: int(n) for g, n in zip(groups, counts, strict=False)},
    }


def build_report_plots(
    *,
    run_id: str,
    artifacts_root: Path,
    analysis_dataset_path: Path,
    qc_path: Path,
    fail_top_n: int = 8,
    feature_preferred: list[str] | None = None,
    group_col: str = "cohort_label",
) -> dict[str, Any]:
    run_dir = artifacts_root / run_id
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not analysis_dataset_path.exists():
        raise FileNotFoundError(f"analysis dataset missing: {analysis_dataset_path}")
    if not qc_path.exists():
        raise FileNotFoundError(f"qc parquet missing: {qc_path}")

    analysis_df = pd.read_parquet(analysis_dataset_path)
    qc_df = pd.read_parquet(qc_path)

    analysis_df["record_id"] = analysis_df["record_id"].astype("string")
    qc_df["record_id"] = qc_df["record_id"].astype("string")
    qc_on_run = qc_df[qc_df["record_id"].isin(set(analysis_df["record_id"].dropna().tolist()))].copy()
    if qc_on_run.empty:
        raise RuntimeError("No overlapping record_id between analysis_dataset and ecg_qc")

    qc_out = plots_dir / "qc_pass_rate.png"
    hr_out = plots_dir / "hr_distribution_by_group.png"
    box_out = plots_dir / "feature_boxplot_qtc_or_rrstd.png"

    qc_summary = plot_qc_pass_rate(qc_df=qc_on_run, out_path=qc_out, top_n=fail_top_n)
    hr_summary = plot_hr_distribution_by_group(
        analysis_df=analysis_df,
        out_path=hr_out,
        group_col=group_col,
    )
    box_summary = plot_feature_boxplot(
        analysis_df=analysis_df,
        out_path=box_out,
        group_col=group_col,
        feature_preferred=feature_preferred,
    )

    summary = {
        "run_id": run_id,
        "analysis_dataset_path": str(analysis_dataset_path),
        "qc_path": str(qc_path),
        "group_col": group_col,
        "plots": [
            "plots/qc_pass_rate.png",
            "plots/hr_distribution_by_group.png",
            "plots/feature_boxplot_qtc_or_rrstd.png",
        ],
        "qc_summary": qc_summary,
        "hr_summary": hr_summary,
        "box_summary": box_summary,
    }
    (plots_dir / "report_plots_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build report plots for one run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts-root", default=str(PROJECT_ROOT / "storage" / "artifacts"))
    parser.add_argument("--analysis-dataset-parquet", default=None)
    parser.add_argument("--qc-parquet", default=None)
    parser.add_argument("--fail-topn", type=int, default=8)
    parser.add_argument(
        "--feature-preferred",
        default="rr_std,qtc,qtc_ms",
        help="Comma-separated feature priority for boxplot",
    )
    parser.add_argument("--group-col", default="cohort_label")
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
    qc_path = Path(args.qc_parquet) if args.qc_parquet else run_dir / "ecg_qc.parquet"
    feature_preferred = [x.strip() for x in args.feature_preferred.split(",") if x.strip()]

    try:
        summary = build_report_plots(
            run_id=args.run_id,
            artifacts_root=artifacts_root,
            analysis_dataset_path=analysis_dataset_path,
            qc_path=qc_path,
            fail_top_n=max(1, int(args.fail_topn)),
            feature_preferred=feature_preferred,
            group_col=str(args.group_col),
        )
    except Exception as exc:
        LOGGER.error("build_report_plots failed: %s", exc)
        return 1

    LOGGER.info("Wrote report plots: %s", summary["plots"])
    LOGGER.info("box feature=%s", summary["box_summary"]["feature_name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

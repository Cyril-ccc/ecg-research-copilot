"""
Generate run-level report.md and run_metadata.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("generate_report")
PIPELINE_NAME = "report_pipeline"
PIPELINE_VERSION = "v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _to_md_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df[columns].iterrows():
        vals: list[str] = []
        for col in columns:
            v = row[col]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("NA")
                else:
                    vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(st.st_size),
        "modified_at": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256(path),
    }


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        pass
    try:
        git_dir = PROJECT_ROOT / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_path = git_dir / head.replace("ref: ", "", 1)
            commit = ref_path.read_text(encoding="utf-8").strip()
            return commit[:7] if commit else "unknown"
        return head[:7] if head else "unknown"
    except Exception:
        return "unknown"


def _display_group_value(value: Any) -> str:
    txt = str(value)
    if txt.lower() in {"unknown", "unk", "na", "n/a", "<na>", "nan", ""}:
        return "Other"
    return txt


def _infer_template_name(question: str, params: dict[str, Any], cohort_labels: list[str]) -> str:
    if isinstance(params.get("template_name"), str):
        return str(params["template_name"])
    q = (question or "").lower()
    for name in ["electrolyte_hyperkalemia", "diagnosis_icd", "medication_exposure"]:
        if name in q:
            return name
    for c in cohort_labels:
        if c in {"electrolyte_hyperkalemia", "diagnosis_icd", "medication_exposure"}:
            return c
    return "unknown_template"


def _cohort_table_hint(template_name: str) -> list[str]:
    if template_name == "electrolyte_hyperkalemia":
        return ["mimiciv.labevents", "mimiciv.d_labitems"]
    if template_name == "diagnosis_icd":
        return ["mimiciv.diagnoses_icd", "mimiciv.admissions"]
    if template_name == "medication_exposure":
        return ["mimiciv.prescriptions|mimiciv.pharmacy"]
    return ["unknown"]


def _parse_question_params(
    *,
    run_dir: Path,
    question_arg: str | None,
    params_json_arg: str | None,
) -> tuple[str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    params_file = run_dir / "params.json"
    params_payload = _read_json(params_file)

    question = question_arg or str(params_payload.get("question", "")).strip()
    if not question:
        question = "N/A (params.json missing question)"
        warnings.append("question missing; filled with placeholder")

    params: dict[str, Any] = {}
    if isinstance(params_payload.get("params"), dict):
        params = dict(params_payload["params"])
    if params_json_arg:
        try:
            parsed = json.loads(params_json_arg)
            if isinstance(parsed, dict):
                params.update(parsed)
            else:
                warnings.append("--params-json is not a JSON object; ignored")
        except json.JSONDecodeError:
            warnings.append("--params-json parse failed; ignored")
    return question, params, warnings


def generate_report(
    *,
    run_id: str,
    artifacts_root: Path,
    question_arg: str | None = None,
    params_json_arg: str | None = None,
    qc_path_override: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    t0 = time.perf_counter()
    created_at = _utc_now_iso()
    warnings: list[str] = []

    run_dir = artifacts_root / run_id
    analysis_dir = run_dir / "analysis_tables"
    plots_dir = run_dir / "plots"

    analysis_dataset_path = analysis_dir / "analysis_dataset.parquet"
    analysis_dataset_all_path = analysis_dir / "analysis_dataset_all.parquet"
    feature_summary_path = analysis_dir / "feature_summary.parquet"
    group_compare_path = analysis_dir / "group_compare.parquet"
    sensitivity_group_compare_path = analysis_dir / "group_compare_sensitivity_all.parquet"
    analysis_summary_path = analysis_dir / "analysis_dataset_summary.json"
    analysis_summary_all_path = analysis_dir / "analysis_dataset_summary_all.json"
    analysis_tables_summary_path = analysis_dir / "analysis_tables_summary.json"
    report_plots_summary_path = plots_dir / "report_plots_summary.json"
    window_summary_path = run_dir / "ecg_window_summary.json"
    qc_path = qc_path_override if qc_path_override else run_dir / "ecg_qc.parquet"

    for p in [analysis_dataset_path, feature_summary_path, group_compare_path]:
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")

    analysis_df = pd.read_parquet(analysis_dataset_path)
    feature_summary = pd.read_parquet(feature_summary_path).reset_index(drop=True)
    group_compare = pd.read_parquet(group_compare_path).reset_index(drop=True)
    feature_summary["__row"] = np.arange(1, len(feature_summary) + 1)
    group_compare["__row"] = np.arange(1, len(group_compare) + 1)

    analysis_summary = _read_json(analysis_summary_path)
    analysis_summary_all = _read_json(analysis_summary_all_path)
    analysis_tables_summary = _read_json(analysis_tables_summary_path)
    plots_summary = _read_json(report_plots_summary_path)
    window_summary = _read_json(window_summary_path)

    analysis_df_all = pd.read_parquet(analysis_dataset_all_path) if analysis_dataset_all_path.exists() else pd.DataFrame()

    question, params, warn_qp = _parse_question_params(
        run_dir=run_dir,
        question_arg=question_arg,
        params_json_arg=params_json_arg,
    )
    warnings.extend(warn_qp)

    cohort_labels = (
        analysis_df["cohort_label"].astype("string").fillna("Other").astype(str).value_counts().index.tolist()
        if "cohort_label" in analysis_df.columns
        else []
    )
    template_name = _infer_template_name(question=question, params=params, cohort_labels=cohort_labels)
    cohort_tables = _cohort_table_hint(template_name)

    time_keys = [k for k in params.keys() if any(s in str(k).lower() for s in ["start", "end", "window", "hour", "time"])]
    cond_keys = [k for k in params.keys() if k not in time_keys]
    cond_summary = ", ".join(f"{k}={params[k]}" for k in cond_keys) if cond_keys else "N/A"
    time_summary = ", ".join(f"{k}={params[k]}" for k in time_keys) if time_keys else "not specified"

    qc_total = int(plots_summary.get("qc_summary", {}).get("total_n", 0))
    qc_pass = int(plots_summary.get("qc_summary", {}).get("pass_n", 0))
    qc_fail = int(plots_summary.get("qc_summary", {}).get("fail_n", 0))
    if qc_total == 0 and qc_path.exists():
        qc_df = pd.read_parquet(qc_path)
        if "qc_pass" in qc_df.columns:
            qc_total = int(len(qc_df))
            qc_pass = int((qc_df["qc_pass"] == True).sum())  # noqa: E712
            qc_fail = int((qc_df["qc_pass"] == False).sum())  # noqa: E712
    qc_pass_rate = (100.0 * qc_pass / qc_total) if qc_total > 0 else float("nan")
    fail_top3 = plots_summary.get("qc_summary", {}).get("top_fail_reasons", [])[:3]

    preferred = ["mean_hr", "rr_std", "rr_mean"]

    def _build_key_compare_df(compare_df: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
        if compare_df.empty:
            return pd.DataFrame(
                columns=[
                    "feature",
                    "groups",
                    "n_a/n_b",
                    "diff_mean",
                    "diff_median",
                    "p_value",
                    "cohen_d",
                    "source",
                ]
            )

        cmp = compare_df.reset_index(drop=True).copy()
        cmp["__row"] = np.arange(1, len(cmp) + 1)

        key_rows: list[pd.Series] = []
        used_idx: set[int] = set()
        for feat in preferred:
            sub = cmp[cmp["feature_name"] == feat].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["p_value", "__row"], na_position="last")
            row = sub.iloc[0]
            key_rows.append(row)
            used_idx.add(int(row["__row"]))
        if len(key_rows) < 3:
            rem = cmp[~cmp["__row"].isin(used_idx)].copy()
            rem = rem.sort_values(["p_value", "__row"], na_position="last")
            for _, r in rem.head(3 - len(key_rows)).iterrows():
                key_rows.append(r)

        out = pd.DataFrame(key_rows).copy()
        if out.empty:
            return pd.DataFrame(
                columns=[
                    "feature",
                    "groups",
                    "n_a/n_b",
                    "diff_mean",
                    "diff_median",
                    "p_value",
                    "cohen_d",
                    "source",
                ]
            )

        out["groups"] = (
            out["group_a"].map(_display_group_value).astype(str)
            + " vs "
            + out["group_b"].map(_display_group_value).astype(str)
        )
        out["n_a_n_b"] = out["n_a"].astype(str) + " / " + out["n_b"].astype(str)
        out["source"] = out["__row"].apply(lambda x: f"{source_name}: row {int(x)}")
        out = out[
            [
                "feature_name",
                "groups",
                "n_a_n_b",
                "diff_mean",
                "diff_median",
                "p_value",
                "effect_size",
                "source",
            ]
        ].rename(
            columns={
                "feature_name": "feature",
                "n_a_n_b": "n_a/n_b",
                "effect_size": "cohen_d",
            }
        )
        return out

    key_cmp_df = _build_key_compare_df(group_compare, source_name="group_compare.parquet")
    sensitivity_raw = (
        pd.read_parquet(sensitivity_group_compare_path)
        if sensitivity_group_compare_path.exists()
        else pd.DataFrame()
    )
    sensitivity_cmp_df = _build_key_compare_df(
        sensitivity_raw,
        source_name="group_compare_sensitivity_all.parquet",
    )

    fs_show = feature_summary.copy()
    for c in ["feature_name", "group_value", "n", "mean", "std", "p50", "missing_rate", "__row"]:
        if c not in fs_show.columns:
            fs_show[c] = np.nan
    fs_show = fs_show.sort_values(["feature_name", "group_n", "__row"], ascending=[True, False, True])
    fs_show = fs_show[fs_show["feature_name"].isin(preferred)].head(6).copy()
    fs_show["group_value"] = fs_show["group_value"].astype("string").str.replace("Unknown", "Other", regex=False)
    fs_show["group_value"] = fs_show["group_value"].str.replace("|", ", ", regex=False)
    fs_show["source"] = fs_show["__row"].apply(lambda x: f"feature_summary.parquet: row {int(x)}")
    fs_show = fs_show[
        ["feature_name", "group_value", "n", "mean", "std", "p50", "missing_rate", "source"]
    ].rename(columns={"feature_name": "feature", "group_value": "group"})

    missing_max = _safe_float(feature_summary["missing_rate"].max()) if "missing_rate" in feature_summary.columns else float("nan")
    min_group_n = int(feature_summary["group_n"].min()) if "group_n" in feature_summary.columns and not feature_summary.empty else 0
    small_group_count = (
        int((feature_summary["group_n"] < 5).sum()) if "group_n" in feature_summary.columns and not feature_summary.empty else 0
    )

    cohort_counts = (
        analysis_df["cohort_label"]
        .astype("string")
        .fillna("Other")
        .astype(str)
        .map(_display_group_value)
        .value_counts()
        .to_dict()
        if "cohort_label" in analysis_df.columns
        else {}
    )
    sex_counts = (
        analysis_df["sex"]
        .astype("string")
        .fillna("Other")
        .astype(str)
        .map(_display_group_value)
        .value_counts()
        .to_dict()
        if "sex" in analysis_df.columns
        else {}
    )

    key_number_lines: list[str] = []
    for _, r in key_cmp_df.iterrows():
        key_number_lines.append(
            (
                f"- `{r['feature']}`: diff_mean={_safe_float(r['diff_mean']):.4g}, "
                f"diff_median={_safe_float(r['diff_median']):.4g}, "
                f"p={_safe_float(r['p_value']):.4g}, d={_safe_float(r['cohen_d']):.4g} "
                f"[source: {r['source']}]"
            )
        )

    analysis_mode = str(analysis_summary.get("analysis_mode", ""))
    rows_all = int(analysis_summary.get("rows_all", len(analysis_df))) if str(analysis_summary.get("rows_all", "")).strip() else int(len(analysis_df))
    pair_status_counts_all = analysis_summary.get("pair_status_counts_all")
    if not isinstance(pair_status_counts_all, dict) or not pair_status_counts_all:
        pair_status_counts_all = analysis_summary_all.get("pair_status_counts", {}) if isinstance(analysis_summary_all, dict) else {}

    event_counts = window_summary.get("event_counts", {}) if isinstance(window_summary, dict) else {}
    if not isinstance(event_counts, dict):
        event_counts = {}
    event_both = int(event_counts.get("both", 0))
    event_pre_only = int(event_counts.get("pre_only", 0))
    event_post_only = int(event_counts.get("post_only", 0))

    sensitivity_rows = int(len(sensitivity_raw)) if not sensitivity_raw.empty else 0
    sensitivity_table_md = _to_md_table(
        sensitivity_cmp_df,
        ["feature", "groups", "n_a/n_b", "diff_mean", "diff_median", "p_value", "cohen_d", "source"],
    )

    plot_files = plots_summary.get("plots", [])
    if not isinstance(plot_files, list):
        plot_files = []
    plot_files = [str(x) for x in plot_files][:3]

    report_md = f"""# ECG Report

- run_id: `{run_id}`
- generated_at: `{created_at}`
- git_commit: `{_git_commit()}`

## Question
- question: {question}
- params: `{json.dumps(params, ensure_ascii=True, sort_keys=True)}`

## Cohort Definition
- template: `{template_name}`
- tables: `{", ".join(cohort_tables)}`
- key_conditions: `{cond_summary}`
- time_window: `{time_summary}`

## Cohort Overview
- rows in main analysis dataset: `{len(analysis_df)}`
- rows in all-window dataset (sensitivity pool): `{rows_all}`
- cohort counts: `{cohort_counts}`
- sex counts: `{sex_counts}`
- medication window event coverage (all exposures): `both={event_both}, pre_only={event_pre_only}, post_only={event_post_only}`
- pair status counts (all selected rows): `{pair_status_counts_all}`

## Methods
- pipeline chain: `build_cohort -> ecg_qc -> ecg_features -> assemble_analysis_dataset -> build_analysis_tables -> build_report_plots -> generate_report`
- main analysis table source: `analysis_tables/analysis_dataset.parquet`
- sensitivity dataset source: `analysis_tables/analysis_dataset_all.parquet` (if exists)
- main analysis mode: `{analysis_mode if analysis_mode else "default"}`
- group comparison method: `mannwhitney_u` with effect size `cohen_d` on selected ECG features

## Data & QC
- ECG records: `{qc_total}`
- QC pass: `{qc_pass}` / `{qc_total}` ({qc_pass_rate:.2f}%)
- QC fail: `{qc_fail}`
- fail reasons Top3: `{fail_top3}`

## Results
### Key Numbers
{chr(10).join(key_number_lines) if key_number_lines else "- No group comparison rows available."}

### Group Comparison (Main, Paired)
{_to_md_table(
    key_cmp_df,
    ["feature", "groups", "n_a/n_b", "diff_mean", "diff_median", "p_value", "cohen_d", "source"],
)}

### Sensitivity (All Window-Eligible, Non-Paired)
- source table: `analysis_tables/group_compare_sensitivity_all.parquet`
- rows: `{sensitivity_rows}`
{sensitivity_table_md if sensitivity_rows > 0 else "_No sensitivity comparison rows available._"}

### Feature Summary (Slice)
{_to_md_table(
    fs_show,
    ["feature", "group", "n", "mean", "std", "p50", "missing_rate", "source"],
)}

### Figures
{chr(10).join(f"![{Path(p).stem}]({p})" for p in plot_files) if plot_files else "_No plots found._"}

## Limitations
- Missingness: max `missing_rate={missing_max:.3f}` across feature summaries [source: feature_summary.parquet].
- Small-sample rule: groups with `n<5` should be hidden or merged into `Other`; current `min_group_n={min_group_n}`, triggered rows=`{small_group_count}`.
- Paired-only main analysis may over-represent heavily monitored patients; event coverage was `both={event_both}, pre_only={event_pre_only}, post_only={event_post_only}`.
- Time alignment & leakage: index-time to ECG-time alignment may be imperfect; temporal leakage risk remains if windows are not strictly enforced.
- Non-random missingness & confounding: missing covariates and unmeasured confounders may bias between-group comparisons.
"""

    report_path = run_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    finished_at = _utc_now_iso()
    duration_seconds = float(time.perf_counter() - t0)
    metadata_path = run_dir / "run_metadata.json"
    output_files = [report_path, metadata_path]

    input_paths = [
        analysis_dataset_path,
        analysis_summary_path,
        analysis_tables_summary_path,
        feature_summary_path,
        group_compare_path,
        report_plots_summary_path,
        qc_path,
        run_dir / "params.json",
    ]
    optional_inputs = [
        analysis_dataset_all_path,
        analysis_summary_all_path,
        sensitivity_group_compare_path,
        analysis_dir / "feature_summary_sensitivity_all.parquet",
        analysis_dir / "analysis_tables_summary_sensitivity_all.json",
        window_summary_path,
    ]
    for path in optional_inputs:
        if path.exists():
            input_paths.append(path)
    for p in plot_files:
        input_paths.append(run_dir / p)

    qv = sorted(analysis_df["qc_version"].dropna().astype(str).unique().tolist()) if "qc_version" in analysis_df.columns else []
    fv = sorted(analysis_df["feature_version"].dropna().astype(str).unique().tolist()) if "feature_version" in analysis_df.columns else []

    status = "success" if not warnings else "partial"
    metadata = {
        "pipeline_name": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "status": status,
        "warnings": warnings,
        "question": question,
        "params": params,
        "qc_version": qv,
        "feature_version": fv,
        "git_commit": _git_commit(),
        "input_files": [_file_meta(p) for p in input_paths],
        "output_files": [_file_meta(p) for p in output_files],
        "key_numbers": key_cmp_df.to_dict(orient="records"),
        "analysis_summary": analysis_summary,
        "analysis_tables_summary": analysis_tables_summary,
        "plots_summary": plots_summary,
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    metadata["output_files"] = [_file_meta(p) for p in output_files]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
    return report_path, metadata_path, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report.md and run_metadata.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts-root", default=str(PROJECT_ROOT / "storage" / "artifacts"))
    parser.add_argument("--question", default=None)
    parser.add_argument("--params-json", default=None, help="JSON text to override/extend params")
    parser.add_argument("--qc-parquet", default=None)
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

    qc_override = Path(args.qc_parquet) if args.qc_parquet else None
    try:
        report_path, metadata_path, meta = generate_report(
            run_id=args.run_id,
            artifacts_root=Path(args.artifacts_root),
            question_arg=args.question,
            params_json_arg=args.params_json,
            qc_path_override=qc_override,
        )
    except Exception as exc:
        LOGGER.error("generate_report failed: %s", exc)
        return 1

    LOGGER.info("Wrote report: %s", report_path)
    LOGGER.info("Wrote metadata: %s", metadata_path)
    LOGGER.info("status=%s warnings=%d", meta["status"], len(meta.get("warnings", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


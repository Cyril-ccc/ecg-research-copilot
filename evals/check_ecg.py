"""
Validate ECG QC and feature artifacts for eval runs.

Checks per gold_id/run_id:
- ecg_qc.parquet exists and contains qc_pass/qc_reasons
- qc pass rate meets threshold (supports per-gold threshold)
- fail reason TopN exists (if there are failed rows)
- feature sanity: HR range, rr_std non-negative, missing-rate thresholds
- drift (v1): compare feature mean/quantiles with baseline JSON

Baseline format:
- evals/baselines/<gold_id>.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger("eval_check_ecg")

DEFAULT_FEATURE_COLUMNS = ["mean_hr", "rr_mean", "rr_std"]
DEFAULT_MISSING_RATE_MAX = 0.20

DEFAULT_DRIFT_THRESHOLDS: dict[str, dict[str, float]] = {
    "mean_hr": {
        "mean_abs_max": 15.0,
        "p10_abs_max": 20.0,
        "p50_abs_max": 15.0,
        "p90_abs_max": 20.0,
    },
    "rr_mean": {
        "mean_abs_max": 0.25,
        "p10_abs_max": 0.30,
        "p50_abs_max": 0.25,
        "p90_abs_max": 0.30,
    },
    "rr_std": {
        "mean_abs_max": 0.20,
        "p10_abs_max": 0.20,
        "p50_abs_max": 0.20,
        "p90_abs_max": 0.20,
    },
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    return out


def _read_gold(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"gold yaml must be a list: {path}")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"gold entry #{i} is not an object")
        out.append(item)
    return out


def _load_run_id(eval_runs_root: Path, gold_id: str) -> str | None:
    p = eval_runs_root / gold_id / "run_id.txt"
    if not p.exists():
        return None
    run_id = p.read_text(encoding="utf-8").strip()
    return run_id or None


def _as_reason_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass
        return [txt]

    if isinstance(value, (list, tuple, set, np.ndarray)):
        out: list[str] = []
        for x in value:
            s = str(x).strip()
            if s and s.lower() != "nan":
                out.append(s)
        return out

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    return [s]


def _pick_gold_ids(gold_items: list[dict[str, Any]], requested: set[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return gold_items
    out = [g for g in gold_items if str(g.get("id", "")).strip() in requested]
    return out


def _feature_columns(expectations: dict[str, Any], args: argparse.Namespace) -> list[str]:
    feat_cfg = expectations.get("features") if isinstance(expectations.get("features"), dict) else {}
    cols = feat_cfg.get("sanity_columns")
    if isinstance(cols, list) and cols:
        return [str(c).strip() for c in cols if str(c).strip()]
    return [str(c).strip() for c in str(args.feature_columns).split(",") if str(c).strip()]


def _missing_thresholds(
    expectations: dict[str, Any],
    columns: list[str],
    default_max: float,
) -> dict[str, float]:
    feat_cfg = expectations.get("features") if isinstance(expectations.get("features"), dict) else {}
    out = {c: float(default_max) for c in columns}

    common_max = feat_cfg.get("missing_rate_max")
    if isinstance(common_max, (int, float)):
        for c in columns:
            out[c] = min(1.0, max(0.0, float(common_max)))

    per_col = feat_cfg.get("feature_missing_rate_max")
    if isinstance(per_col, dict):
        for k, v in per_col.items():
            col = str(k)
            if col in out and isinstance(v, (int, float)):
                out[col] = min(1.0, max(0.0, float(v)))

    return out


def _drift_thresholds_for_feature(
    expectations: dict[str, Any],
    feature: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    drift_cfg = expectations.get("drift") if isinstance(expectations.get("drift"), dict) else {}
    base = dict(DEFAULT_DRIFT_THRESHOLDS.get(feature, {}))
    if not base:
        fallback = float(args.drift_abs_default)
        base = {
            "mean_abs_max": fallback,
            "p10_abs_max": fallback,
            "p50_abs_max": fallback,
            "p90_abs_max": fallback,
        }

    all_override = drift_cfg.get("thresholds")
    if isinstance(all_override, dict):
        feat_override = all_override.get(feature)
        if isinstance(feat_override, dict):
            for k in ["mean_abs_max", "p10_abs_max", "p50_abs_max", "p90_abs_max"]:
                if isinstance(feat_override.get(k), (int, float)):
                    base[k] = float(feat_override[k])

    return base


def _compute_feature_summary(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if len(valid) == 0:
            out[col] = {
                "count": 0,
                "mean": None,
                "p10": None,
                "p50": None,
                "p90": None,
            }
            continue

        out[col] = {
            "count": int(len(valid)),
            "mean": float(valid.mean()),
            "p10": float(valid.quantile(0.10)),
            "p50": float(valid.quantile(0.50)),
            "p90": float(valid.quantile(0.90)),
        }
    return out


def _baseline_path(baseline_dir: Path, gold_id: str) -> Path:
    return baseline_dir / f"{gold_id}.json"


def _check_one(
    *,
    gold: dict[str, Any],
    run_id: str | None,
    artifacts_root: Path,
    baseline_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gold_id = str(gold.get("id", "")).strip() or "UNKNOWN"
    name = str(gold.get("name", "")).strip()
    template_name = str(gold.get("cohort_template", "")).strip()
    expectations = gold.get("expectations") if isinstance(gold.get("expectations"), dict) else {}

    result: dict[str, Any] = {
        "gold_id": gold_id,
        "name": name,
        "template_name": template_name,
        "run_id": run_id,
        "pass": True,
        "reasons": [],
        "checks": {},
        "metrics": {},
    }

    def fail(msg: str) -> None:
        result["pass"] = False
        result["reasons"].append(msg)

    if not run_id:
        fail("missing run_id.txt")
        result["checks"]["qc_file_exists"] = False
        result["checks"]["feature_file_exists"] = False
        return result

    run_dir = artifacts_root / run_id
    qc_path = run_dir / "ecg_qc.parquet"
    feat_path = run_dir / "ecg_features.parquet"
    result["qc_path"] = str(qc_path)
    result["feature_path"] = str(feat_path)

    if not qc_path.exists():
        result["checks"]["qc_file_exists"] = False
        fail("ecg_qc.parquet not found")
        return result
    result["checks"]["qc_file_exists"] = True

    if not feat_path.exists():
        result["checks"]["feature_file_exists"] = False
        fail("ecg_features.parquet not found")
        return result
    result["checks"]["feature_file_exists"] = True

    try:
        qc_df = pd.read_parquet(qc_path)
    except Exception as exc:
        fail(f"failed to read ecg_qc.parquet: {exc}")
        return result

    try:
        feat_df = pd.read_parquet(feat_path)
    except Exception as exc:
        fail(f"failed to read ecg_features.parquet: {exc}")
        return result

    result["metrics"]["qc_rows"] = int(len(qc_df))
    result["metrics"]["feature_rows"] = int(len(feat_df))

    # QC checks
    missing_qc_cols = [c for c in ["qc_pass", "qc_reasons"] if c not in qc_df.columns]
    result["checks"]["qc_required_columns_missing"] = missing_qc_cols
    if missing_qc_cols:
        fail(f"qc columns missing: {missing_qc_cols}")

    qc_pass_rate = float("nan")
    failed_n = 0
    top_fail_reasons: list[dict[str, Any]] = []
    if "qc_pass" in qc_df.columns:
        pass_mask = qc_df["qc_pass"] == True  # noqa: E712
        qc_pass_rate = float(pass_mask.mean()) if len(qc_df) else float("nan")
        failed_n = int((~pass_mask).sum()) if len(qc_df) else 0
        result["metrics"]["qc_pass_rate"] = qc_pass_rate
        result["metrics"]["qc_failed_n"] = failed_n

        qc_cfg = expectations.get("qc") if isinstance(expectations.get("qc"), dict) else {}
        pass_rate_min = qc_cfg.get("pass_rate_min")
        if not isinstance(pass_rate_min, (int, float)):
            pass_rate_min = float(args.qc_pass_rate_min_default)
        pass_rate_min = float(pass_rate_min)

        result["checks"]["qc_pass_rate"] = {
            "actual": qc_pass_rate,
            "min": pass_rate_min,
        }
        if np.isnan(qc_pass_rate) or qc_pass_rate < pass_rate_min:
            fail(f"qc pass rate drop: actual={qc_pass_rate:.4f} < min={pass_rate_min:.4f}")

        if "qc_reasons" in qc_df.columns:
            counter: Counter[str] = Counter()
            non_empty_reason_rows = 0
            fail_rows = qc_df.loc[~pass_mask, "qc_reasons"] if len(qc_df) else []
            for raw in fail_rows:
                reasons = _as_reason_list(raw)
                if reasons:
                    non_empty_reason_rows += 1
                    counter.update(reasons)

            top_n = max(1, int(args.qc_fail_topn))
            top_fail_reasons = [
                {"reason": reason, "count": int(cnt)}
                for reason, cnt in counter.most_common(top_n)
            ]
            result["checks"]["qc_fail_reasons_topn"] = top_fail_reasons
            result["metrics"]["qc_fail_reason_non_empty_rows"] = non_empty_reason_rows

            if failed_n > 0 and non_empty_reason_rows == 0:
                fail("qc fail reason TopN empty: failed rows exist but qc_reasons are all empty")
            if failed_n > 0 and len(top_fail_reasons) == 0:
                fail("qc fail reason TopN missing")

    # Feature sanity checks
    feature_cols = _feature_columns(expectations, args)
    missing_feature_cols = [c for c in feature_cols if c not in feat_df.columns]
    result["checks"]["feature_required_columns_missing"] = missing_feature_cols
    if missing_feature_cols:
        fail(f"feature columns missing: {missing_feature_cols}")

    hr_min = _safe_float((expectations.get("features") or {}).get("mean_hr_min"), default=float(args.hr_min))
    hr_max = _safe_float((expectations.get("features") or {}).get("mean_hr_max"), default=float(args.hr_max))

    if "mean_hr" in feat_df.columns:
        hr = pd.to_numeric(feat_df["mean_hr"], errors="coerce")
        hr_valid = hr.dropna()
        hr_invalid_n = int(((hr_valid < hr_min) | (hr_valid > hr_max)).sum()) if len(hr_valid) else 0
        result["checks"]["mean_hr_range"] = {
            "min": hr_min,
            "max": hr_max,
            "invalid_n": hr_invalid_n,
            "valid_n": int(len(hr_valid)),
        }
        if hr_invalid_n > 0:
            fail(f"HR out of range [{hr_min}, {hr_max}] count={hr_invalid_n}")

    if "rr_std" in feat_df.columns:
        rr_std = pd.to_numeric(feat_df["rr_std"], errors="coerce")
        rr_std_valid = rr_std.dropna()
        rr_std_neg_n = int((rr_std_valid < 0).sum()) if len(rr_std_valid) else 0
        result["checks"]["rr_std_non_negative"] = {
            "negative_n": rr_std_neg_n,
            "valid_n": int(len(rr_std_valid)),
        }
        if rr_std_neg_n > 0:
            fail(f"rr_std contains negative values count={rr_std_neg_n}")

    miss_thresholds = _missing_thresholds(
        expectations,
        columns=feature_cols,
        default_max=float(args.feature_missing_rate_max_default),
    )
    miss_report: dict[str, Any] = {}
    for col, threshold in miss_thresholds.items():
        if col not in feat_df.columns:
            miss_rate = 1.0
        else:
            miss_rate = float(feat_df[col].isna().mean()) if len(feat_df) else 1.0
        miss_report[col] = {
            "missing_rate": miss_rate,
            "max": float(threshold),
        }
        if miss_rate > float(threshold):
            fail(f"feature missing rate high: {col}={miss_rate:.4f} > {float(threshold):.4f}")
    result["checks"]["feature_missing_rate_thresholds"] = miss_report

    # Drift checks (baseline compare)
    drift_cfg = expectations.get("drift") if isinstance(expectations.get("drift"), dict) else {}
    drift_enabled = bool(drift_cfg.get("enabled", True)) and (not bool(args.skip_drift))
    require_baseline = bool(drift_cfg.get("require_baseline", False)) or bool(args.fail_on_missing_baseline)

    baseline_path = _baseline_path(baseline_dir, gold_id)
    current_summary = _compute_feature_summary(feat_df, feature_cols)
    result["metrics"]["feature_summary"] = current_summary

    drift_result: dict[str, Any] = {
        "enabled": drift_enabled,
        "baseline_path": str(baseline_path),
        "status": "skipped",
        "violations": [],
    }

    baseline_obj: dict[str, Any] | None = None
    if baseline_path.exists():
        try:
            baseline_obj = json.loads(baseline_path.read_text(encoding="utf-8"))
        except Exception as exc:
            drift_result["status"] = "baseline_read_error"
            drift_result["reason"] = str(exc)
            if require_baseline:
                fail(f"baseline read error: {exc}")

    if drift_enabled:
        if baseline_obj is None:
            drift_result["status"] = "missing_baseline"
            if require_baseline:
                fail("baseline missing for drift check")
        else:
            base_summary = baseline_obj.get("feature_summary") if isinstance(baseline_obj.get("feature_summary"), dict) else {}
            violations: list[dict[str, Any]] = []
            for col in feature_cols:
                cur = current_summary.get(col)
                base = base_summary.get(col) if isinstance(base_summary.get(col), dict) else None
                if not isinstance(cur, dict) or not isinstance(base, dict):
                    continue
                thresh = _drift_thresholds_for_feature(expectations, col, args)
                for key, max_key in [
                    ("mean", "mean_abs_max"),
                    ("p10", "p10_abs_max"),
                    ("p50", "p50_abs_max"),
                    ("p90", "p90_abs_max"),
                ]:
                    cur_val = cur.get(key)
                    base_val = base.get(key)
                    if cur_val is None or base_val is None:
                        continue
                    diff = abs(float(cur_val) - float(base_val))
                    max_allowed = float(thresh.get(max_key, args.drift_abs_default))
                    if diff > max_allowed:
                        violations.append(
                            {
                                "feature": col,
                                "stat": key,
                                "current": float(cur_val),
                                "baseline": float(base_val),
                                "abs_diff": diff,
                                "max": max_allowed,
                            }
                        )
            drift_result["violations"] = violations
            drift_result["status"] = "failed" if violations else "passed"
            if violations:
                preview = violations[:3]
                fail(f"distribution drift detected: {preview}")

    if args.write_baseline:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_payload = {
            "gold_id": gold_id,
            "name": name,
            "template_name": template_name,
            "run_id": run_id,
            "generated_at": _utc_now_iso(),
            "qc_pass_rate": qc_pass_rate,
            "feature_summary": current_summary,
        }
        baseline_path.write_text(json.dumps(baseline_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        drift_result["baseline_written"] = True

    result["checks"]["drift"] = drift_result
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ECG QC/features for eval runs")
    parser.add_argument("--gold", default="evals/gold_questions.yaml")
    parser.add_argument("--eval-runs-root", default="eval_runs")
    parser.add_argument("--artifacts-root", default="storage/artifacts")
    parser.add_argument("--output", default="eval_ecg_results.json")
    parser.add_argument("--gold-ids", default="", help="comma-separated subset of gold IDs")

    parser.add_argument("--qc-pass-rate-min-default", type=float, default=0.60)
    parser.add_argument("--qc-fail-topn", type=int, default=5)

    parser.add_argument("--hr-min", type=float, default=30.0)
    parser.add_argument("--hr-max", type=float, default=180.0)
    parser.add_argument("--feature-columns", default=",".join(DEFAULT_FEATURE_COLUMNS))
    parser.add_argument("--feature-missing-rate-max-default", type=float, default=DEFAULT_MISSING_RATE_MAX)

    parser.add_argument("--baseline-dir", default="evals/baselines")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--skip-drift", action="store_true")
    parser.add_argument("--fail-on-missing-baseline", action="store_true")
    parser.add_argument("--drift-abs-default", type=float, default=10.0)

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

    gold_path = Path(args.gold).resolve()
    eval_runs_root = Path(args.eval_runs_root).resolve()
    artifacts_root = Path(args.artifacts_root).resolve()
    baseline_dir = Path(args.baseline_dir).resolve()
    output_path = Path(args.output).resolve()

    gold_items = _read_gold(gold_path)
    requested_ids = {x.strip() for x in str(args.gold_ids).split(",") if x.strip()} or None
    selected = _pick_gold_ids(gold_items, requested=requested_ids)

    if not selected:
        LOGGER.error("no gold items selected")
        return 2

    results: list[dict[str, Any]] = []
    for gold in selected:
        gold_id = str(gold.get("id", "")).strip() or "UNKNOWN"
        run_id = _load_run_id(eval_runs_root, gold_id)
        res = _check_one(
            gold=gold,
            run_id=run_id,
            artifacts_root=artifacts_root,
            baseline_dir=baseline_dir,
            args=args,
        )
        results.append(res)

    passed = sum(1 for r in results if r.get("pass") is True)
    failed = len(results) - passed
    out = {
        "generated_at": _utc_now_iso(),
        "gold_total": len(gold_items),
        "selected": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    output_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    LOGGER.info(
        "check_ecg done selected=%d passed=%d failed=%d output=%s",
        len(results),
        passed,
        failed,
        output_path,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

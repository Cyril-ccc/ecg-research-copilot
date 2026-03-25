"""
Validate report artifacts for eval runs.

Checks per gold_id/run_id:
- report.md exists
- required sections exist
- required limitation keywords exist (missing/confounding/time window, plus gold overrides)
- simple provenance checks:
  - report mentions at least one analysis_tables filename
  - expected plot files exist and are referenced in report
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("eval_check_report")

DEFAULT_REQUIRED_SECTIONS = [
    "Cohort definition",
    "Data & QC",
    "Results",
    "Limitations",
]

DEFAULT_LIMITATION_KEYWORDS = [
    "missing",
    "confounding",
    "time window",
]

DEFAULT_ANALYSIS_TABLE_FILENAMES = [
    "analysis_dataset.parquet",
    "feature_summary.parquet",
    "group_compare.parquet",
    "analysis_dataset_summary.json",
    "analysis_tables_summary.json",
]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def _pick_gold_ids(gold_items: list[dict[str, Any]], requested: set[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return gold_items
    return [g for g in gold_items if str(g.get("id", "")).strip() in requested]


def _required_sections(gold: dict[str, Any]) -> list[str]:
    expectations = gold.get("expectations") if isinstance(gold.get("expectations"), dict) else {}
    report_cfg = expectations.get("report") if isinstance(expectations.get("report"), dict) else {}
    sections = report_cfg.get("must_have_sections")
    if isinstance(sections, list) and sections:
        return [str(x).strip() for x in sections if str(x).strip()]
    return list(DEFAULT_REQUIRED_SECTIONS)


def _required_keywords(gold: dict[str, Any]) -> list[str]:
    expectations = gold.get("expectations") if isinstance(gold.get("expectations"), dict) else {}
    report_cfg = expectations.get("report") if isinstance(expectations.get("report"), dict) else {}
    must_mention = report_cfg.get("must_mention") if isinstance(report_cfg.get("must_mention"), list) else []
    out = [str(x).strip() for x in must_mention if str(x).strip()]

    # Force minimal limitations keywords even if gold YAML omitted them.
    for kw in DEFAULT_LIMITATION_KEYWORDS:
        if kw not in out:
            out.append(kw)

    return out


def _analysis_table_filenames(args: argparse.Namespace) -> list[str]:
    return [x.strip() for x in str(args.analysis_table_filenames).split(",") if x.strip()]


def _normalize_keyword_text(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def _keyword_present(keyword: str, text_lower: str, text_norm: str) -> bool:
    kw = str(keyword).strip().lower()
    if not kw:
        return True
    variants = {
        kw,
        kw.replace(" ", "_"),
        kw.replace(" ", "-"),
    }
    if any(v in text_lower for v in variants):
        return True
    return _normalize_keyword_text(kw) in text_norm


def _load_expected_plot_paths(run_dir: Path) -> list[str]:
    summary_path = run_dir / "plots" / "report_plots_summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            plots = payload.get("plots")
            if isinstance(plots, list):
                out = [str(p).strip() for p in plots if str(p).strip()]
                if out:
                    return out
        except Exception:
            pass

    # Fallback to files physically under plots/
    if not (run_dir / "plots").exists():
        return []

    out: list[str] = []
    for p in sorted((run_dir / "plots").glob("*")):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
            out.append(str(Path("plots") / p.name).replace("\\", "/"))
    return out


def _check_one(
    *,
    gold: dict[str, Any],
    run_id: str | None,
    artifacts_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gold_id = str(gold.get("id", "")).strip() or "UNKNOWN"
    name = str(gold.get("name", "")).strip()

    result: dict[str, Any] = {
        "gold_id": gold_id,
        "name": name,
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
        result["checks"]["report_exists"] = False
        fail("missing run_id.txt")
        return result

    run_dir = artifacts_root / run_id
    report_path = run_dir / "report.md"
    result["report_path"] = str(report_path)

    if not report_path.exists():
        result["checks"]["report_exists"] = False
        fail("report.md not found")
        return result

    result["checks"]["report_exists"] = True
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    lower_text = text.lower()
    normalized_text = _normalize_keyword_text(text)

    # Sections
    required_sections = _required_sections(gold)
    missing_sections = [s for s in required_sections if str(s).lower() not in lower_text]
    result["checks"]["missing_sections"] = missing_sections
    result["metrics"]["required_sections"] = required_sections
    if missing_sections:
        fail(f"missing required sections: {missing_sections}")

    # Limitation keywords
    required_keywords = _required_keywords(gold)
    missing_keywords = [k for k in required_keywords if not _keyword_present(str(k), lower_text, normalized_text)]
    result["checks"]["missing_keywords"] = missing_keywords
    result["metrics"]["required_keywords"] = required_keywords
    if missing_keywords:
        fail(f"missing required keywords: {missing_keywords}")

    # Simple provenance check against analysis_tables filenames
    analysis_filenames = _analysis_table_filenames(args)
    referenced_analysis = [f for f in analysis_filenames if f.lower() in lower_text]
    result["checks"]["referenced_analysis_tables"] = referenced_analysis
    if not referenced_analysis:
        fail("no analysis_tables filename referenced in report")

    # Plots existence + reference checks
    expected_plots = _load_expected_plot_paths(run_dir)
    existing_plot_paths = []
    missing_plot_files = []
    unreferenced_plots = []

    for rel in expected_plots:
        p = run_dir / rel
        if p.exists() and p.is_file():
            existing_plot_paths.append(rel)
            rel_l = rel.lower()
            base_l = Path(rel).name.lower()
            if rel_l not in lower_text and base_l not in lower_text:
                unreferenced_plots.append(rel)
        else:
            missing_plot_files.append(rel)

    result["checks"]["expected_plots"] = expected_plots
    result["checks"]["missing_plot_files"] = missing_plot_files
    result["checks"]["unreferenced_plots"] = unreferenced_plots
    result["metrics"]["existing_plot_count"] = len(existing_plot_paths)

    if not expected_plots:
        fail("no expected plots found (summary or plots dir empty)")
    if missing_plot_files:
        fail(f"plot files missing: {missing_plot_files}")
    if unreferenced_plots:
        fail(f"plot files not referenced in report: {unreferenced_plots}")

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check report artifacts for eval runs")
    parser.add_argument("--gold", default="evals/gold_questions.yaml")
    parser.add_argument("--eval-runs-root", default="eval_runs")
    parser.add_argument("--artifacts-root", default="storage/artifacts")
    parser.add_argument("--output", default="eval_report_results.json")
    parser.add_argument("--gold-ids", default="", help="comma-separated subset of gold IDs")
    parser.add_argument(
        "--analysis-table-filenames",
        default=",".join(DEFAULT_ANALYSIS_TABLE_FILENAMES),
        help="comma-separated filenames; report must reference at least one",
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

    gold_path = Path(args.gold).resolve()
    eval_runs_root = Path(args.eval_runs_root).resolve()
    artifacts_root = Path(args.artifacts_root).resolve()
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
        "check_report done selected=%d passed=%d failed=%d output=%s",
        len(results),
        passed,
        failed,
        output_path,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

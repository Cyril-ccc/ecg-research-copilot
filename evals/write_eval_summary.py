"""
Generate eval_summary.md from eval_runs/summary_smoke.json.

Summary includes:
- pass rate
- failed gold list
- failure reason statistics
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("summary json must be an object")
    return payload


def _short_text(text: str, max_len: int = 200) -> str:
    t = " ".join(str(text).split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _normalize_reason_label(reason: str) -> str:
    text = str(reason).strip().lower()
    if not text:
        return "unknown"

    # Collapse numeric details to reduce noisy cardinality.
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _reasons_from_checks(checks: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if not isinstance(checks, dict):
        return out

    for key, value in checks.items():
        if key.endswith("_violation"):
            out.append(key)
            continue

        if key in {
            "missing_required_artifacts",
            "report_missing_sections",
            "report_missing_mentions",
        }:
            if isinstance(value, list) and value:
                out.append(f"{key}: {', '.join(str(v) for v in value)}")
            continue

        if key in {"passed", "cohort_subjects", "features_mean_hr", "qc_pass_rate"}:
            continue

        if isinstance(value, dict) and value.get("status") == "failed":
            out.append(f"{key}: failed")

    return out


def _extract_failure_reasons(item: dict[str, Any]) -> list[str]:
    reasons: list[str] = []

    status = str(item.get("status", "")).upper()
    error_text = str(item.get("error", "")).strip()
    if error_text:
        reasons.append(error_text)

    listed_reasons = item.get("reasons")
    if isinstance(listed_reasons, list):
        for reason in listed_reasons:
            txt = str(reason).strip()
            if txt:
                reasons.append(txt)

    checks = item.get("checks")
    if isinstance(checks, dict):
        reasons.extend(_reasons_from_checks(checks))

    if not reasons:
        reasons.append(f"status={status or 'UNKNOWN'}")
    return reasons


def _failed_items(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in results:
        status = str(item.get("status", "")).upper()
        if status in {"FAILED", "ERROR"}:
            out.append(item)
    return out


def _build_markdown(summary: dict[str, Any], source_path: Path) -> str:
    mode = str(summary.get("mode", "unknown"))
    selected = int(summary.get("gold_selected", len(summary.get("results", [])) or 0))
    passed = int(summary.get("passed", 0))
    failed = int(summary.get("failed", 0))
    total = max(selected, passed + failed)
    pass_rate = (passed / total * 100.0) if total > 0 else 0.0

    results = summary.get("results") if isinstance(summary.get("results"), list) else []
    failed_list = _failed_items([x for x in results if isinstance(x, dict)])

    reason_counter: Counter[str] = Counter()
    for item in failed_list:
        for r in _extract_failure_reasons(item):
            reason_counter[_normalize_reason_label(r)] += 1

    lines: list[str] = []
    lines.append("# Eval Smoke Summary")
    lines.append("")
    lines.append(f"- generated_at: `{_utc_now_iso()}`")
    lines.append(f"- source: `{source_path.as_posix()}`")
    lines.append(f"- mode: `{mode}`")
    lines.append(f"- pass_rate: `{passed}/{total} ({pass_rate:.2f}%)`")
    lines.append(f"- failed: `{failed}`")
    lines.append("")

    lines.append("## Failed Golds")
    if not failed_list:
        lines.append("- None")
    else:
        for item in failed_list:
            gold_id = str(item.get("gold_id", "UNKNOWN"))
            name = str(item.get("name", ""))
            run_id = str(item.get("run_id", ""))
            reasons = _extract_failure_reasons(item)
            reason_text = "; ".join(_short_text(r, max_len=180) for r in reasons[:3])
            lines.append(f"- `{gold_id}` ({name}) run_id=`{run_id}`: {reason_text}")
    lines.append("")

    lines.append("## Failure Reason Stats")
    if not reason_counter:
        lines.append("- None")
    else:
        for reason, count in reason_counter.most_common():
            lines.append(f"- `{reason}`: {count}")

    return "\n".join(lines) + "\n"


def _emit_github_annotations(summary: dict[str, Any]) -> None:
    results = summary.get("results") if isinstance(summary.get("results"), list) else []
    for item in results:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).upper()
        if status not in {"FAILED", "ERROR"}:
            continue

        gold_id = str(item.get("gold_id", "UNKNOWN"))
        name = str(item.get("name", ""))
        reasons = _extract_failure_reasons(item)
        message = _short_text("; ".join(reasons), max_len=500)
        print(f"::error title=Eval failed {gold_id}::{name} | {message}")


def _append_step_summary(markdown: str) -> None:
    step_summary = os.getenv("GITHUB_STEP_SUMMARY", "").strip()
    if not step_summary:
        return
    path = Path(step_summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(markdown)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate eval_summary.md from summary JSON")
    parser.add_argument("--summary-json", default="eval_runs/summary_smoke.json")
    parser.add_argument("--output", default="eval_summary.md")
    parser.add_argument("--github-annotations", action="store_true")
    parser.add_argument("--github-step-summary", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit non-zero if failed > 0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_path = Path(args.summary_json).resolve()
    output_path = Path(args.output).resolve()

    if not summary_path.exists():
        text = (
            "# Eval Smoke Summary\n\n"
            f"- generated_at: `{_utc_now_iso()}`\n"
            f"- source: `{summary_path.as_posix()}`\n"
            "- error: summary json not found\n"
        )
        output_path.write_text(text, encoding="utf-8")
        if args.github_step_summary:
            _append_step_summary(text)
        if args.strict:
            return 2
        return 0

    summary = _load_summary(summary_path)
    markdown = _build_markdown(summary, source_path=summary_path)
    output_path.write_text(markdown, encoding="utf-8")

    if args.github_annotations:
        _emit_github_annotations(summary)
    if args.github_step_summary:
        _append_step_summary(markdown)

    failed = int(summary.get("failed", 0))
    if args.strict and failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



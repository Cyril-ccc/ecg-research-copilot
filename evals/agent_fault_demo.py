"""
Demonstrate intentional guardrail break switches for agent eval.

This script runs smoke eval twice with fault injection:
1) whitelist_relaxed
2) output_leak

Expected behavior:
- each run returns non-zero (failed eval)
- the target test ID is marked FAILED/ERROR
- target failure reason includes injected fault marker
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAULT_MARKERS = {
    "whitelist_relaxed": "export_patient_records",
    "output_leak": "simulated_subject_id_leak",
}


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary is not an object: {path}")
    return payload


def _run_one_mode(
    *,
    mode: str,
    target_test_id: str,
    smoke_n: int,
    smoke_max_records: int,
    output_path: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evals" / "agent_runner.py"),
        "--mode",
        "smoke",
        "--smoke-n",
        str(smoke_n),
        "--smoke-max-records",
        str(smoke_max_records),
        "--fault-mode",
        mode,
        "--fault-target-test-id",
        target_test_id,
        "--output",
        str(output_path),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"fault mode {mode} did not produce summary: {output_path}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    summary = _load_summary(output_path)
    results = summary.get("results") if isinstance(summary.get("results"), list) else []

    failed_ids = [
        str(item.get("test_id", "")).strip()
        for item in results
        if isinstance(item, dict) and str(item.get("status", "")).upper() in {"FAILED", "ERROR"}
    ]

    target_item = None
    for item in results:
        if not isinstance(item, dict):
            continue
        if str(item.get("test_id", "")).strip().upper() == target_test_id.upper():
            target_item = item
            break

    if proc.returncode == 0:
        raise RuntimeError(
            f"fault mode {mode} unexpectedly passed (returncode=0). "
            f"failed_ids={failed_ids}"
        )

    if target_item is None:
        raise RuntimeError(
            f"fault mode {mode} missing target test {target_test_id} in summary results"
        )

    target_status = str(target_item.get("status", "")).upper()
    if target_status not in {"FAILED", "ERROR"}:
        raise RuntimeError(
            f"fault mode {mode} did not fail target {target_test_id}; status={target_status}"
        )

    marker = FAULT_MARKERS.get(mode, "")
    reason_blob = " ".join(
        [str(target_item.get("error", ""))]
        + [str(x) for x in (target_item.get("reasons") or [])]
        + [json.dumps(target_item.get("checks", {}), ensure_ascii=True)]
    )
    if marker and marker not in reason_blob:
        raise RuntimeError(
            f"fault mode {mode} failed target {target_test_id} but marker '{marker}' not found"
        )

    return {
        "mode": mode,
        "returncode": proc.returncode,
        "summary": str(output_path),
        "failed_ids": failed_ids,
        "target_status": target_status,
        "target_error": str(target_item.get("error", "")),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fault-injection demo for agent eval")
    parser.add_argument(
        "--modes",
        default="whitelist_relaxed,output_leak",
        help="comma-separated fault modes",
    )
    parser.add_argument("--target-test-id", default="AT001")
    parser.add_argument("--smoke-n", type=int, default=5)
    parser.add_argument("--smoke-max-records", type=int, default=20)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "eval_runs" / "fault_demo"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    modes = [x.strip().lower() for x in str(args.modes).split(",") if x.strip()]
    if not modes:
        raise ValueError("--modes cannot be empty")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for mode in modes:
        summary_path = output_dir / f"agent_summary_fault_{mode}.json"
        report = _run_one_mode(
            mode=mode,
            target_test_id=str(args.target_test_id).strip().upper(),
            smoke_n=max(1, int(args.smoke_n)),
            smoke_max_records=max(1, int(args.smoke_max_records)),
            output_path=summary_path,
        )
        reports.append(report)

    out = {
        "target_test_id": str(args.target_test_id).strip().upper(),
        "reports": reports,
    }
    result_path = output_dir / "fault_demo_result.json"
    result_path.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")

    for report in reports:
        print(
            f"mode={report['mode']} returncode={report['returncode']} "
            f"target_status={report['target_status']} failed_ids={report['failed_ids']}"
        )
    print(f"result={result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import ARTIFACTS_DIR

MIN_REPORT_GROUP_N = 10
TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b"
)


@dataclass(frozen=True)
class NumericEvidence:
    label: str
    value: float | int
    source: str
    note: str | None = None


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _safe_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


class AnswerWriter:
    def __init__(
        self,
        *,
        artifacts_root: Path = ARTIFACTS_DIR,
        min_group_n: int = MIN_REPORT_GROUP_N,
    ) -> None:
        self.artifacts_root = Path(artifacts_root).resolve()
        self.min_group_n = max(1, int(min_group_n))

    def write_final_answer(
        self,
        *,
        run_id: str,
        question: str | None = None,
    ) -> tuple[Path, Path, list[NumericEvidence]]:
        run_dir = self.artifacts_root / str(run_id)
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"run artifacts directory not found: {run_dir}")

        evidence, privacy_meta = self._collect_evidence(run_dir=run_dir)
        if len(evidence) < 3:
            raise RuntimeError(
                "final answer requires at least 3 traceable numeric facts, "
                f"only got {len(evidence)}"
            )

        facts_path = run_dir / "final_answer_facts.json"
        facts_payload = {
            "run_id": str(run_id),
            "fact_count": len(evidence),
            "privacy": privacy_meta,
            "facts": [
                {
                    "label": fact.label,
                    "value": fact.value,
                    "source": fact.source,
                    "note": fact.note,
                }
                for fact in evidence
            ],
        }
        facts_path.write_text(
            json.dumps(facts_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        final_answer_path = run_dir / "final_answer.md"
        content = self._render_markdown(
            run_id=str(run_id),
            question=question,
            evidence=evidence,
            privacy_meta=privacy_meta,
        )
        final_answer_path.write_text(content, encoding="utf-8")
        return final_answer_path, facts_path, evidence

    def _collect_evidence(
        self,
        *,
        run_dir: Path,
    ) -> tuple[list[NumericEvidence], dict[str, int]]:
        out: list[NumericEvidence] = []
        privacy_meta = {
            "small_group_rows_hidden": 0,
            "small_compare_rows_hidden": 0,
        }

        out.extend(self._from_cohort_summary(run_dir=run_dir))
        analysis_evidence, analysis_privacy = self._from_analysis_tables(run_dir=run_dir)
        out.extend(analysis_evidence)
        privacy_meta.update(analysis_privacy)
        out.extend(self._from_plots(run_dir=run_dir))

        deduped: list[NumericEvidence] = []
        seen: set[tuple[str, str]] = set()
        for item in out:
            key = (item.label, item.source)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped, privacy_meta

    def _from_cohort_summary(self, *, run_dir: Path) -> list[NumericEvidence]:
        path = run_dir / "cohort_summary.json"
        if not path.exists():
            return []

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return []

        out: list[NumericEvidence] = []
        distinct_subjects = _safe_int(payload.get("distinct_subjects"))
        if distinct_subjects is not None:
            out.append(
                NumericEvidence(
                    label="distinct_subjects",
                    value=distinct_subjects,
                    source="cohort_summary.json",
                )
            )
        total_rows = _safe_int(payload.get("total_rows"))
        if total_rows is not None:
            out.append(
                NumericEvidence(
                    label="cohort_total_rows",
                    value=total_rows,
                    source="cohort_summary.json",
                )
            )
        return out

    def _from_analysis_tables(
        self,
        *,
        run_dir: Path,
    ) -> tuple[list[NumericEvidence], dict[str, int]]:
        out: list[NumericEvidence] = []
        privacy_meta = {
            "small_group_rows_hidden": 0,
            "small_compare_rows_hidden": 0,
        }
        analysis_dir = run_dir / "analysis_tables"
        feature_summary_path = analysis_dir / "feature_summary.parquet"
        group_compare_path = analysis_dir / "group_compare.parquet"

        if feature_summary_path.exists():
            fs_df = pd.read_parquet(feature_summary_path)
            fs_safe = fs_df.copy()
            if "group_n" in fs_df.columns:
                group_n = pd.to_numeric(fs_df["group_n"], errors="coerce")
                fs_safe = fs_df[group_n >= self.min_group_n].copy()
            elif "n" in fs_df.columns:
                n_col = pd.to_numeric(fs_df["n"], errors="coerce")
                fs_safe = fs_df[n_col >= self.min_group_n].copy()

            hidden_n = max(0, len(fs_df) - len(fs_safe))
            privacy_meta["small_group_rows_hidden"] = int(hidden_n)

            out.append(
                NumericEvidence(
                    label="feature_summary_rows_visible",
                    value=int(len(fs_safe)),
                    source="analysis_tables/feature_summary.parquet",
                )
            )
            if hidden_n > 0:
                out.append(
                    NumericEvidence(
                        label="feature_summary_rows_hidden_small_n",
                        value=int(hidden_n),
                        source="analysis_tables/feature_summary.parquet",
                    )
                )

            if not fs_safe.empty and "missing_rate" in fs_safe.columns:
                missing_rate = pd.to_numeric(fs_safe["missing_rate"], errors="coerce")
                if missing_rate.notna().any():
                    row_idx = int(missing_rate.idxmax())
                    row_no = row_idx + 1
                    out.append(
                        NumericEvidence(
                            label="max_missing_rate",
                            value=float(missing_rate.loc[row_idx]),
                            source=f"analysis_tables/feature_summary.parquet:row {row_no}",
                        )
                    )

        if group_compare_path.exists():
            gc_df = pd.read_parquet(group_compare_path)
            gc_safe = gc_df.copy()
            if {"n_a", "n_b"}.issubset(set(gc_df.columns)):
                n_a = pd.to_numeric(gc_df["n_a"], errors="coerce")
                n_b = pd.to_numeric(gc_df["n_b"], errors="coerce")
                gc_safe = gc_df[(n_a >= self.min_group_n) & (n_b >= self.min_group_n)].copy()

            hidden_compare_n = max(0, len(gc_df) - len(gc_safe))
            privacy_meta["small_compare_rows_hidden"] = int(hidden_compare_n)

            out.append(
                NumericEvidence(
                    label="group_compare_rows_visible",
                    value=int(len(gc_safe)),
                    source="analysis_tables/group_compare.parquet",
                )
            )
            if hidden_compare_n > 0:
                out.append(
                    NumericEvidence(
                        label="group_compare_rows_hidden_small_n",
                        value=int(hidden_compare_n),
                        source="analysis_tables/group_compare.parquet",
                    )
                )

            if not gc_safe.empty and "p_value" in gc_safe.columns:
                p_values = pd.to_numeric(gc_safe["p_value"], errors="coerce")
                if p_values.notna().any():
                    row_idx = int(p_values.idxmin())
                    row_no = row_idx + 1
                    feature = (
                        str(gc_safe.loc[row_idx, "feature_name"])
                        if "feature_name" in gc_safe.columns
                        else "unknown"
                    )
                    out.append(
                        NumericEvidence(
                            label="min_p_value",
                            value=float(p_values.loc[row_idx]),
                            source=f"analysis_tables/group_compare.parquet:row {row_no}",
                            note=f"feature={feature}",
                        )
                    )
                    if "diff_mean" in gc_safe.columns:
                        diff_mean = _safe_float(gc_safe.loc[row_idx, "diff_mean"])
                        if diff_mean is not None:
                            out.append(
                                NumericEvidence(
                                    label="diff_mean_at_min_p",
                                    value=diff_mean,
                                    source=f"analysis_tables/group_compare.parquet:row {row_no}",
                                    note=f"feature={feature}",
                                )
                            )
        return out, privacy_meta

    def _from_plots(self, *, run_dir: Path) -> list[NumericEvidence]:
        plots_dir = run_dir / "plots"
        if not plots_dir.exists() or not plots_dir.is_dir():
            return []
        pngs = sorted(p for p in plots_dir.glob("*.png") if p.is_file())
        return [
            NumericEvidence(
                label="plot_file_count",
                value=int(len(pngs)),
                source="plots/*.png (file existence)",
            )
        ]

    def _render_markdown(
        self,
        *,
        run_id: str,
        question: str | None,
        evidence: list[NumericEvidence],
        privacy_meta: dict[str, int],
    ) -> str:
        lines: list[str] = []
        lines.append("# Final Answer")
        lines.append("")
        lines.append(f"- run_id: `{run_id}`")
        if question:
            lines.append(f"- question: {question}")
        lines.append("")
        lines.append("## Key Numbers (with sources)")
        for idx, fact in enumerate(evidence, start=1):
            tail = f" ({fact.note})" if fact.note else ""
            lines.append(
                f"{idx}. `{fact.label}` = `{_format_number(fact.value)}` "
                f"[source: {fact.source}]{tail}"
            )

        lines.append("")
        lines.append("## Limitations")
        lines.append("- missing: 本答案仅基于已有 artifacts，缺失变量及缺失机制会影响稳定性。")
        lines.append("- confounding: 未进行完整因果校正，残余混杂因素可能影响比较结果。")
        lines.append(
            "- time window: 指标受 index-time 与 ECG 采样时间窗定义影响，"
            "请结合窗口设置解释。"
        )
        lines.append(
            "- small-sample: n<10 的分组统计已隐藏；"
            f"hidden_feature_rows={privacy_meta.get('small_group_rows_hidden', 0)}, "
            f"hidden_compare_rows={privacy_meta.get('small_compare_rows_hidden', 0)}。"
        )
        lines.append("")
        raw = "\n".join(lines)
        return self._sanitize_output(raw)

    @staticmethod
    def _sanitize_output(text: str) -> str:
        out = str(text)
        out = re.sub(r"\bsubject_id\b", "subject", out, flags=re.IGNORECASE)
        out = TIMESTAMP_PATTERN.sub("[time_hidden]", out)
        out = re.sub(r"\b\d{6,}(?:\s*,\s*\d{6,})+\b", "[id_list_hidden]", out)
        return out

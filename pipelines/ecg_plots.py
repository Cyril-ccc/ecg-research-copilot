"""
ECG QC/feature plotting helpers.

Outputs under artifacts/<run_id>/plots/:
- 3 random QC-pass examples + 1 QC-fail example (12-lead grids)
- HR distribution histogram
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "services" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.core.datasets.reader import read_ecg_record  # noqa: E402

matplotlib.use("Agg")

LOGGER = logging.getLogger("ecg_plots")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2), "utf-8")
    tmp.replace(path)


def _sample_ids(
    qc_df: pd.DataFrame,
    *,
    n_pass: int = 3,
    n_fail: int = 1,
    seed: int = 42,
) -> list[tuple[str, bool]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[str, bool]] = []

    pass_ids = qc_df[qc_df["qc_pass"] == True]["record_id"].astype(str).tolist()  # noqa: E712
    fail_ids = qc_df[qc_df["qc_pass"] == False]["record_id"].astype(str).tolist()  # noqa: E712

    if pass_ids:
        k = min(n_pass, len(pass_ids))
        chosen = rng.choice(pass_ids, size=k, replace=False).tolist()
        out.extend([(rid, True) for rid in chosen])
    if fail_ids:
        k = min(n_fail, len(fail_ids))
        chosen = rng.choice(fail_ids, size=k, replace=False).tolist()
        out.extend([(rid, False) for rid in chosen])
    return out


def _plot_12_leads(
    *,
    waveform: np.ndarray,
    fs: int,
    lead_names: list[str],
    title: str,
    out_path: Path,
) -> None:
    n_samples, n_leads = waveform.shape
    if n_samples == 0 or n_leads == 0:
        raise ValueError("Empty waveform")

    show_leads = min(12, n_leads)
    t = np.arange(n_samples, dtype=np.float64) / float(max(fs, 1))

    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    axes_1d = axes.ravel()
    for i in range(12):
        ax = axes_1d[i]
        if i < show_leads:
            x = waveform[:, i]
            y = np.nan_to_num(x, nan=0.0)
            ax.plot(t, y, linewidth=0.8, color="#2563eb")
            name = lead_names[i] if i < len(lead_names) else f"lead_{i+1}"
            ax.set_title(name, fontsize=9)
            ax.grid(alpha=0.25, linewidth=0.4)
        else:
            ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.supxlabel("Time (s)")
    fig.supylabel("Amplitude (mV)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_hr_distribution(features_df: pd.DataFrame, out_path: Path) -> bool:
    if features_df.empty or "mean_hr" not in features_df.columns:
        return False
    hr = pd.to_numeric(features_df["mean_hr"], errors="coerce").dropna().astype(float)
    if hr.empty:
        return False

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(hr.values, bins=min(20, max(6, int(np.sqrt(len(hr))))), color="#0ea5e9", alpha=0.9)
    ax.set_title("Heart Rate Distribution (QC-pass features)")
    ax.set_xlabel("HR (bpm)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def generate_qc_feature_plots(
    *,
    run_id: str,
    artifacts_root: Path,
    data_dir: Path,
    global_manifest_path: Path,
    random_seed: int = 42,
) -> dict[str, Any]:
    run_dir = artifacts_root / run_id
    qc_path = run_dir / "ecg_qc.parquet"
    features_path = run_dir / "ecg_features.parquet"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not qc_path.exists():
        raise FileNotFoundError(f"QC parquet missing: {qc_path}")

    qc_df = pd.read_parquet(qc_path)
    qc_df["record_id"] = qc_df["record_id"].astype(str)

    if not {"record_id", "qc_pass"}.issubset(qc_df.columns):
        raise RuntimeError("ecg_qc.parquet missing required columns record_id/qc_pass")

    if (run_dir / "ecg_manifest.parquet").exists():
        manifest_df = pd.read_parquet(run_dir / "ecg_manifest.parquet", columns=["record_id", "path", "source"])
    else:
        manifest_df = pd.read_parquet(global_manifest_path, columns=["record_id", "path", "source"])
    manifest_df["record_id"] = manifest_df["record_id"].astype(str)

    merged = qc_df[["record_id", "qc_pass", "qc_reasons"]].merge(
        manifest_df[["record_id", "path", "source"]],
        on="record_id",
        how="left",
    )
    merged = merged.dropna(subset=["path"]).copy()

    samples = _sample_ids(merged, seed=random_seed)
    sample_outputs: list[dict[str, Any]] = []
    for idx, (record_id, is_pass) in enumerate(samples, start=1):
        row = merged[merged["record_id"] == record_id].head(1)
        if row.empty:
            continue
        rec = row.iloc[0]
        waveform, fs, lead_names, _meta = read_ecg_record(
            base_dir=data_dir,
            record_path=str(rec["path"]),
            source=str(rec.get("source", "mimic_ecg")),
        )
        label = "pass" if is_pass else "fail"
        short_id = record_id[:48]
        out_name = f"qc_{label}_{idx}_{short_id}.png"
        out_path = plots_dir / out_name
        reasons = rec.get("qc_reasons", [])
        reasons_txt = ", ".join(reasons[:3]) if isinstance(reasons, list) else str(reasons)
        title = f"{record_id} | QC={label.upper()} | fs={fs} | {reasons_txt}"
        _plot_12_leads(
            waveform=waveform,
            fs=int(fs),
            lead_names=list(lead_names),
            title=title,
            out_path=out_path,
        )
        sample_outputs.append(
            {
                "record_id": record_id,
                "qc_pass": bool(is_pass),
                "plot_file": str(out_path.relative_to(run_dir)).replace("\\", "/"),
            }
        )

    hr_plot_file: str | None = None
    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        hr_plot_path = plots_dir / "hr_distribution.png"
        if _plot_hr_distribution(features_df, hr_plot_path):
            hr_plot_file = str(hr_plot_path.relative_to(run_dir)).replace("\\", "/")

    summary = {
        "run_id": run_id,
        "plots_dir": str(plots_dir),
        "qc_sample_plots": sample_outputs,
        "hr_distribution_plot": hr_plot_file,
        "requested_qc_samples": {"pass": 3, "fail": 1},
        "actual_qc_samples": {
            "pass": int(sum(1 for x in sample_outputs if x["qc_pass"])),
            "fail": int(sum(1 for x in sample_outputs if not x["qc_pass"])),
        },
    }
    _atomic_write_json(run_dir / "plots" / "plots_summary.json", summary)
    return summary

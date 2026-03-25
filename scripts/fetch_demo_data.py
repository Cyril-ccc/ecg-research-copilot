"""Download public PhysioNet demo datasets required by smoke evals.

The datasets are open-access demos and are intentionally fetched at runtime
instead of being committed into this repository.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DemoDataset:
    key: str
    zip_url: str
    target_dirname: str
    required_paths: tuple[str, ...]


DATASETS: dict[str, DemoDataset] = {
    "clinical": DemoDataset(
        key="clinical",
        zip_url="https://physionet.org/content/mimic-iv-demo/get-zip/2.2/",
        target_dirname="mimic-iv-clinical-database-demo-2.2",
        required_paths=("hosp", "LICENSE.txt"),
    ),
    "ecg": DemoDataset(
        key="ecg",
        zip_url="https://physionet.org/content/mimic-iv-ecg-demo/get-zip/0.1/",
        target_dirname="mimic-iv-ecg-demo-diagnostic-electrocardiogram-matched-subset-demo-0.1",
        required_paths=("files", "record_list.csv", "LICENSE.txt"),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch PhysioNet demo datasets into ./data")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--dataset",
        choices=["all", "clinical", "ecg"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the target directory already exists",
    )
    return parser.parse_args()


def _is_ready(target_dir: Path, spec: DemoDataset) -> bool:
    return target_dir.exists() and all((target_dir / rel).exists() for rel in spec.required_paths)


def _download_zip(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract_dataset(zip_path: Path, target_dir: Path) -> None:
    with tempfile.TemporaryDirectory(dir=str(target_dir.parent)) as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_dir)

        extracted_entries = [p for p in tmp_dir.iterdir()]
        if len(extracted_entries) != 1:
            raise RuntimeError(
                f"unexpected archive layout for {zip_path.name}: {len(extracted_entries)} top-level entries"
            )

        extracted_root = extracted_entries[0]
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.move(str(extracted_root), str(target_dir))


def _ensure_dataset(data_dir: Path, spec: DemoDataset, *, force: bool) -> Path:
    target_dir = data_dir / spec.target_dirname
    if _is_ready(target_dir, spec) and not force:
        print(f"[skip] {spec.key} already present: {target_dir}")
        return target_dir

    print(f"[download] {spec.key} -> {target_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{spec.target_dirname}.zip"
    _download_zip(spec.zip_url, zip_path)
    _extract_dataset(zip_path, target_dir)
    zip_path.unlink(missing_ok=True)

    if not _is_ready(target_dir, spec):
        raise RuntimeError(f"downloaded dataset is incomplete: {target_dir}")

    print(f"[ok] {spec.key} ready: {target_dir}")
    return target_dir


def main() -> int:
    args = _parse_args()
    data_dir = Path(args.data_dir).resolve()

    wanted = ["clinical", "ecg"] if args.dataset == "all" else [args.dataset]
    for key in wanted:
        _ensure_dataset(data_dir, DATASETS[key], force=bool(args.force))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

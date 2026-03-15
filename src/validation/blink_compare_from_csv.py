"""Legacy CSV-driven comparison helper for murat_2018."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation._paths import SUMMARY_METRICS_PATH
from src.validation.blink_compare import process_recording_comparison

if TYPE_CHECKING:
    from src.validation.stat import RecordingComparison


CSV_PATH = SUMMARY_METRICS_PATH
DATASET_ROOT = Path("D:/dataset/murat_2018")
TOLERANCE_SAMPLES = 20
OVERWRITE = False


def load_recording_ids(csv_path: Path = CSV_PATH, *, n_rows: int) -> list[str]:
    summary = pd.read_csv(csv_path, dtype={"recording_id": "string"})
    if "recording_id" not in summary.columns:
        raise KeyError(f"Column 'recording_id' not found in {csv_path}")

    recording_ids = (
        summary["recording_id"]
        .dropna()
        .astype("string")
        .str.strip()
        .head(n_rows)
        .tolist()
    )
    return [recording_id for recording_id in recording_ids if recording_id]


def compare_first_rows(
    n_rows: int,
    csv_path: Path = CSV_PATH,
    *,
    dataset_root: Path = DATASET_ROOT,
    tolerance_samples: int = TOLERANCE_SAMPLES,
    overwrite: bool = OVERWRITE,
) -> list[RecordingComparison]:
    comparisons: list[RecordingComparison] = []

    for recording_id in load_recording_ids(csv_path, n_rows=n_rows):
        recording_dir = dataset_root / recording_id
        comparisons.append(
            process_recording_comparison(
                recording_dir,
                recording_dir / "pyblinker_results.pkl",
                recording_dir / "blinker_results.pkl",
                recording_dir / f"{recording_id}.fif",
                recording_id,
                tolerance_samples=tolerance_samples,
                overwrite=overwrite,
            )
        )

    return comparisons


def main() -> int:
    print(compare_first_rows(9))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

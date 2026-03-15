from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import mne
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from o.blink_compare import prepare_event_tables
from o.stat import RecordingComparison, build_summary_frame
from pyblinker.blinker.pyblinker import BlinkDetector
from pyblinker.utils.evaluation import blink_comparison


DATASET_ROOT = Path(r"D:\dataset\murat_2018")
SUMMARY_CSV = PROJECT_ROOT / "blinker_pyblinker_validation" / "summary_metrics.csv"
DEFAULT_SUBJECTS = ("9636595", "12400406")
TOLERANCE_SAMPLES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fresh PyBlinker output against stored MATLAB subject results.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=len(DEFAULT_SUBJECTS),
        help="Use the first N recording IDs from summary_metrics.csv.",
    )
    return parser.parse_args()


def load_subjects(n_rows: int) -> list[str]:
    if n_rows <= 0:
        raise ValueError("Expected n_rows to be positive.")

    summary = pd.read_csv(SUMMARY_CSV, dtype={"recording_id": "string"})
    recording_ids = (
        summary["recording_id"]
        .dropna()
        .astype("string")
        .str.strip()
        .head(n_rows)
        .tolist()
    )
    return [recording_id for recording_id in recording_ids if recording_id]


def compare_subject(recording_id: str) -> RecordingComparison:
    recording_dir = DATASET_ROOT / recording_id
    edf_path = recording_dir / f"{recording_id}.edf"
    blinker_path = recording_dir / "blinker_results.pkl"

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    sampling_rate = float(raw.info["sfreq"])

    detector = BlinkDetector(
        raw.copy(),
        visualize=False,
        annot_label="eye_blink",
        filter_low=1.0,
        filter_high=20.0,
        resample_rate=int(round(sampling_rate)),
        n_jobs=1,
        use_multiprocessing=False,
    )
    _annot, channel, _n_good, blink_details, _fig_data, _selected = detector.get_blink()

    py_payload = {
        "events": blink_details,
        "metrics": {
            "channel": channel,
        },
    }
    blinker_payload = pickle.load(open(blinker_path, "rb"))
    py_events, blinker_events = prepare_event_tables(py_payload, blinker_payload)
    signal = detector.raw_data.get_data(picks=[channel])[0]

    comparison = blink_comparison.compare_detected_vs_ground_truth(
        py_events,
        blinker_events,
        sampling_rate,
        tolerance_samples=TOLERANCE_SAMPLES,
        n_preview_rows=10,
        n_diff_rows=20,
        detected_signal=signal,
    )

    return RecordingComparison(
        recording_id=recording_id,
        py_events=py_events,
        blinker_events=blinker_events,
        metrics=comparison.metrics,
    )


def main() -> int:
    args = parse_args()
    comparisons = [compare_subject(recording_id) for recording_id in load_subjects(args.n)]
    summary = build_summary_frame(comparisons)
    print(summary.to_csv(index=False))
    print(json.dumps([item.metrics for item in comparisons], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

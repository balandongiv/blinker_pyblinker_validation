"""Execute the pyblinker pipeline for every FIF file in the dataset."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import pandas as pd
from pyblinker.blinker.pyblinker import BlinkDetector


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_RAW = os.environ.get("MURAT_DATASET_ROOT")
DEFAULT_ROOT = Path(DEFAULT_ROOT_RAW) if DEFAULT_ROOT_RAW else REPO_ROOT / "data" / "murat_2018"
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PyBlinkerParams:
    filter_low: float | None = 0.5
    filter_high: float | None = 30.0
    resample_rate: float | None = 100.0
    n_jobs: int = 1
    use_multiprocessing: bool = True


def discover_fif_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.fif"))


def _serialise_events(raw_events) -> pd.DataFrame:
    if isinstance(raw_events, pd.DataFrame):
        frame = raw_events.copy()
    else:
        frame = pd.DataFrame(raw_events)
    if frame.empty:
        frame["onset_sec"] = []
        frame["duration_sec"] = []
        return frame

    columns = {col.lower(): col for col in frame.columns}
    onset_col = None
    for candidate in ("onset", "start", "blink_start", "latency", "time"):
        if candidate in columns:
            onset_col = columns[candidate]
            break
    duration_col = None
    for candidate in ("duration", "blink_duration", "len", "length"):
        if candidate in columns:
            duration_col = columns[candidate]
            break

    if onset_col is not None:
        frame["onset_sec"] = pd.to_numeric(frame[onset_col], errors="coerce")
    elif "sample" in columns:
        frame["onset_sec"] = pd.to_numeric(frame[columns["sample"]], errors="coerce")
    else:
        frame["onset_sec"] = pd.NA

    if duration_col is not None:
        frame["duration_sec"] = pd.to_numeric(frame[duration_col], errors="coerce")
    elif "duration_sec" not in frame.columns:
        frame["duration_sec"] = pd.NA

    return frame


def run_pyblinker(
    fif_path: Path,
    params: PyBlinkerParams,
    overwrite: bool = False,
) -> Path | None:
    output_path = fif_path.with_name("pyblinker_results.pkl")
    if output_path.exists() and not overwrite:
        LOGGER.info("Skipping existing results for %s", fif_path)
        return None

    LOGGER.info("Loading FIF file: %s", fif_path)
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)

    detector = BlinkDetector(
        raw,
        visualize=False,
        annot_label=None,
        filter_low=params.filter_low,
        filter_high=params.filter_high,
        resample_rate=params.resample_rate,
        n_jobs=params.n_jobs,
        use_multiprocessing=params.use_multiprocessing,
    )
    annot, channel, n_good, blink_details, _fig_data, selected_channel = detector.get_blink()

    events = _serialise_events(blink_details)
    metrics = {
        "n_good_blinks": int(n_good),
        "selected_channel": selected_channel,
        "channel": channel,
        "n_events": int(len(events)),
    }

    payload = {
        "events": events,
        "metrics": metrics,
        "params": {
            "filter_low": params.filter_low,
            "filter_high": params.filter_high,
            "resample_rate": params.resample_rate,
            "n_jobs": params.n_jobs,
            "use_multiprocessing": params.use_multiprocessing,
        },
        "annotations": annot,
    }

    payload_path = output_path
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload["events"] = payload["events"].reset_index(drop=True)
    with payload_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_path = fif_path.with_name("pyblinker_results.json")
    with metadata_path.open("w", encoding="utf8") as handle:
        json.dump({"metrics": metrics, "params": payload["params"]}, handle, indent=2)

    LOGGER.info("Saved pyblinker outputs → %s", payload_path)
    return payload_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory that contains the FIF files.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of jobs for pyblinker.")
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        help="Disable multiprocessing inside pyblinker.",
    )
    parser.add_argument("--filter-low", type=float, default=0.5)
    parser.add_argument("--filter-high", type=float, default=30.0)
    parser.add_argument("--resample", type=float, default=100.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    params = PyBlinkerParams(
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        resample_rate=args.resample,
        n_jobs=args.n_jobs,
        use_multiprocessing=not args.no_multiprocessing,
    )

    fif_files = list(discover_fif_files(args.root))
    if not fif_files:
        LOGGER.warning("No FIF files were found below %s", args.root)
        return 0

    processed = 0
    for fif_path in fif_files:
        try:
            if run_pyblinker(fif_path, params, overwrite=args.force) is not None:
                processed += 1
        except Exception as exc:  # noqa: BLE001 - continue with other files
            LOGGER.error("pyblinker failed for %s: %s", fif_path, exc)

    LOGGER.info("pyblinker completed for %s/%s file(s)", processed, len(fif_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

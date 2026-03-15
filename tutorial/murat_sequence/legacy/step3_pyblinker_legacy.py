"""Execute the pyblinker pipeline for every FIF file in the dataset."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
from pyblinker.blinker.pyblinker import BlinkDetector

from src.utils.config_utils import (
    DEFAULT_CONFIG_PATH,
    get_path_setting,
    load_config,
)
from src.utils.blink_events import serialise_events


CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_ROOT = get_path_setting(CONFIG, "raw_downsampled", env_var="MURAT_DATASET_ROOT")
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PyBlinkerParams:
    filter_low: float | None = 0.5
    filter_high: float | None = 30.0
    resample_rate: float | None = 200.0
    n_jobs: int = 1
    use_multiprocessing: bool = True


def discover_fif_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.fif"))


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

    events = serialise_events(blink_details)
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
    parser.add_argument("--resample", type=float, default=200.0)
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

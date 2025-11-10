"""Create and optionally review blink ground truth annotations for a recording.

This utility script loads the previously generated ``pyblinker_results.pkl`` and
``blinker_results.pkl`` payloads for a single recording, aligns the detected blink
intervals against the MATLAB Blinker ground truth using
``blink_comparison.compute_alignments_and_metrics``, and attaches the resulting
comparison annotations to the corresponding ``.fif`` raw file.  The annotations
can be inspected interactively in the MNE Raw browser, and any manual edits made
there will be persisted with an ``annot_inspected`` suffix so they can be reused
as refined ground truth in later steps of the workflow.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import mne
import numpy as np

from src.utils.blink_compare import load_pickle, prepare_event_tables
from src.utils.config_utils import DEFAULT_CONFIG_PATH, get_path_setting, load_config

try:  # pragma: no cover - optional dependency during tests
    from pyblinker.utils.evaluation import blink_comparison
except ModuleNotFoundError as exc:  # pragma: no cover - pyblinker not installed for tests
    raise SystemExit(
        "pyblinker.utils.evaluation.blink_comparison is required to run this script",
    ) from exc


CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_ROOT = get_path_setting(CONFIG, "raw_downsampled", env_var="MURAT_DATASET_ROOT")
DEFAULT_RECORDING_ID = "12400406"
DEFAULT_TOLERANCE_SAMPLES = 20
LOGGER = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing the recording sub-directory.",
    )
    parser.add_argument(
        "--recording-id",
        default=DEFAULT_RECORDING_ID,
        help="Recording identifier used to infer file locations (default: %(default)s).",
    )
    parser.add_argument(
        "--py-path",
        type=Path,
        help="Explicit path to pyblinker_results.pkl (overrides inferred path).",
    )
    parser.add_argument(
        "--blinker-path",
        type=Path,
        help="Explicit path to blinker_results.pkl (overrides inferred path).",
    )
    parser.add_argument(
        "--fif-path",
        type=Path,
        help="Explicit path to the raw FIF recording (overrides inferred path).",
    )
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=DEFAULT_TOLERANCE_SAMPLES,
        help="Blink onset/offset tolerance in samples for alignment metrics.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open the interactive MNE Raw browser to inspect and edit annotations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args(argv)


def infer_recording_dir(root: Path, recording_id: str) -> Path:
    candidate = root / recording_id
    if candidate.is_dir():
        return candidate
    return root


def infer_py_path(recording_dir: Path, recording_id: str) -> Path:
    path = recording_dir / "pyblinker_results.pkl"
    if path.exists():
        return path
    return recording_dir / f"{recording_id}_pyblinker_results.pkl"


def infer_blinker_path(recording_dir: Path, recording_id: str) -> Path:
    path = recording_dir / "blinker_results.pkl"
    if path.exists():
        return path
    return recording_dir / f"{recording_id}_blinker_results.pkl"


def infer_fif_path(recording_dir: Path, recording_id: str) -> Path:
    path = recording_dir / f"{recording_id}.fif"
    if path.exists():
        return path
    return recording_dir / "12400406.fif"


def ensure_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Could not locate {description}: {path}")
    return path


def annotations_equal(left: mne.Annotations | None, right: mne.Annotations | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if len(left) != len(right):
        return False
    if left.orig_time != right.orig_time:
        return False
    onset_equal = np.allclose(left.onset, right.onset)
    duration_equal = np.allclose(left.duration, right.duration)
    description_equal = tuple(left.description) == tuple(right.description)
    return onset_equal and duration_equal and description_equal


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    recording_dir = infer_recording_dir(args.root, args.recording_id)

    py_path = ensure_path(
        args.py_path if args.py_path else infer_py_path(recording_dir, args.recording_id),
        "pyblinker results",
    )
    blinker_path = ensure_path(
        args.blinker_path if args.blinker_path else infer_blinker_path(recording_dir, args.recording_id),
        "blinker results",
    )
    fif_path = ensure_path(
        args.fif_path if args.fif_path else infer_fif_path(recording_dir, args.recording_id),
        "FIF recording",
    )

    LOGGER.info("Loading PyBlinker outputs from %s", py_path)
    py_payload = load_pickle(py_path)
    LOGGER.info("Loading MATLAB Blinker outputs from %s", blinker_path)
    blinker_payload = load_pickle(blinker_path)

    py_events, blinker_events, sample_rate = prepare_event_tables(py_payload, blinker_payload)
    if py_events.empty or blinker_events.empty:
        LOGGER.error("Insufficient event data for alignment (py=%s, blinker=%s)", len(py_events), len(blinker_events))
        return 1
    if sample_rate is None:
        LOGGER.warning("Sampling rate unavailable; defaulting to 100 Hz for annotation timing")
        sample_rate = 100.0

    LOGGER.info(
        "Computing alignment metrics with tolerance=%s sample(s)",
        args.tolerance_samples,
    )
    alignments, metrics = blink_comparison.compute_alignments_and_metrics(
        detected_df=py_events,
        ground_truth_df=blinker_events,
        tolerance_samples=args.tolerance_samples,
    )

    LOGGER.info("Building comparison annotations for visualization")
    annotations = blink_comparison.build_comparison_annotations(
        ground_truth_starts=blinker_events["start_blink"].to_numpy(),
        ground_truth_ends=blinker_events["end_blink"].to_numpy(),
        detected_starts=py_events["start_blink"].to_numpy(),
        detected_ends=py_events["end_blink"].to_numpy(),
        sampling_rate_hz=sample_rate,
        tolerance_samples=args.tolerance_samples,
        alignments=alignments,
    )

    LOGGER.info("Loading raw FIF file from %s", fif_path)
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")

    if annotations is not None:
        LOGGER.info("Applying %s annotation(s) to raw", len(annotations))
        raw.set_annotations(annotations)
    else:
        LOGGER.info("No annotations generated; clearing existing raw annotations")
        raw.set_annotations(None)

    pre_plot_annotations = raw.annotations.copy() if raw.annotations is not None else None

    if args.plot and os.environ.get("PYBLINKER_SKIP_PLOT") != "1":
        matches = int(metrics.get("matches_within_tolerance", 0))
        ground_truth_only = int(metrics.get("ground_truth_only", 0))
        detected_only = int(metrics.get("detected_only", 0))
        pairs_outside = int(metrics.get("pairs_outside_tolerance", 0))
        total_differences = ground_truth_only + detected_only + pairs_outside
        plot_title = (
            "Manual vs PyBlinker Blink Comparison — "
            f"Matches: {matches}, Ground Truth Only: {ground_truth_only}, "
            f"PyBlinker Only: {detected_only}, Differences: {total_differences}"
        )
        LOGGER.info("Opening raw.plot() for manual inspection")
        raw.plot(block=True, title=plot_title)
    elif args.plot:
        LOGGER.info("Skipping raw.plot() because PYBLINKER_SKIP_PLOT=1")

    post_plot_annotations = raw.annotations.copy() if raw.annotations is not None else None

    if not annotations_equal(pre_plot_annotations, post_plot_annotations):
        output_path = recording_dir / f"{args.recording_id}_annot_inspected.csv"
        LOGGER.info("Detected manual changes to annotations; saving to %s", output_path)
        raw.annotations.save(output_path)
    else:
        LOGGER.info("No annotation changes detected; nothing to save")

    LOGGER.info("Alignment metrics: %s", metrics)
    LOGGER.info("Finished creating ground truth annotations")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

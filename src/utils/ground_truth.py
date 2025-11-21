"""Utilities for generating and reviewing blink ground-truth annotations.

This module bundles all helper logic used by the ``step5_create_ground_truth``
CLI so it can be reused by tests and other tooling.  The helpers discover
recordings within the Murat et al. (2018) dataset, load the PyBlinker and
Blinker outputs for each recording, build comparison annotations, and optionally
open the interactive MNE browser so manual edits can be captured.  Any manual
adjustments are persisted with an ``annot_inspected`` suffix to flag that the
recording has already been reviewed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from argparse import Namespace

import mne
import numpy as np

from src.utils.blink_compare import load_pickle, prepare_event_tables
from src.utils.config_utils import DEFAULT_CONFIG_PATH, get_path_setting, load_config

try:  # pragma: no cover - optional dependency during tests
    from pyblinker.utils.evaluation import blink_comparison
except ModuleNotFoundError as exc:  # pragma: no cover - pyblinker not installed for tests
    blink_comparison = None
    _PYBLINKER_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when PyBlinker is available
    _PYBLINKER_IMPORT_ERROR = None


CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_ROOT = get_path_setting(CONFIG, "raw_downsampled", env_var="MURAT_DATASET_ROOT")
DEFAULT_TOLERANCE_SAMPLES = 20
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingTask:
    """Resolved file paths needed to annotate a single recording."""

    recording_id: str
    directory: Path
    fif_path: Path
    py_path: Path
    blinker_path: Path

    @property
    def inspected_path(self) -> Path:
        """Return the CSV path used to persist manually inspected annotations."""

        return self.directory / f"{self.recording_id}_annot_inspected.csv"


def infer_recording_dir(root: Path, recording_id: str) -> Path:
    """Return the sub-directory expected to contain a recording."""

    candidate = root / recording_id
    if candidate.is_dir():
        return candidate
    return root


def infer_py_path(recording_dir: Path, recording_id: str) -> Path:
    """Guess the PyBlinker pickle path for a recording."""

    path = recording_dir / "pyblinker_results.pkl"
    if path.exists():
        return path
    return recording_dir / f"{recording_id}_pyblinker_results.pkl"


def infer_blinker_path(recording_dir: Path, recording_id: str) -> Path:
    """Guess the MATLAB Blinker pickle path for a recording."""

    path = recording_dir / "blinker_results.pkl"
    if path.exists():
        return path
    return recording_dir / f"{recording_id}_blinker_results.pkl"


def infer_fif_path(recording_dir: Path, recording_id: str) -> Path:
    """Guess the raw FIF file path for a recording."""

    path = recording_dir / f"{recording_id}.fif"
    if path.exists():
        return path
    fallback = sorted(recording_dir.glob("*.fif"))
    if fallback:
        return fallback[0]
    return recording_dir / "12400406.fif"


def ensure_path(path: Path, description: str) -> Path:
    """Validate that ``path`` exists, raising ``FileNotFoundError`` otherwise."""

    if not path.exists():
        raise FileNotFoundError(f"Could not locate {description}: {path}")
    return path


def annotations_equal(
    left: mne.Annotations | None, right: mne.Annotations | None
) -> bool:
    """Check whether two annotation collections contain the same data."""

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


def discover_recording_tasks(
    root: Path,
    recording_ids: Sequence[str] | None,
    include_inspected: bool,
) -> Iterator[RecordingTask]:
    """Yield ``RecordingTask`` entries found under ``root``."""

    if recording_ids:
        candidates = [infer_recording_dir(root, rec_id) for rec_id in recording_ids]
        candidate_ids = list(recording_ids)
    else:
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
        candidate_ids = [path.name for path in candidates]

    for candidate_dir, candidate_id in zip(candidates, candidate_ids):
        if not candidate_dir.exists():
            LOGGER.warning("[skip] %s does not exist", candidate_dir)
            continue
        if not candidate_dir.is_dir():
            LOGGER.warning("[skip] %s is not a directory", candidate_dir)
            continue

        fif_path = infer_fif_path(candidate_dir, candidate_id)
        if not fif_path.exists():
            LOGGER.warning("[skip] %s missing FIF file (expected %s)", candidate_dir, fif_path)
            continue
        recording_id = fif_path.stem

        py_path = infer_py_path(candidate_dir, recording_id)
        if not py_path.exists():
            LOGGER.warning("[skip] %s missing PyBlinker outputs (expected %s)", candidate_dir, py_path)
            continue

        blinker_path = infer_blinker_path(candidate_dir, recording_id)
        if not blinker_path.exists():
            LOGGER.warning("[skip] %s missing MATLAB Blinker outputs (expected %s)", candidate_dir, blinker_path)
            continue

        task = RecordingTask(
            recording_id=recording_id,
            directory=candidate_dir,
            fif_path=fif_path,
            py_path=py_path,
            blinker_path=blinker_path,
        )

        if not include_inspected and task.inspected_path.exists():
            LOGGER.info(
                "[skip] %s already inspected (found %s)",
                recording_id,
                task.inspected_path,
            )
            continue

        yield task


def task_from_explicit_paths(py_path: Path, blinker_path: Path, fif_path: Path) -> RecordingTask:
    """Create a ``RecordingTask`` using explicit file overrides."""

    ensure_path(py_path, "pyblinker results")
    ensure_path(blinker_path, "blinker results")
    ensure_path(fif_path, "FIF recording")
    recording_id = fif_path.stem
    return RecordingTask(
        recording_id=recording_id,
        directory=fif_path.parent,
        fif_path=fif_path,
        py_path=py_path,
        blinker_path=blinker_path,
    )


def load_and_align(task: RecordingTask, tolerance_samples: int):
    """Load pickle payloads, compute alignments, and return annotations plus metrics."""

    if blink_comparison is None:  # pragma: no cover - guard when dependency missing
        raise RuntimeError(
            "pyblinker.utils.evaluation.blink_comparison is required to run this workflow"
        ) from _PYBLINKER_IMPORT_ERROR

    LOGGER.info("Loading PyBlinker outputs from %s", task.py_path)
    py_payload = load_pickle(task.py_path)
    LOGGER.info("Loading MATLAB Blinker outputs from %s", task.blinker_path)
    blinker_payload = load_pickle(task.blinker_path)

    py_events, blinker_events, sample_rate = prepare_event_tables(py_payload, blinker_payload)
    if py_events.empty or blinker_events.empty:
        raise ValueError(
            f"Insufficient event data for alignment (py={len(py_events)}, blinker={len(blinker_events)})"
        )
    if sample_rate is None:
        LOGGER.warning("Sampling rate unavailable; defaulting to 100 Hz for annotation timing")
        sample_rate = 100.0

    LOGGER.info(
        "Computing alignment metrics for %s with tolerance=%s sample(s)",
        task.recording_id,
        tolerance_samples,
    )
    alignments, metrics = blink_comparison.compute_alignments_and_metrics(
        detected_df=py_events,
        ground_truth_df=blinker_events,
        tolerance_samples=tolerance_samples,
    )

    LOGGER.info("Building comparison annotations for visualization")
    annotations = blink_comparison.build_comparison_annotations(
        ground_truth_starts=blinker_events["start_blink"].to_numpy(),
        ground_truth_ends=blinker_events["end_blink"].to_numpy(),
        detected_starts=py_events["start_blink"].to_numpy(),
        detected_ends=py_events["end_blink"].to_numpy(),
        sampling_rate_hz=sample_rate,
        tolerance_samples=tolerance_samples,
        alignments=alignments,
    )
    return annotations, metrics


def process_recording(
    task: RecordingTask, tolerance_samples: int, plot: bool, *, overwrite_inspected: bool
) -> dict:
    """Process a single recording and return the computed metrics."""

    annotations, metrics = load_and_align(task, tolerance_samples)

    LOGGER.info("Loading raw FIF file from %s", task.fif_path)
    raw = mne.io.read_raw_fif(task.fif_path, preload=False, verbose="ERROR")

    if annotations is not None:
        LOGGER.info("Applying %s annotation(s) to raw", len(annotations))
        raw.set_annotations(annotations)
    else:
        LOGGER.info("No annotations generated; clearing existing raw annotations")
        raw.set_annotations(None)

    pre_plot_annotations = raw.annotations.copy() if raw.annotations is not None else None

    should_plot = plot and os.environ.get("PYBLINKER_SKIP_PLOT") != "1"
    if should_plot:
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
    elif plot:
        LOGGER.info("Skipping raw.plot() because PYBLINKER_SKIP_PLOT=1")

    post_plot_annotations = raw.annotations.copy() if raw.annotations is not None else None

    if not annotations_equal(pre_plot_annotations, post_plot_annotations):
        output_path = task.inspected_path
        if not overwrite_inspected:
            LOGGER.info(
                "Detected manual changes to annotations; skipping save because overwrite is disabled",
            )
        elif output_path.exists():
            LOGGER.info(
                "Detected manual changes to annotations; skipping save because %s exists",
                output_path,
            )
        else:
            LOGGER.info("Detected manual changes to annotations; saving to %s", output_path)
            raw.annotations.save(output_path)
    else:
        LOGGER.info("No annotation changes detected; nothing to save")

    LOGGER.info("Alignment metrics for %s: %s", task.recording_id, metrics)
    return metrics


def process_all(
    tasks: Sequence[RecordingTask],
    tolerance_samples: int,
    plot: bool,
    *,
    overwrite_inspected: bool,
) -> list[tuple[RecordingTask, dict]]:
    """Process each recording task, returning the metrics for every run."""

    results: list[tuple[RecordingTask, dict]] = []
    for task in tasks:
        LOGGER.info("Processing recording %s", task.recording_id)
        metrics = process_recording(
            task, tolerance_samples, plot, overwrite_inspected=overwrite_inspected
        )
        results.append((task, metrics))
    return results


def main(args: Namespace) -> int:
    """Run the ground-truth generation workflow with ``args``."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    plot = not args.no_plot
    overwrite_inspected = getattr(args, "overwrite_inspected", True)

    if args.py_path or args.blinker_path or args.fif_path:
        if not (args.py_path and args.blinker_path and args.fif_path):
            raise SystemExit(
                "--py-path, --blinker-path, and --fif-path must be provided together"
            )
        if args.recording_ids and len(args.recording_ids) != 1:
            raise SystemExit(
                "Explicit file overrides only support processing a single recording"
            )
        LOGGER.info(
            "[focus] Using explicit file overrides for %s", args.fif_path
        )
        tasks = [
            task_from_explicit_paths(args.py_path, args.blinker_path, args.fif_path)
        ]
    else:
        root = args.root
        if not root.exists():
            raise SystemExit(f"Dataset root does not exist: {root}")
        if args.recording_ids:
            LOGGER.info(
                "[focus] Limiting processing to recording IDs: %s",
                ", ".join(args.recording_ids),
            )
        elif not args.include_inspected:
            LOGGER.info(
                "[focus] Batch processing all recordings under %s (skipping inspected)",
                root,
            )
        else:
            LOGGER.info(
                "[focus] Batch processing all recordings under %s (including inspected)",
                root,
            )
        tasks = list(
            discover_recording_tasks(
                root=root,
                recording_ids=args.recording_ids,
                include_inspected=args.include_inspected,
            )
        )
        if not tasks:
            LOGGER.info("No recordings matched the provided criteria; exiting")
            return 0
        LOGGER.info("Discovered %s recording(s) to process", len(tasks))

    try:
        process_all(
            tasks,
            tolerance_samples=args.tolerance_samples,
            plot=plot,
            overwrite_inspected=overwrite_inspected,
        )
    except Exception:  # pragma: no cover - pass failure status through CLI
        LOGGER.exception("Ground-truth generation failed")
        return 1

    LOGGER.info("Finished creating ground truth annotations")
    return 0

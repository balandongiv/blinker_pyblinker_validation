"""Batch-create and review blink ground-truth annotations for Murat et al. (2018).

This tutorial step walks through comparing PyBlinker detections against the
original MATLAB Blinker outputs for each recording located under
``data/murat_2018``.  For every recording the helper utilities load the
``pyblinker_results.pkl`` and ``blinker_results.pkl`` payloads, compute
alignment metrics, attach comparison annotations to the corresponding
``.fif`` raw file, and automatically open the interactive MNE browser so the
annotations can be inspected or edited.  Recordings that already include an
``*_annot_inspected.csv`` file are skipped during batch runs unless the
``--include-inspected`` flag is provided.  You can also target a specific
recording by passing one or more ``--recording-id`` values or explicit file
paths when you want to re-run the comparison for a single subject.
"""

from __future__ import annotations

from typing import Iterable

import argparse
from pathlib import Path

from src.utils.ground_truth import (
    DEFAULT_ROOT,
    DEFAULT_TOLERANCE_SAMPLES,
    main as run_ground_truth,
)


# Murat et al. (2018) recording identifiers that are highlighted in the
# tutorial walkthrough.  The workflow still supports overriding the list
# through ``--recording-id`` or explicit file paths when required.
# DEFAULT_RECORDING_IDS = (
#     "12400385",
#     "12400349",
#     "12400376",
#     "12400397",
#     "12400391",
#     "12400370",
#     "12400346",
#     "12400373",
#     "9636511",
#     "12400394",
# )

DEFAULT_RECORDING_IDS =None
def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser used by the ground truth workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate blink ground-truth annotations by comparing PyBlinker "
            "detections against MATLAB Blinker output.  Supports batch "
            "processing of the Murat et al. (2018) dataset as well as "
            "annotating a single recording with explicit file overrides."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing Murat et al. (2018) recording folders.",
    )
    parser.add_argument(
        "--recording-id",
        dest="recording_ids",
        nargs="+",
        help="Specific recording identifier(s) to process instead of the full dataset.",
    )
    parser.add_argument(
        "--include-inspected",
        action="store_true",
        help="Reprocess recordings that already have *_annot_inspected.csv outputs.",
    )
    parser.add_argument(
        "--no-overwrite-inspected",
        dest="overwrite_inspected",
        action="store_false",
        help=(
            "Do not overwrite existing *_annot_inspected.csv files when manual "
            "changes are detected in the MNE browser."
        ),
    )
    parser.add_argument(
        "--py-path",
        type=Path,
        help="Explicit path to a pyblinker_results.pkl file (single-recording mode).",
    )
    parser.add_argument(
        "--blinker-path",
        type=Path,
        help="Explicit path to a blinker_results.pkl file (single-recording mode).",
    )
    parser.add_argument(
        "--fif-path",
        type=Path,
        help="Explicit path to a raw FIF recording (single-recording mode).",
    )
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=DEFAULT_TOLERANCE_SAMPLES,
        help="Blink onset/offset tolerance in samples for alignment metrics.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip opening the interactive MNE browser even when available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ground-truth generation script."""

    return build_argument_parser().parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """Parse ``argv`` and run the ground-truth workflow."""

    args = parse_args(argv)

    return run_ground_truth(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

"""Run PyBlinker on Raja segment FIF files and save per-segment pickle outputs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.raja_pyblinker import (  # noqa: E402
    DEFAULT_PROCESSED_ROOT,
    configure_logging,
    count_all_segments,
    discover_subject_dirs,
    iter_segments,
    run_pyblinker_on_segment,
    status_records_to_frame,
)


LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Processed Raja dataset root containing subject folders such as S1, S2, ...",
    )
    parser.add_argument(
        "--filter-subject-id",
        default=None,
        help="Optional subject filter such as S1.",
    )
    parser.add_argument(
        "--filter-filename",
        default=None,
        help="Optional segment folder filter such as S01_20170519_043933.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing canonical pyblinker_results.pkl output.",
    )
    parser.add_argument("--filter-low", type=float, default=0.5, help="PyBlinker high-pass filter.")
    parser.add_argument("--filter-high", type=float, default=30.0, help="PyBlinker low-pass filter.")
    parser.add_argument(
        "--resample-rate",
        type=float,
        default=200.0,
        help="Target sampling rate used inside PyBlinker.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Worker count passed to PyBlinker.")
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        help="Disable PyBlinker multiprocessing for easier debugging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _log_status(status) -> None:
    message = (
        f"{status.subject_id}/{status.segment_id}: {status.status}"
        f"{' - ' + status.reason if status.reason else ''}"
    )
    if status.status == "processed":
        LOGGER.info(message)
    elif status.status.startswith("skipped_"):
        LOGGER.warning(message)
    else:
        LOGGER.error(message)


def _build_run_counts(status_frame) -> dict[str, int]:
    if status_frame.empty:
        return {}
    return {
        str(status): int(count)
        for status, count in status_frame["status"].value_counts().sort_index().items()
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    processed_root = args.processed_root
    if not processed_root.exists():
        LOGGER.error("Processed dataset root does not exist: %s", processed_root)
        return 1

    discovered_subjects = discover_subject_dirs(processed_root)
    matched_segments = list(
        iter_segments(
            processed_root,
            filter_subject_id=args.filter_subject_id,
            filter_filename=args.filter_filename,
        )
    )

    if not matched_segments:
        LOGGER.warning(
            "No Raja segments matched filter_subject_id=%r filter_filename=%r under %s",
            args.filter_subject_id,
            args.filter_filename,
            processed_root,
        )
        return 0

    statuses = []
    for segment in matched_segments:
        status = run_pyblinker_on_segment(
            segment,
            overwrite=args.force,
            filter_low=args.filter_low,
            filter_high=args.filter_high,
            resample_rate=args.resample_rate,
            n_jobs=args.n_jobs,
            use_multiprocessing=not args.no_multiprocessing,
        )
        statuses.append(status)
        _log_status(status)

    status_frame = status_records_to_frame(statuses)
    counts = _build_run_counts(status_frame)

    LOGGER.info(
        "Raja segment PyBlinker run complete. discovered_subjects=%d discovered_segments=%d matched_segments=%d counts=%s",
        len(discovered_subjects),
        count_all_segments(processed_root),
        len(matched_segments),
        counts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

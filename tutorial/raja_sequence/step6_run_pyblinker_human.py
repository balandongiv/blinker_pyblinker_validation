"""Compare Raja segment-level PyBlinker outputs against human blink annotations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation._paths import REPORTS_DIR  # noqa: E402
from src.validation.raja_pyblinker import (  # noqa: E402
    DEFAULT_HUMAN_ANNOTATION_ROOT,
    DEFAULT_PROCESSED_ROOT,
    build_subject_summary_frame,
    compare_segment_with_human_annotations,
    configure_logging,
    count_all_segments,
    discover_subject_dirs,
    iter_segments,
    status_records_to_frame,
    write_json,
    write_segment_comparison_outputs,
)
from src.validation.stat import build_overall_summary, build_summary_frame  # noqa: E402


LOGGER = logging.getLogger(__name__)
DEFAULT_TOLERANCE_SAMPLES = 20
DEFAULT_REPORTS_DIR = REPORTS_DIR / "raja_pyblinker_human"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Processed Raja dataset root containing subject folders such as S1, S2, ...",
    )
    parser.add_argument(
        "--human-annotation-root",
        type=Path,
        default=DEFAULT_HUMAN_ANNOTATION_ROOT,
        help="Root folder containing <subject>/<segment>/ear_eog.csv files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory where aggregate comparison reports should be written.",
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
        "--tolerance-samples",
        type=int,
        default=DEFAULT_TOLERANCE_SAMPLES,
        help="Tolerance window in samples used by the existing comparison logic.",
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
    if status.status == "compared":
        LOGGER.info(message)
    elif status.status.startswith("skipped_"):
        LOGGER.warning(message)
    else:
        LOGGER.error(message)


def _build_run_counts(status_frame: pd.DataFrame) -> dict[str, int]:
    if status_frame.empty:
        return {}
    return {
        str(status): int(count)
        for status, count in status_frame["status"].value_counts().sort_index().items()
    }


def _enrich_segment_summary(segment_summary: pd.DataFrame, results) -> pd.DataFrame:
    if segment_summary.empty:
        return segment_summary

    info_frame = pd.DataFrame(
        [
            {
                "recording_id": result.recording_comparison.recording_id,
                "subject_id": result.segment.subject_id,
                "segment_id": result.segment.segment_id,
                "sampling_rate_hz": result.sampling_rate_hz,
                "pyblinker_output_path": str(result.py_path),
                "annotation_csv_path": str(result.annotation_path),
            }
            for result in results
        ]
    )
    enriched = info_frame.merge(segment_summary, on="recording_id", how="left")
    return enriched.sort_values(["subject_id", "segment_id"], kind="mergesort").reset_index(drop=True)


def _write_aggregate_outputs(
    reports_dir: Path,
    *,
    segment_summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
    overall_summary,
    status_frame: pd.DataFrame,
    run_summary: dict[str, object],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    segment_summary.to_csv(reports_dir / "segment_summary.csv", index=False)
    subject_summary.to_csv(reports_dir / "subject_summary.csv", index=False)
    status_frame.to_csv(reports_dir / "comparison_status.csv", index=False)
    write_json(
        reports_dir / "overall_summary.json",
        dict(overall_summary) if not overall_summary.empty else {},
    )
    write_json(reports_dir / "run_summary.json", run_summary)


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
    comparison_results = []
    for segment in matched_segments:
        status, result = compare_segment_with_human_annotations(
            segment,
            human_annotation_root=args.human_annotation_root,
            tolerance_samples=args.tolerance_samples,
        )
        statuses.append(status)
        _log_status(status)
        if result is None:
            continue
        write_segment_comparison_outputs(result)
        comparison_results.append(result)

    status_frame = status_records_to_frame(statuses)
    run_counts = _build_run_counts(status_frame)

    segment_summary = _enrich_segment_summary(
        build_summary_frame([result.recording_comparison for result in comparison_results]),
        comparison_results,
    )
    subject_summary = build_subject_summary_frame(segment_summary)
    overall_summary = build_overall_summary(segment_summary)

    run_summary = {
        "processed_root": str(processed_root),
        "human_annotation_root": str(args.human_annotation_root),
        "reports_dir": str(args.reports_dir),
        "filter_subject_id": args.filter_subject_id or "",
        "filter_filename": args.filter_filename or "",
        "tolerance_samples": int(args.tolerance_samples),
        "discovered_subject_count": len(discovered_subjects),
        "discovered_segment_count": count_all_segments(processed_root),
        "matched_segment_count": len(matched_segments),
        "compared_segment_count": int(len(comparison_results)),
        "subject_summary_count": int(len(subject_summary)),
        "status_counts": run_counts,
    }
    _write_aggregate_outputs(
        args.reports_dir,
        segment_summary=segment_summary,
        subject_summary=subject_summary,
        overall_summary=overall_summary,
        status_frame=status_frame,
        run_summary=run_summary,
    )

    LOGGER.info(
        "Raja PyBlinker vs human comparison complete. matched_segments=%d compared_segments=%d reports_dir=%s counts=%s",
        len(matched_segments),
        len(comparison_results),
        args.reports_dir,
        run_counts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

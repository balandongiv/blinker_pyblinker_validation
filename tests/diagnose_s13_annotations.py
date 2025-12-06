"""Diagnostic script to reproduce S13 annotation loading outside the UI.

The script mirrors the Raja UI data flow:
load CVAT ZIP -> build annotation frame -> convert to MNE Annotations ->
load FIF -> raw.set_annotations(...).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import mne

# Ensure the src directory is importable when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_PATH)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.ui_raja.annotation_import import (  # noqa: E402
    DEFAULT_SAMPLING_RATE,
    expected_zip_path,
    import_annotations,
)
from src.ui_murat.annotation_io import annotations_from_frame  # noqa: E402
from src.ui_raja.discovery import SessionInfo  # noqa: E402


# Toggle to enable manual plotting when running interactively.
PLOT = False


class DiagnosticError(Exception):
    """Raised when a sanity check fails."""


# Hardcoded paths for the known mock data pair with annotations.
FIF_PATH = (
    PROJECT_ROOT
    / "mock_data"
    / "dataset"
    / "drowsy_driving_raja_processed"
    / "S13"
    / "S26_20190108_035218_3"
    / "ear_eog.fif"
)
CVAT_ROOT = (
    PROJECT_ROOT
    / "mock_data"
    / "CVAT_visual_annotation"
    / "cvat_zip_final"
)
SESSION_NAME = "S26_20190108_035218_3"
SUBJECT_ID = "S13"


def _print_header(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def _preview_iterable(values: Iterable, limit: int = 5) -> list:
    preview: list = []
    for idx, value in enumerate(values):
        if idx >= limit:
            break
        preview.append(value)
    return preview


def _candidate_in_range_count(
    onsets: Iterable[float], *, start: float, duration: float
) -> int:
    end = start + duration
    return sum(start <= onset <= end for onset in onsets)


def _rebuild_annotations(
    annotations: mne.Annotations,
    *,
    onset: Iterable[float],
    orig_time,
) -> mne.Annotations:
    """Create a new annotations object with updated onsets and orig_time."""

    return mne.Annotations(
        onset=list(onset),
        duration=list(annotations.duration),
        description=list(annotations.description),
        orig_time=orig_time,
    )


def _choose_alignment(raw: mne.io.BaseRaw, annotations: mne.Annotations) -> mne.Annotations:
    """Select an annotation alignment that maximizes in-range coverage.

    This evaluates several candidate timelines so the script works for any
    segmented FIF (e.g., first, middle, or later chunks of a longer recording).
    """

    duration = float(raw.times[-1] - raw.times[0])

    candidates: list[dict] = []

    def register_candidate(name: str, base_time: float, aligned: mne.Annotations) -> None:
        in_range = _candidate_in_range_count(aligned.onset, start=base_time, duration=duration)
        candidates.append(
            {
                "name": name,
                "base_time": base_time,
                "annotations": aligned,
                "in_range": in_range,
            }
        )

    # Candidate 1: assume onsets already relative to file start (common case for segment files).
    register_candidate("relative_to_file", float(raw.times[0]), annotations.copy())

    # Candidate 2: treat onsets as absolute within the full recording and offset by first_time.
    shifted_onsets = annotations.onset + float(raw.first_time)
    shifted_orig_time = raw.info.get("meas_date", annotations.orig_time)
    shifted = _rebuild_annotations(
        annotations,
        onset=shifted_onsets,
        orig_time=shifted_orig_time,
    )
    register_candidate("offset_by_first_time", float(raw.first_time), shifted)

    # Pick the candidate with the highest number of in-range onsets, preferring the smallest
    # shift from the raw's time axis in the event of a tie.
    candidates.sort(key=lambda entry: (-entry["in_range"], abs(entry["base_time"])))
    best = candidates[0]

    _print_header("Annotation alignment candidates")
    for entry in candidates:
        print(
            f"  {entry['name']}: {entry['in_range']}/{len(annotations)} within "
            f"[{entry['base_time']}, {entry['base_time'] + duration}]",
        )
    print(f"Using alignment: {best['name']}")

    return best["annotations"].copy()


def main() -> None:
    _print_header("Initial configuration")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"FIF path: {FIF_PATH}")
    print(f"CVAT root: {CVAT_ROOT}")

    session = SessionInfo(
        subject_id=SUBJECT_ID,
        session_name=SESSION_NAME,
        fif_path=FIF_PATH,
    )

    zip_path = expected_zip_path(CVAT_ROOT, session)
    print(f"Expected annotation ZIP: {zip_path}")

    if not FIF_PATH.exists():
        raise DiagnosticError(f"FIF file missing at {FIF_PATH}")
    if not zip_path.exists():
        raise DiagnosticError(f"Annotation ZIP missing at {zip_path}")

    _print_header("Loading annotations from ZIP")
    frame, source_path = import_annotations(
        session, CVAT_ROOT, sampling_rate=DEFAULT_SAMPLING_RATE
    )
    print(f"Annotation source: {source_path}")
    print(f"Frame shape: {frame.shape}")
    print(f"Frame columns: {list(frame.columns)}")
    if not frame.empty:
        print("Sample rows:")
        print(frame.head().to_string(index=False))

    if frame.empty:
        raise DiagnosticError("Annotation frame is empty; expected blink data.")
    expected_columns = {"onset", "duration", "description"}
    if not expected_columns.issubset(frame.columns):
        raise DiagnosticError(f"Missing expected columns: {expected_columns - set(frame.columns)}")

    _print_header("Converting to MNE Annotations")
    annotations = annotations_from_frame(frame)
    print(f"Annotations type: {type(annotations)}")
    print(f"Number of annotations: {len(annotations)}")
    print("Preview (onset, duration, description):")
    preview = _preview_iterable(
        zip(annotations.onset, annotations.duration, annotations.description),
        limit=5,
    )
    for onset, duration, desc in preview:
        print(f"  onset={onset:.3f}, duration={duration:.3f}, description={desc}")

    if len(annotations) == 0:
        raise DiagnosticError("No annotations were created from the frame.")

    _print_header("Loading FIF as Raw")
    raw = mne.io.read_raw_fif(FIF_PATH, preload=True)
    duration_seconds = raw.times[-1] - raw.times[0]
    print(raw)
    print(f"Sampling frequency: {raw.info['sfreq']}")
    print(f"Duration (s): {duration_seconds:.2f}")
    print(f"First time point: {raw.first_time}")
    print(f"Last time point: {raw.times[-1]}")

    annotations = _choose_alignment(raw, annotations)

    data_start = float(raw.times[0])
    data_end = float(raw.times[-1])
    out_of_range = [
        (onset, desc)
        for onset, desc in zip(annotations.onset, annotations.description)
        if onset < data_start or onset > data_end
    ]
    print(f"Annotations outside Raw range before trimming: {len(out_of_range)}")
    if out_of_range:
        print(
            "WARNING: Trimming annotations outside Raw range; see details below.",
            file=sys.stderr,
        )
        for onset, desc in _preview_iterable(out_of_range, limit=10):
            print(f"  onset={onset}, description={desc}")
        keep_mask = [data_start <= onset <= data_end for onset in annotations.onset]
        trimmed_annotations = annotations.copy()[keep_mask]
        print(
            f"Trimming {len(annotations) - len(trimmed_annotations)} annotations; "
            f"{len(trimmed_annotations)} remain."
        )
        annotations = trimmed_annotations

    _print_header("Attaching annotations to Raw")
    raw.set_annotations(annotations)
    print(f"Raw annotations after set_annotations: {raw.annotations}")
    print(f"Annotation count on Raw: {len(raw.annotations)}")

    if len(raw.annotations) != len(annotations):
        raise DiagnosticError(
            "Mismatch between annotations before and after set_annotations: "
            f"{len(annotations)} -> {len(raw.annotations)}"
        )

    if len(raw.annotations) == 0:
        raise DiagnosticError(
            "No annotations remain after set_annotations; verify onset alignment."
        )

    min_onset = float(raw.annotations.onset.min())
    max_onset = float(raw.annotations.onset.max())
    print(f"Minimum onset: {min_onset}")
    print(f"Maximum onset: {max_onset}")

    counts = Counter(raw.annotations.description)
    print("Counts by description (top 10):")
    for label, count in counts.most_common(10):
        print(f"  {label}: {count}")

    if PLOT:
        raw.plot(title="S13 annotations diagnostic", block=True)


if __name__ == "__main__":
    main()

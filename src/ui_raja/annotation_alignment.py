"""Helpers for aligning and attaching annotations to Raw recordings."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Tuple

import mne

from src.ui_murat.annotation_io import annotations_from_frame

logger = logging.getLogger(__name__)


class AnnotationAlignmentError(RuntimeError):
    """Raised when annotations cannot be aligned to the Raw time axis."""


@dataclass(frozen=True)
class AlignmentCandidate:
    """Summary of an evaluated annotation alignment candidate."""

    name: str
    base_time: float
    in_range: int


def _candidate_in_range_count(onsets: Iterable[float], *, start: float, duration: float) -> int:
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


def align_annotations_to_raw(
    raw: mne.io.BaseRaw,
    annotations: mne.Annotations,
    *,
    return_candidates: bool = False,
) -> mne.Annotations | Tuple[mne.Annotations, list[AlignmentCandidate], str]:
    """Select an annotation alignment that maximizes in-range coverage.

    Parameters
    ----------
    raw
        Recording to align against.
    annotations
        Candidate annotations to shift/align.
    return_candidates
        When True, also return the evaluated candidates and the chosen name for
        diagnostic printing.
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

    if not candidates:
        raise AnnotationAlignmentError("No annotation alignment candidates were generated.")

    candidates.sort(key=lambda entry: (-entry["in_range"], abs(entry["base_time"])))
    best = candidates[0]

    # Rebase onto the Raw's visible timeline (typically starting at 0) so that
    # plotting windows anchored to the UI's start/end operate on the same
    # coordinate system regardless of Raw.first_time. This avoids situations
    # where aligned onsets live past the visible end of the plot.
    data_start = float(raw.times[0])
    if data_start != 0.0:
        rebased_onsets = [onset - data_start for onset in best["annotations"].onset]
        best_annotations = _rebuild_annotations(
            best["annotations"], onset=rebased_onsets, orig_time=None
        )
    else:
        best_annotations = best["annotations"].copy()

    candidate_summaries = [
        AlignmentCandidate(
            name=entry["name"],
            base_time=entry["base_time"],
            in_range=entry["in_range"],
        )
        for entry in candidates
    ]

    if return_candidates:
        return best_annotations, candidate_summaries, best["name"]

    return best_annotations


def trim_annotations_to_raw_range(raw: mne.io.BaseRaw, annotations: mne.Annotations) -> mne.Annotations:
    """Return a copy of annotations restricted to the Raw's time range."""

    # Work in the Raw's visible window (typically 0..duration) to match the
    # rebased alignment we apply before attaching annotations for plotting.
    data_start = 0.0
    data_end = float(raw.times[-1] - raw.times[0])
    keep_mask = [data_start <= onset <= data_end for onset in annotations.onset]
    trimmed = annotations.copy()[keep_mask]
    return trimmed


def attach_frame_annotations_to_raw(raw: mne.io.BaseRaw, frame) -> mne.io.BaseRaw:
    """Convert, align, trim, and attach annotations from a DataFrame to Raw."""

    annotations = annotations_from_frame(frame)
    aligned = align_annotations_to_raw(raw, annotations)
    trimmed = trim_annotations_to_raw_range(raw, aligned)

    if len(trimmed) == 0:
        logger.warning(
            "No annotations remain after alignment/trim for %s; attaching empty annotations.",
            getattr(raw, "filenames", None),
        )
    raw.set_annotations(trimmed)
    return raw

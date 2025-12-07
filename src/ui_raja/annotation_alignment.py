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


def attach_frame_annotations_to_raw(
    raw: mne.io.BaseRaw, frame
) -> tuple[mne.io.BaseRaw, mne.Annotations]:
    """Convert, align, trim, and attach annotations from a DataFrame to Raw.

    Returns the mutated ``raw`` along with the aligned/trimmed annotations that
    were attached. Callers that need a frame aligned to the visible Raw timeline
    should convert the returned annotations instead of ``raw.annotations`` to
    avoid the first-time offset that MNE applies internally when storing
    annotations.

    Historical context:
    -------------------
    We start from a ~1.5-hour continuous preprocessed EEG recording per subject
    (e.g., ``S1.fif``) and split it into 3 temporal segments, each aligned to
    the duration of its corresponding driving video session. MNE keeps an
    absolute time axis for Raw, so after cropping a long recording the
    resulting ``raw.first_time`` reflects the original offset into the full
    file (e.g., ~1799.08 s) instead of 0. Downstream UI code assumed each
    cropped segment starts at 0 s, which caused misalignment between frame-
    based annotations and the visible Raw timeline.
    """

    annotations = annotations_from_frame(frame)
    aligned = align_annotations_to_raw(raw, annotations)
    trimmed = trim_annotations_to_raw_range(raw, aligned)

    if len(trimmed) == 0:
        logger.warning(
            "No annotations remain after alignment/trim for %s; attaching empty annotations.",
            getattr(raw, "filenames", None),
        )

    # Attach to the current Raw so the annotations live alongside the data even
    # before we normalize the time axis.
    raw.set_annotations(trimmed)

    # ---------------------------------------------------------------------
    # IMPORTANT CHANGE:
    # Historically, we cropped a ~1.5-hour continuous Raw (e.g., S1.fif) into
    # 3 temporal segments, each aligned to a driving video session. MNE keeps
    # an absolute time axis, so after cropping, ``raw.first_time`` might be
    # non-zero (e.g., 1799.08 s) rather than 0.
    #
    # Our annotation/frame alignment logic and UI treat each cropped segment as
    # if it starts at t = 0 s. To enforce this contract and eliminate the
    # first-time offset, we rebuild the Raw as a RawArray using the same data
    # and info. RawArray initializes with first_samp = 0, so first_time = 0.0.
    # ---------------------------------------------------------------------
    data = raw.get_data()
    info = raw.info.copy()
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(trimmed)

    return raw, trimmed

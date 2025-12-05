from __future__ import annotations

from typing import Iterable, Set

import mne
import pandas as pd

from src.utils.annotations import annotations_to_frame

from .annotation_io import annotations_from_frame
from .annotation_merge import split_annotations_by_window


def filter_annotations_by_description(
    frame: pd.DataFrame, skip_labels: Set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split annotations into those kept for plotting and those filtered out."""

    if not skip_labels:
        return frame, frame.iloc[0:0]

    mask = frame["description"].isin(skip_labels)
    filtered = frame[mask]
    kept = frame[~mask]
    return kept, filtered


def segment_already_annotated(frame: pd.DataFrame, start: float, end: float) -> bool:
    """Return ``True`` if any annotation overlaps the requested segment."""

    durations = frame["duration"].fillna(0)
    overlaps = (frame["onset"] < end) & ((frame["onset"] + durations) > start)
    return bool(overlaps.any())


def prepare_annotations_for_window(
    frame: pd.DataFrame,
    start: float,
    end: float,
    skip_labels: Iterable[str] | None = None,
    *,
    align_to_start: bool = False,
) -> tuple[mne.Annotations, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare annotations for editing within a time window.

    Returns a tuple of ``(local_annotations, inside, outside, filtered)`` where
    ``local_annotations`` are ready to be attached to a Raw object for plotting.
    When ``align_to_start`` is ``True``, onsets are shifted so the window begins
    at 0 seconds.
    """

    skip_labels_set: Set[str] = set(skip_labels or [])
    inside, outside = split_annotations_by_window(frame, start, end)
    inside_filtered, filtered = filter_annotations_by_description(
        inside, skip_labels_set
    )

    local_frame = inside_filtered.copy()
    if align_to_start:
        local_frame["onset"] = (local_frame["onset"] - start).clip(lower=0.0)

    local_annotations = annotations_from_frame(local_frame)
    return local_annotations, inside_filtered, outside, filtered


def merge_updated_annotations(
    updated_inside: pd.DataFrame | mne.Annotations,
    outside: pd.DataFrame,
) -> pd.DataFrame:
    """Combine updated inside-window annotations with the untouched outside rows."""

    if isinstance(updated_inside, mne.Annotations):
        inside_frame = annotations_to_frame(updated_inside)
    else:
        inside_frame = updated_inside.copy()

    merged = (
        pd.concat([inside_frame, outside], ignore_index=True)
        .sort_values("onset")
        .reset_index(drop=True)
    )
    return merged

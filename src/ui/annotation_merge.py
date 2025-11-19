"""Functions for segment-aware annotation merging."""

from __future__ import annotations

import pandas as pd

from .constants import ANNOTATION_COLUMNS


def split_annotations_by_window(frame: pd.DataFrame, start: float, end: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return clipped inside-window annotations and preserved outside parts."""

    inside_rows: list[dict[str, float | str]] = []
    outside_rows: list[dict[str, float | str]] = []

    for _, row in frame.iterrows():
        onset = float(row.get("onset", 0.0))
        duration = float(row.get("duration", 0.0) or 0.0)
        description = str(row.get("description", ""))
        offset_end = onset + duration

        overlaps = (onset < end) and (offset_end > start)
        if not overlaps:
            outside_rows.append({
                "onset": onset,
                "duration": duration,
                "description": description,
            })
            continue

        if onset < start:
            outside_rows.append({
                "onset": onset,
                "duration": max(0.0, start - onset),
                "description": description,
            })

        clipped_start = max(onset, start)
        clipped_end = min(offset_end, end)
        inside_rows.append({
            "onset": clipped_start,
            "duration": max(0.0, clipped_end - clipped_start),
            "description": description,
        })

        if offset_end > end:
            outside_rows.append({
                "onset": end,
                "duration": max(0.0, offset_end - end),
                "description": description,
            })

    inside_frame = pd.DataFrame(inside_rows, columns=ANNOTATION_COLUMNS)
    outside_frame = pd.DataFrame(outside_rows, columns=ANNOTATION_COLUMNS)
    return inside_frame, outside_frame


def merge_annotations(
    global_frame: pd.DataFrame, segment_frame: pd.DataFrame, start: float, end: float
) -> tuple[pd.DataFrame, int]:
    """Merge a freshly edited segment back into the untouched annotations."""

    inside_segment, outside_segment = split_annotations_by_window(global_frame, start, end)
    merged = pd.concat([outside_segment, segment_frame], ignore_index=True)
    merged = merged.sort_values("onset").reset_index(drop=True)
    return merged, int(len(inside_segment))


def summarize_segment_changes(original_segment: pd.DataFrame, updated_segment: pd.DataFrame) -> tuple[int, int]:
    """Return counts of added and removed annotations within a segment."""

    def _to_set(df: pd.DataFrame) -> set[tuple[float, float, str]]:
        return {
            (
                round(float(row["onset"]), 6),
                round(float(row["duration"] or 0), 6),
                str(row["description"]),
            )
            for _, row in df.iterrows()
        }

    before = _to_set(original_segment)
    after = _to_set(updated_segment)
    added = len(after - before)
    removed = len(before - after)
    return added, removed

"""Functions for segment-aware annotation merging."""

from __future__ import annotations

import pandas as pd

from .constants import ANNOTATION_COLUMNS


import pandas as pd
import numpy as np

def split_annotations_by_window(df: pd.DataFrame, start: float, end: float):
    """
    Split annotation rows into two DataFrames (`inside_frame` and `outside_frame`)
    based on whether each annotation overlaps the given time window.

    Each row is assumed to represent a time segment with:
      - `onset`      : start time of the annotation
      - `duration`   : length of the annotation
      - `description`: annotation label (not required for computation)

    The time interval of an annotation is:

        [onset, onset + duration]

    A segment is labeled **inside** if it **overlaps** the window:

        [start, end]

    i.e. when:

        onset + duration >= start  AND  onset <= end

    (inclusive). Otherwise it belongs in the **outside** frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `onset` and `duration` columns.
    start : float
        Window start time in seconds.
    end : float
        Window end time in seconds.

    Returns
    -------
    inside_frame : pandas.DataFrame
        Rows whose intervals overlap the window.
    outside_frame : pandas.DataFrame
        Rows with no overlap.

    Examples
    --------
    Example 1
    ---------
    >>> data = {
    ...     "onset": [10.0, 20.0, 18.9],
    ...     "duration": [0.3, 0.3, 1.0],
    ...     "description": ["A", "B", "C"],
    ... }
    >>> df = pd.DataFrame(data)
    >>> inside, outside = split_annotations_by_window(df, start=0, end=19)
    >>> inside
       onset  duration description
    0   10.0       0.3           A
    2   18.9       1.0           C
    >>> outside
       onset  duration description
    1   20.0       0.3           B

    Example 2 (requested)
    ---------------------
    >>> data = {
    ...     "onset": [10, 11, 14.1, 20, 30],
    ...     "duration": [0.3, 0.3, 0.3, 0.3, 0.3],
    ...     "description": ["A", "B", "C", "D", "E"],
    ... }
    >>> df = pd.DataFrame(data)
    >>> inside, outside = split_annotations_by_window(df, start=14, end=15)
    >>> inside
       onset  duration description
    2   14.1       0.3           C
    >>> outside
       onset  duration description
    0   10.0       0.3           A
    1   11.0       0.3           B
    3   20.0       0.3           D
    4   30.0       0.3           E
    """
    df_out = df.copy()

    onset = df_out["onset"].to_numpy()
    duration = df_out["duration"].to_numpy()
    segment_end = onset + duration

    # Any overlap between [onset, segment_end] and [start, end]
    inside_mask = (segment_end >= start) & (onset <= end)

    inside_frame = df_out.loc[inside_mask].reset_index(drop=True)
    outside_frame = df_out.loc[~inside_mask].reset_index(drop=True)

    return inside_frame, outside_frame


def merge_annotations(global_frame: pd.DataFrame, segment_frame: pd.DataFrame, start: float, end: float) -> tuple[pd.DataFrame, int]:
    """Merge a freshly edited segment back into the untouched annotations."""

    durations = global_frame["duration"].fillna(0)
    overlaps = (global_frame["onset"] < end) & ((global_frame["onset"] + durations) > start)
    outside_segment = global_frame.loc[~overlaps]
    merged = pd.concat([outside_segment, segment_frame], ignore_index=True)
    merged = merged.sort_values("onset").reset_index(drop=True)
    return merged, int(overlaps.sum())


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

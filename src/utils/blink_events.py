"""Shared helpers for working with blink detection outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def serialise_events(raw_events) -> pd.DataFrame:
    """Return a normalised :class:`DataFrame` from assorted event payloads."""

    if isinstance(raw_events, pd.DataFrame):
        frame = raw_events.copy()
    else:
        frame = pd.DataFrame(raw_events)
    if frame.empty:
        frame["onset_sec"] = []
        frame["duration_sec"] = []
        return frame

    columns = {col.lower(): col for col in frame.columns}
    onset_col = None
    for candidate in ("onset", "start", "blink_start", "latency", "time"):
        if candidate in columns:
            onset_col = columns[candidate]
            break
    duration_col = None
    for candidate in ("duration", "blink_duration", "len", "length"):
        if candidate in columns:
            duration_col = columns[candidate]
            break

    if onset_col is not None:
        frame["onset_sec"] = pd.to_numeric(frame[onset_col], errors="coerce")
    elif "sample" in columns:
        frame["onset_sec"] = pd.to_numeric(frame[columns["sample"]], errors="coerce")
    else:
        frame["onset_sec"] = pd.NA

    if duration_col is not None:
        frame["duration_sec"] = pd.to_numeric(frame[duration_col], errors="coerce")
    elif "duration_sec" not in frame.columns:
        frame["duration_sec"] = pd.NA

    return frame


def prepare_blinker_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert MATLAB tables with ``numpy.ndarray`` cells into serialisable frames."""

    if frame.empty:
        return frame
    return frame.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


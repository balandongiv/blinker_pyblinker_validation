from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd


@dataclass
class AnnotationChangeSummary:
    """Summary statistics describing changes between two annotation sets."""

    added: int
    removed: int
    changed: int


def annotations_to_frame(annotations: mne.Annotations) -> pd.DataFrame:
    """Convert an :class:`mne.Annotations` instance to a DataFrame."""

    return pd.DataFrame(
        {
            "onset": np.asarray(annotations.onset, dtype=float),
            "duration": np.asarray(annotations.duration, dtype=float),
            "description": np.asarray(annotations.description, dtype=str),
        }
    )


def summarize_annotation_changes(
    original: pd.DataFrame,
    updated: pd.DataFrame,
    *,
    onset_decimals: int = 6,
    duration_tolerance: float = 1e-6,
) -> AnnotationChangeSummary:
    """Return counts of added, removed, and modified annotations.

    Notes
    -----
    The comparison is keyed on rounded onset times to avoid floating-point
    jitter when matching annotations. Duration differences are evaluated with
    ``duration_tolerance`` to ignore negligible floating-point noise.
    """

    original_norm = original.copy()
    updated_norm = updated.copy()
    original_norm["onset_key"] = original_norm["onset"].round(onset_decimals)
    updated_norm["onset_key"] = updated_norm["onset"].round(onset_decimals)

    original_keys = set(original_norm["onset_key"].to_list())
    updated_keys = set(updated_norm["onset_key"].to_list())

    added = len(updated_keys - original_keys)
    removed = len(original_keys - updated_keys)

    changed = 0
    for key in original_keys & updated_keys:
        orig_row = original_norm[original_norm["onset_key"] == key].iloc[0]
        updated_row = updated_norm[updated_norm["onset_key"] == key].iloc[0]
        duration_differs = not np.isclose(
            float(orig_row["duration"]), float(updated_row["duration"]), atol=duration_tolerance
        )
        description_differs = str(orig_row["description"]) != str(updated_row["description"])
        if duration_differs or description_differs:
            changed += 1

    return AnnotationChangeSummary(added=added, removed=removed, changed=changed)

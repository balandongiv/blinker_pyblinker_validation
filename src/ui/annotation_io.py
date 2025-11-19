"""Helpers for loading, converting, and saving annotation data."""

from __future__ import annotations

import shutil
from pathlib import Path

import mne
import pandas as pd

from .constants import ANNOTATION_COLUMNS


def load_annotation_frame(csv_path: Path) -> pd.DataFrame:
    """Load annotations from CSV or return an empty, normalised frame."""

    if not csv_path.exists():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    frame = pd.read_csv(csv_path)
    for column in ANNOTATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[ANNOTATION_COLUMNS].copy()


def annotations_from_frame(frame: pd.DataFrame) -> mne.Annotations:
    """Convert a dataframe into an :class:`mne.Annotations` object."""

    if frame.empty:
        return mne.Annotations(onset=[], duration=[], description=[])
    return mne.Annotations(
        onset=frame["onset"].to_numpy(float),
        duration=frame["duration"].fillna(0).to_numpy(float),
        description=frame["description"].fillna("").astype(str).tolist(),
    )


def frame_from_annotations(annotations: mne.Annotations, *, base_time: float = 0.0) -> pd.DataFrame:
    """Return a dataframe from ``annotations`` with a time offset applied."""

    return pd.DataFrame(
        {
            "onset": annotations.onset + base_time,
            "duration": annotations.duration,
            "description": annotations.description,
        }
    )


def backup_existing_csv(csv_path: Path) -> None:
    """Create a timestamped backup of an existing annotation CSV."""

    if not csv_path.exists():
        return
    backup_dir = csv_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{csv_path.stem}_{timestamp}{csv_path.suffix}"
    shutil.copy2(csv_path, backup_dir / backup_name)


def save_annotations(csv_path: Path, frame: pd.DataFrame) -> None:
    """Persist annotations with a backup of the previous CSV."""

    backup_existing_csv(csv_path)
    frame.to_csv(csv_path, index=False)
    print(f"Saved annotations to {csv_path}")

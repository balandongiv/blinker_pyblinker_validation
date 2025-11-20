from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.ui.annotation_io import load_annotation_frame


def create_mock_raw(duration: float = 300, sfreq: float = 100) -> mne.io.Raw:
    """Create a synthetic Raw object used as a deterministic test fixture.

    Parameters
    ----------
    duration : float, optional
        Length of the synthetic recording in seconds. Default is 300 s.
    sfreq : float, optional
        Sampling frequency in Hz. Default is 100 Hz.

    Returns
    -------
    raw : mne.io.Raw
        An MNE Raw object containing a single EEG channel called ``"chan1"``,
        filled with zeros, with length ``duration`` seconds and sampling
        frequency ``sfreq``.

    Notes
    -----
    The exact signal values do not matter for these tests; only the time axis
    and annotation placement are relevant. The returned object is used as the
    target on which annotations are attached and optionally plotted.
    """
    n_samples = int(duration * sfreq)
    data = np.zeros((1, n_samples))
    info = mne.create_info(["chan1"], sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


def load_raw_with_annotations(csv_path: Path, *, duration: float = 300, sfreq: float = 100) -> mne.io.Raw:
    """Load annotations from ``csv_path`` and attach them to a synthetic Raw."""
    raw = create_mock_raw(duration=duration, sfreq=sfreq)
    frame = load_annotation_frame(csv_path)
    annotations = mne.Annotations(
        onset=frame["onset"].astype(float).to_numpy(),
        duration=frame["duration"].fillna(0).astype(float).to_numpy(),
        description=frame["description"].fillna("").astype(str).tolist(),
    )
    raw.set_annotations(annotations)
    return raw


def annotations_to_frame(annotations: mne.Annotations) -> pd.DataFrame:
    """Convert an :class:`mne.Annotations` instance to a DataFrame."""
    return pd.DataFrame(
        {
            "onset": np.asarray(annotations.onset, dtype=float),
            "duration": np.asarray(annotations.duration, dtype=float),
            "description": np.asarray(annotations.description, dtype=str),
        }
    )

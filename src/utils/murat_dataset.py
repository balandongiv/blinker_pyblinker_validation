"""Utility helpers for working with the murat_2018 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mne
from mne.export import export_raw
from pyblinker.utils.evaluation import mat_data


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversionResult:
    """Summary of FIF/EDF conversion outputs for a single recording."""

    recording_id: str
    fif_path: Path
    edf_path: Path


def resolve_dataset_file(dataset_file: Path | str, *, reference_dir: Path | None = None) -> Path:
    """Return an absolute path to ``dataset_file``.

    When ``dataset_file`` is relative it is resolved against ``reference_dir``. If
    ``reference_dir`` is not supplied the current working directory is used.
    """

    candidate = Path(dataset_file)
    if candidate.is_absolute():
        return candidate
    base = reference_dir if reference_dir is not None else Path.cwd()
    return (base / candidate).resolve()


def parse_channel_argument(arg: str | Sequence[str] | None) -> Sequence[str] | None:
    """Interpret CLI-style channel selection arguments."""

    if arg is None:
        return None
    if isinstance(arg, str):
        if not arg.strip():
            return None
        if any(sep in arg for sep in {",", "-"}):
            return mat_data.parse_channel_spec(arg)
        return [part.strip() for part in arg.split() if part.strip()]
    return arg


def create_raw(
    mat_path: Path,
    sampling_rate: float | None,
    channels: Sequence[str] | None,
) -> mne.io.BaseRaw:
    """Load an MNE ``Raw`` instance for ``mat_path`` using pyblinker helpers."""

    LOGGER.info("Loading MAT file via pyblinker helpers: %s", mat_path)
    raw = mat_data.load_raw_from_mat(mat_path, sfreq=sampling_rate)
    if channels:
        LOGGER.debug("Selecting channels: %s", ", ".join(channels))
        raw = mat_data.pick_channels(raw, channels)
    return raw


def convert_recording(
    mat_path: Path,
    force: bool,
    fif_path: Path,
    edf_path: Path,
    sampling_rate: float | None,
    channels: Sequence[str] | None,
) -> ConversionResult | None:
    """Create FIF/EDF files for ``mat_path`` if required."""

    if fif_path.exists() and edf_path.exists() and not force:
        LOGGER.info("Skipping %s (FIF & EDF already exist)", mat_path.parent.name)
        return None

    raw = create_raw(mat_path, sampling_rate, channels)

    fif_path.parent.mkdir(parents=True, exist_ok=True)
    edf_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving FIF → %s", fif_path)
    raw.save(fif_path, overwrite=True)

    LOGGER.info("Exporting EDF → %s", edf_path)
    export_raw(
        fname=str(edf_path),
        raw=raw,
        fmt="edf",
        overwrite=True,
    )

    return ConversionResult(mat_path.stem, fif_path, edf_path)


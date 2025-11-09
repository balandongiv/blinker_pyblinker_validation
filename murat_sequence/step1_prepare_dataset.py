"""Convert murat_2018 ``.mat`` recordings into FIF and EDF files."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mne
import numpy as np
from mne.export import export_raw
from scipy.io import loadmat


DEFAULT_ROOT = Path(os.environ.get("MURAT_DATASET_ROOT", "D:/dataset/murat_2018"))
LOGGER = logging.getLogger(__name__)


DATA_KEYS = ("data", "signal", "eeg", "EEG")
SFREQ_KEYS = ("srate", "fs", "sampling_rate", "rate", "sampRate")
CHANNEL_KEYS = ("chanlabels", "labels", "channels", "chanlocs")


@dataclass(slots=True)
class ConversionResult:
    recording_id: str
    fif_path: Path
    edf_path: Path


def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_fieldnames"):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {}


def _try_get(container, keys: Sequence[str]):
    mapping = _as_dict(container)
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _normalise_channels(channel_data, n_channels: int) -> list[str]:
    names: list[str] = []
    if channel_data is None:
        return [f"EEG{idx:03d}" for idx in range(1, n_channels + 1)]

    if isinstance(channel_data, (list, tuple, np.ndarray)):
        names = [str(item).strip() for item in np.ravel(channel_data)]
    else:
        names = [str(channel_data).strip()]

    names = [name for name in names if name]
    if len(names) == n_channels:
        return names

    if len(names) == 1:
        return [f"{names[0]}_{idx:03d}" for idx in range(1, n_channels + 1)]

    LOGGER.warning(
        "Channel name count mismatch (%s vs %s); generating generic labels.",
        len(names),
        n_channels,
    )
    return [f"EEG{idx:03d}" for idx in range(1, n_channels + 1)]


def _ensure_2d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim > 2:
        raise ValueError(f"Expected 1D or 2D EEG data; got shape {data.shape}")
    return data


def _load_mat(mat_path: Path):
    LOGGER.info("Loading MAT file: %s", mat_path)
    return loadmat(mat_path, squeeze_me=True, struct_as_record=False)


def _extract_eeg_payload(mat_content: dict):
    payload = mat_content
    if "EEG" in mat_content:
        payload = _as_dict(mat_content["EEG"])
    return payload


def _extract_array(payload: dict):
    data = _try_get(payload, DATA_KEYS)
    if data is None:
        raise KeyError(
            "Could not locate EEG samples in MAT file. Tried keys: "
            + ", ".join(DATA_KEYS)
        )
    array = np.asarray(data, dtype=float)
    array = _ensure_2d(array)

    if array.shape[0] > array.shape[1]:
        LOGGER.info(
            "Detected samples-first layout (shape %s); transposing to channels × samples.",
            array.shape,
        )
        array = array.T
    return array


def _extract_sfreq(payload: dict) -> float:
    value = _try_get(payload, SFREQ_KEYS)
    if value is None:
        raise KeyError(
            "Could not determine sampling rate. Tried keys: " + ", ".join(SFREQ_KEYS)
        )
    sfreq = float(np.asarray(value).squeeze())
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid sampling rate extracted from MAT file: {sfreq}")
    return sfreq


def _extract_channel_names(payload: dict, n_channels: int) -> list[str]:
    channel_data = _try_get(payload, CHANNEL_KEYS)
    return _normalise_channels(channel_data, n_channels)


def _convert_single(
    mat_path: Path,
    force: bool,
    fif_path: Path,
    edf_path: Path,
) -> ConversionResult | None:
    if fif_path.exists() and edf_path.exists() and not force:
        LOGGER.info("Skipping %s (FIF & EDF already exist)", mat_path.parent.name)
        return None

    content = _load_mat(mat_path)
    payload = _extract_eeg_payload(content)
    data = _extract_array(payload)
    sfreq = _extract_sfreq(payload)
    channel_names = _extract_channel_names(payload, data.shape[0])

    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    fif_path.parent.mkdir(parents=True, exist_ok=True)
    edf_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving FIF → %s", fif_path)
    raw.save(fif_path, overwrite=True)

    LOGGER.info("Exporting EDF → %s", edf_path)
    export_raw(raw, edf_path, fmt="edf", overwrite=True)

    return ConversionResult(mat_path.stem, fif_path, edf_path)


def iter_mat_files(root: Path) -> Iterable[Path]:
    for candidate in sorted(root.glob("*/")):
        mat_candidates = list(candidate.glob("*.mat"))
        if not mat_candidates:
            continue
        yield from mat_candidates


def convert_all(root: Path, force: bool = False) -> list[ConversionResult]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    results: list[ConversionResult] = []
    for mat_path in iter_mat_files(root):
        recording_id = mat_path.stem
        fif_path = mat_path.with_name(f"{recording_id}.fif")
        edf_path = mat_path.with_name(f"{recording_id}.edf")
        try:
            result = _convert_single(mat_path, force, fif_path, edf_path)
        except Exception as exc:  # noqa: BLE001 - log and continue
            LOGGER.error("Failed to convert %s: %s", mat_path, exc)
            continue
        if result is not None:
            results.append(result)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory that contains per-recording subfolders.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate FIF/EDF files even when they already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    try:
        results = convert_all(args.root, force=args.force)
    except Exception as exc:  # noqa: BLE001 - surface the error for CLI
        LOGGER.error("Conversion failed: %s", exc)
        return 1

    if not results:
        LOGGER.warning("No MAT files were converted.")
    else:
        LOGGER.info("Converted %s recording(s)", len(results))
        for result in results:
            LOGGER.debug(
                "Recording %s → FIF: %s | EDF: %s",
                result.recording_id,
                result.fif_path,
                result.edf_path,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

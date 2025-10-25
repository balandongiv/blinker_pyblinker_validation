"""Helpers for running pyblinker blink detection on FIF data."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mne
from pyblinker.blinker.pyblinker import BlinkDetector


class BlinkDetectionError(RuntimeError):
    """Base error raised when blink detection fails."""


class NoBlinkDetectedError(BlinkDetectionError):
    """Error raised when the detector reports no blink events."""


@dataclass(slots=True)
class PyBlinkerSettings:
    """Configuration parameters used when running the detector."""

    channels: Sequence[str] | None = None
    filter_low: float | None = None
    filter_high: float | None = None
    resample_rate: float | None = None
    n_jobs: int = 1
    use_multiprocessing: bool = True
    overwrite: bool = True


def build_settings(config: dict | None) -> PyBlinkerSettings:
    """Derive :class:`PyBlinkerSettings` from the provided configuration."""

    if not config:
        return PyBlinkerSettings(
            filter_low=0.5,
            filter_high=30.0,
            resample_rate=100.0,
        )

    raw_preprocessing = config.get("raw_preprocessing", {})
    pyblinker_cfg = config.get("pyblinker", {})

    channels = pyblinker_cfg.get("channels")
    if isinstance(channels, str):
        channels = [channels]
    elif channels is not None:
        channels = list(channels)

    filter_low = pyblinker_cfg.get("filter_low_hz", raw_preprocessing.get("highpass_hz"))
    filter_high = pyblinker_cfg.get("filter_high_hz", raw_preprocessing.get("lowpass_hz"))
    resample_rate = pyblinker_cfg.get("resample_hz", raw_preprocessing.get("resample_hz"))

    filter_low = float(filter_low) if filter_low is not None else None
    filter_high = float(filter_high) if filter_high is not None else None
    resample_rate = float(resample_rate) if resample_rate is not None else None

    n_jobs = int(pyblinker_cfg.get("n_jobs", 1))
    use_multiprocessing = bool(pyblinker_cfg.get("use_multiprocessing", True))
    overwrite = bool(pyblinker_cfg.get("overwrite", True))

    return PyBlinkerSettings(
        channels=channels,
        filter_low=filter_low,
        filter_high=filter_high,
        resample_rate=resample_rate,
        n_jobs=n_jobs,
        use_multiprocessing=use_multiprocessing,
        overwrite=overwrite,
    )


def discover_fif_files(dataset_root: Path) -> list[Path]:
    """Return all FIF files under ``dataset_root`` sorted by name."""

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    return sorted(dataset_root.rglob("*.fif"))


def _filter_available_channels(raw: mne.io.BaseRaw, candidates: Sequence[str]) -> list[str]:
    """Return the subset of ``candidates`` present in the raw file."""

    available = [ch for ch in candidates if ch in raw.ch_names]
    missing = sorted(set(candidates) - set(available))
    if missing:
        logging.warning("Ignoring channels not present in FIF: %s", ", ".join(missing))
    return available


def _choose_default_channels(raw: mne.io.BaseRaw) -> Iterable[str]:
    """Choose default channels prioritising EOG, then EEG, then fall back to all."""

    eog_picks = mne.pick_types(raw.info, eog=True)
    if len(eog_picks):
        selected = [raw.ch_names[idx] for idx in eog_picks]
        logging.info("Using EOG channels for blink detection: %s", ", ".join(selected))
        return selected

    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks):
        selected = [raw.ch_names[idx] for idx in eeg_picks]
        logging.info(
            "No dedicated EOG channels found; using EEG channels (%s available).",
            len(selected),
        )
        return selected

    logging.warning("No EOG or EEG channels identified; falling back to all channels.")
    return raw.ch_names


def prepare_raw(fif_path: Path, settings: PyBlinkerSettings) -> mne.io.BaseRaw:
    """Load, optionally pick, filter, and resample the raw FIF data."""

    if not fif_path.exists():
        raise FileNotFoundError(f"FIF file not found: {fif_path}")

    logging.info("Loading FIF file: %s", fif_path)
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

    if settings.channels:
        available = _filter_available_channels(raw, settings.channels)
        if not available:
            raise BlinkDetectionError(
                "None of the configured channels are present in the FIF file."
            )
        raw.pick(available)
    else:
        default_channels = list(_choose_default_channels(raw))
        if default_channels and set(default_channels) != set(raw.ch_names):
            raw.pick(default_channels)

    logging.info("Data contains %s channel(s) at %.2f Hz", len(raw.ch_names), raw.info["sfreq"])

    if settings.filter_low is not None or settings.filter_high is not None:
        logging.info(
            "Applying band-pass filter: low=%s Hz, high=%s Hz",
            settings.filter_low,
            settings.filter_high,
        )
        raw.filter(settings.filter_low, settings.filter_high, fir_design="firwin")

    if settings.resample_rate:
        logging.info("Resampling data to %.2f Hz", settings.resample_rate)
        raw.resample(settings.resample_rate)

    logging.info(
        "Prepared raw data with %s samples per channel and sampling rate %.2f Hz",
        raw.n_times,
        raw.info["sfreq"],
    )
    return raw


def run_blink_detection(raw: mne.io.BaseRaw, settings: PyBlinkerSettings):
    """Run pyblinker blink detection and return its outputs."""

    logging.info("Running BlinkDetector with pyblinker")
    detector = BlinkDetector(
        raw,
        visualize=False,
        annot_label=None,
        filter_low=settings.filter_low,
        filter_high=settings.filter_high,
        resample_rate=settings.resample_rate,
        n_jobs=settings.n_jobs,
        use_multiprocessing=settings.use_multiprocessing,
    )
    return detector.get_blink()


def _resolve_output_dir(
    output_root: Path, fif_path: Path, dataset_root: Path | None = None
) -> Path:
    """Return the directory where results for ``fif_path`` should be stored."""

    if dataset_root:
        try:
            relative = fif_path.relative_to(dataset_root)
        except ValueError:
            relative = fif_path.name
        else:
            relative = Path(relative)
    else:
        relative = fif_path.name

    if isinstance(relative, Path):
        target_dir = output_root / relative.parent / relative.stem
    else:
        target_dir = output_root / Path(relative).stem

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _save_pickle(obj: object, path: Path, overwrite: bool) -> None:
    """Persist ``obj`` as a pickle file respecting the ``overwrite`` flag."""

    if path.exists() and not overwrite:
        raise BlinkDetectionError(
            f"Output already exists and overwriting is disabled: {path}"
        )

    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Saved %s", path)


def process_fif_file(
    fif_path: Path,
    output_root: Path,
    settings: PyBlinkerSettings,
    dataset_root: Path | None = None,
) -> tuple[Path, Path]:
    """Process a single FIF file and persist blink detection outputs."""

    raw = prepare_raw(fif_path, settings)

    (
        annotations,
        channel,
        number_good_blinks,
        blink_details,
        fig_data,
        ch_selected,
    ) = run_blink_detection(raw, settings)

    logging.info("Selected channel reported by BlinkDetector: %s", ch_selected)
    logging.info("Primary channel used for annotations: %s", channel)
    logging.info("Detected %s blink(s)", number_good_blinks)

    if not number_good_blinks:
        raise NoBlinkDetectedError(f"No blinks were detected in {fif_path}")

    target_dir = _resolve_output_dir(output_root, fif_path, dataset_root)

    selected_path = target_dir / "selected_ch.pkl"
    blink_details_path = target_dir / "blink_details.pkl"

    _save_pickle(ch_selected, selected_path, settings.overwrite)
    _save_pickle(blink_details, blink_details_path, settings.overwrite)

    logging.info("Blink details pickle stored at: %s", blink_details_path)
    logging.info("Selected channel pickle stored at: %s", selected_path)

    raw.set_annotations(annotations)

    return selected_path, blink_details_path


def _resolve_base_output(project_root: Path, config: dict | None) -> Path:
    """Determine the base directory where pyblinker outputs should live."""

    pyblinker_cfg = (config or {}).get("pyblinker", {})
    configured_root = pyblinker_cfg.get("output_root")

    if configured_root:
        root = Path(configured_root).expanduser()
        if not root.is_absolute():
            root = project_root / root
    else:
        root = project_root

    output_root = root / "pyblinker"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def run_blinker_batch(
    dataset_root: Path,
    project_root: Path,
    config: dict | None,
) -> dict:
    """Run blink detection for all FIF files found under ``dataset_root``."""

    output_root = _resolve_base_output(project_root, config)
    settings = build_settings(config)

    fif_files = discover_fif_files(dataset_root)
    if not fif_files:
        logging.warning("No FIF files discovered under %s", dataset_root)
        return {
            "processed": 0,
            "total": 0,
            "output_root": output_root,
            "errors": {},
        }

    processed = 0
    errors: dict[Path, str] = {}

    for fif_path in fif_files:
        logging.info("Processing FIF file %s", fif_path)
        try:
            process_fif_file(fif_path, output_root, settings, dataset_root)
        except NoBlinkDetectedError as exc:
            logging.error("%s", exc)
            errors[fif_path] = str(exc)
        except BlinkDetectionError as exc:
            logging.error("Blink detection failed for %s: %s", fif_path, exc)
            errors[fif_path] = str(exc)
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.error("Unexpected failure for %s: %s", fif_path, exc)
            errors[fif_path] = repr(exc)
        else:
            processed += 1

    logging.info(
        "Blink detection finished for %s/%s FIF files", processed, len(fif_files)
    )

    return {
        "processed": processed,
        "total": len(fif_files),
        "output_root": output_root,
        "errors": errors,
    }

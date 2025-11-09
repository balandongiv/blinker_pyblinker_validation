"""Download and convert murat_2018 ``.mat`` recordings into FIF/EDF files."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mne
from mne.export import export_raw

# Ensure the repository root (which contains the ``src`` package) is importable when
# this script is executed directly via ``python murat_sequence/step1_prepare_dataset``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyblinker.utils.evaluation import mat_data  # noqa: E402 - add repo path first

from src.murat.download_dataset import (  # noqa: E402 - deferred import for path setup
    DEFAULT_LIMIT,
    DEFAULT_ROOT as DOWNLOAD_DEFAULT_ROOT,
    DownloadError,
    download_dataset,
)


DEFAULT_DATASET_FILE = r"../murat_2018_dataset.txt"
DEFAULT_ROOT = Path(os.environ.get("MURAT_DATASET_ROOT", str(DOWNLOAD_DEFAULT_ROOT)))
DEFAULT_SAMPLING_RATE = 200.0
DEFAULT_CHANNELS = "CH1,CH2"
DEFAULT_CHANNEL_SEQUENCE: Sequence[str] | None = ("CH1", "CH2")
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversionResult:
    recording_id: str
    fif_path: Path
    edf_path: Path


def _resolve_dataset_file(default: str) -> Path:
    """Resolve the dataset list relative to this script."""

    script_dir = Path(__file__).resolve().parent
    return (script_dir / default).resolve()


def _parse_channel_argument(arg: str | Sequence[str] | None) -> Sequence[str] | None:
    """Interpret the channel selection argument."""

    if arg is None:
        return None
    if isinstance(arg, str):
        if not arg.strip():
            return None
        if any(sep in arg for sep in {",", "-"}):
            return mat_data.parse_channel_spec(arg)
        return [part.strip() for part in arg.split() if part.strip()]
    return arg


def _create_raw(
    mat_path: Path,
    sampling_rate: float | None,
    channels: Sequence[str] | None,
) -> mne.io.BaseRaw:
    """Load an MNE ``Raw`` instance using the helper utilities shipped with pyblinker."""

    LOGGER.info("Loading MAT file via pyblinker helpers: %s", mat_path)
    raw = mat_data.load_raw_from_mat(mat_path, sfreq=sampling_rate)
    if channels:
        LOGGER.debug("Selecting channels: %s", ", ".join(channels))
        raw = mat_data.pick_channels(raw, channels)
    return raw


def _convert_recording(
    mat_path: Path,
    force: bool,
    fif_path: Path,
    edf_path: Path,
    sampling_rate: float | None,
    channels: Sequence[str] | None,
) -> ConversionResult | None:
    if fif_path.exists() and edf_path.exists() and not force:
        LOGGER.info("Skipping %s (FIF & EDF already exist)", mat_path.parent.name)
        return None

    raw = _create_raw(mat_path, sampling_rate, channels)

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


def iter_mat_files(root: Path) -> Iterable[Path]:
    for candidate in sorted(root.glob("*/")):
        mat_candidates = list(candidate.glob("*.mat"))
        if not mat_candidates:
            continue
        yield from mat_candidates


def convert_all(
    root: Path,
    force: bool = False,
    sampling_rate: float | None = DEFAULT_SAMPLING_RATE,
    channels: Sequence[str] | None = DEFAULT_CHANNEL_SEQUENCE,
) -> list[ConversionResult]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    results: list[ConversionResult] = []
    for mat_path in iter_mat_files(root):
        recording_id = mat_path.stem
        fif_path = mat_path.with_name(f"{recording_id}.fif")
        edf_path = mat_path.with_name(f"{recording_id}.edf")
        try:
            result = _convert_recording(
                mat_path,
                force,
                fif_path,
                edf_path,
                sampling_rate,
                channels,
            )
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
        "--dataset-file",
        type=Path,
        default=None,
        help=(
            "Text file enumerating dataset URLs (default: ../murat_2018_dataset.txt "
            "relative to this script)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "Maximum number of dataset URLs to process. Use a negative value to"
            " disable the limiter."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download phase and convert existing MAT files only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate FIF/EDF files even when they already exist.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=DEFAULT_SAMPLING_RATE,
        help="Optional sampling rate to provide when the MAT file omits it.",
    )
    parser.add_argument(
        "--channel-spec",
        type=str,
        default=DEFAULT_CHANNELS,
        help=(
            "Optional channel selection. Accepts comma-separated names or ranges like"
            " '1-3' which expand to CH1-CH3."
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Explicit list of channel names to keep (overrides --channel-spec).",
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

    limit: int | None = args.limit
    if limit is not None and limit < 0:
        limit = None

    if args.channels and args.channel_spec and args.channel_spec != DEFAULT_CHANNELS:
        LOGGER.warning(
            "Both --channels and --channel-spec provided; preferring explicit channel list."
        )
    channel_arg = args.channels if args.channels else args.channel_spec
    channels = _parse_channel_argument(channel_arg)

    if not args.skip_download:
        dataset_file = args.dataset_file or _resolve_dataset_file(DEFAULT_DATASET_FILE)
        try:
            download_dataset(
                dataset_file=dataset_file,
                root=args.root,
                limit=limit,
            )
        except DownloadError as exc:
            LOGGER.error("Dataset download failed: %s", exc)
            return 1

    try:
        results = convert_all(
            args.root,
            force=args.force,
            sampling_rate=args.sampling_rate,
            channels=channels,
        )
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

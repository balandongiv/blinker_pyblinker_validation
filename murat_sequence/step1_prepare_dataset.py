"""Download and convert murat_2018 ``.mat`` recordings into FIF/EDF files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

# Ensure the repository root (which contains the ``src`` package) is importable when
# this script is executed directly via ``python murat_sequence/step1_prepare_dataset``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.murat.download_dataset import (  # noqa: E402 - deferred import for path setup
    DEFAULT_LIMIT,
    DownloadError,
    download_dataset,
)
from src.utils.config_utils import (  # noqa: E402 - deferred import for path setup
    DEFAULT_CONFIG_PATH,
    get_default_channels,
    get_default_sampling_rate,
    get_path_setting,
    load_config,
)
from src.utils.murat_dataset import (  # noqa: E402 - deferred import for path setup
    ConversionResult,
    convert_recording,
    parse_channel_argument,
    resolve_dataset_file,
)


CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_DATASET_FILE = get_path_setting(CONFIG, "dataset_file")
DEFAULT_ROOT = get_path_setting(CONFIG, "download_root", env_var="MURAT_DATASET_ROOT")
DEFAULT_SAMPLING_RATE = (
    get_default_sampling_rate(CONFIG)
    or 200.0
)
DEFAULT_CHANNELS: Sequence[str] | None = get_default_channels(CONFIG) or ("CH1", "CH2")
LOGGER = logging.getLogger(__name__)


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
    channels: Sequence[str] | None = DEFAULT_CHANNELS,
) -> list[ConversionResult]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    results: list[ConversionResult] = []
    for mat_path in iter_mat_files(root):
        recording_id = mat_path.stem
        fif_path = mat_path.with_name(f"{recording_id}.fif")
        edf_path = mat_path.with_name(f"{recording_id}.edf")
        try:
            result = convert_recording(
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

    if args.channels and args.channel_spec:
        LOGGER.warning(
            "Both --channels and --channel-spec provided; preferring explicit channel list."
        )
    channel_arg = args.channels if args.channels else args.channel_spec
    channels = parse_channel_argument(channel_arg)

    if not args.skip_download:
        dataset_file = args.dataset_file or resolve_dataset_file(
            DEFAULT_DATASET_FILE,
            reference_dir=Path(__file__).resolve().parent,
        )
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

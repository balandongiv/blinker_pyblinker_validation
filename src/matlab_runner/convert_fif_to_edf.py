"""Convert annotated FIF recordings into EDF files.

This script walks the directory tree pointed at by the ``raw_downsampled``
setting in ``config/config.yaml`` (or a custom config path supplied on the
command line) and converts every ``seg_annotated_raw.fif`` file it finds into
an EDF file with the same name in the same directory.

The implementation mirrors the manual workflow that previously lived in this
file while making it configurable, repeatable, and suitable for batch
execution as described in ``Project_Execution_Flowchart.md``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Tuple

import mne

from src.utils.config_utils import get_dataset_root, load_config


logger = logging.getLogger(__name__)


def sanitise_metadata(raw: mne.io.BaseRaw) -> None:
    """Make metadata EDF-friendly by replacing spaces in string fields."""

    for field in ("device_info", "subject_info"):
        info_value = raw.info.get(field)
        if isinstance(info_value, dict):
            for key, value in info_value.items():
                if isinstance(value, str):
                    info_value[key] = value.replace(" ", "_")


def convert_file(fif_path: Path, overwrite: bool = False) -> bool:
    """Convert a single FIF file to EDF, returning ``True`` on success."""

    edf_path = fif_path.with_suffix(".edf")

    if edf_path.exists() and not overwrite:
        logger.info("Skipping existing EDF: %s", edf_path)
        return True

    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
        raw.pick(["eeg", "eog"])   # no stim/annotation/count channels
        raw = raw.copy().set_eeg_reference("average")
        sanitise_metadata(raw)
        raw.export(edf_path, fmt="edf", overwrite=True)
    except Exception:
        logger.exception("Failed to convert FIF -> EDF: %s", fif_path)
        return False

    logger.info("Converted FIF -> EDF: %s -> %s", fif_path, edf_path)
    return True


def iter_fif_files(dataset_root: Path) -> Iterable[Path]:
    """Yield all ``seg_annotated_raw.fif`` files below ``dataset_root``."""

    pattern = "seg_annotated_raw.fif"
    for path in dataset_root.rglob(pattern):
        if path.is_file():
            yield path


def convert_all(dataset_root: Path, overwrite: bool = False) -> Tuple[int, int]:
    """Convert every matching FIF file inside ``dataset_root``.

    Parameters
    ----------
    dataset_root
        Root directory that contains subject/session folders.
    overwrite
        When ``True`` existing EDF files are regenerated.

    Returns
    -------
    Tuple[int, int]
        ``(success_count, total_files)`` summarising the conversion run.
    """

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    logger.info("Searching for FIF files under %s", dataset_root)

    fif_files = list(iter_fif_files(dataset_root))

    if not fif_files:
        logger.warning("No seg_annotated_raw.fif files found under %s", dataset_root)
        return 0, 0

    success = 0
    for fif_path in fif_files:
        if convert_file(fif_path, overwrite=overwrite):
            success += 1

    logger.info(
        "Finished conversion: %s/%s files converted successfully.",
        success,
        len(fif_files),
    )
    return success, len(fif_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset root directory. When omitted the value is"
        " read from the configuration file (paths.raw_downsampled).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate EDF files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_root = args.dataset_root
    if dataset_root is None:
        config = load_config(args.config)
        dataset_root = get_dataset_root(config)

    convert_all(dataset_root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

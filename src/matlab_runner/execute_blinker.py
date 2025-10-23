"""Execute the Blinker MATLAB pipeline for every EDF file in the dataset."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import matlab.engine
import pandas as pd
import numpy as np

from src.matlab_runner.helper import to_dataframe
from src.utils.config_utils import get_dataset_root, load_config


BLINKER_KEYS = ("blinkFits", "blinkProps", "blinkStats", "blinks", "params")
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file used to locate the dataset.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset root directory (paths.raw_downsampled).",
    )
    parser.add_argument(
        "--edf",
        type=Path,
        nargs="*",
        help="Specific EDF files to process. When omitted every"
        " seg_annotated_raw.edf file inside the dataset root is processed.",
    )
    parser.add_argument(
        "--eeglab-root",
        type=Path,
        default=os.environ.get("EEGLAB_ROOT"),
        help="Path to the EEGLAB installation (defaults to $EEGLAB_ROOT).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Location of the MATLAB runner project to add to the MATLAB path.",
    )
    parser.add_argument(
        "--blinker-plugin",
        type=str,
        default="Blinker1.2.0",
        help="Blinker plugin folder name inside <eeglab_root>/plugins.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Pickle outputs.",
    )
    return parser.parse_args()


def iter_edf_files(dataset_root: Path) -> Iterable[Path]:
    for path in dataset_root.rglob("seg_annotated_raw.edf"):
        if path.is_file():
            yield path


def start_matlab(eeglab_root: Path, project_root: Path, blinker_plugin: str):
    eng = matlab.engine.start_matlab("-nojvm -nosplash -nodesktop")

    eng.addpath(eng.genpath(str(project_root)), nargout=0)
    eng.addpath(str(eeglab_root), nargout=0)
    eng.eeglab("nogui", nargout=0)
    eng.addpath(
        eng.genpath(str(Path(eeglab_root) / "plugins" / blinker_plugin)),
        nargout=0,
    )

    return eng


def run_blinker(eng, edf_path: Path) -> Dict[str, pd.DataFrame]:
    output = eng.run_blinker_pipeline_wrap(str(edf_path), nargout=1)
    return {key: to_dataframe(output[key]) for key in BLINKER_KEYS}


def _serialise_value(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return [_serialise_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_serialise_value(v) for v in value]
    return value


def _prepare_for_pickle(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with nested arrays converted to plain Python lists."""

    if frame.empty:
        return frame

    return frame.applymap(_serialise_value)


def save_outputs(edf_path: Path, frames: Dict[str, pd.DataFrame], overwrite: bool) -> None:
    out_dir = edf_path.parent / "blinker"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, frame in frames.items():
        target = out_dir / f"{key}.pkl"
        if target.exists() and not overwrite:
            logger.info("Skipping existing Pickle: %s", target)
            continue
        cleaned = _prepare_for_pickle(frame)
        cleaned.to_pickle(target)
        logger.info("Saved %s", target)


def run_blinker_batch(
    dataset_root: Path,
    eeglab_root: Path,
    project_root: Path = DEFAULT_PROJECT_ROOT,
    blinker_plugin: str = "Blinker1.2.0",
    edf_files: Optional[Iterable[Path]] = None,
    overwrite: bool = False,
) -> int:
    """Run the MATLAB Blinker pipeline across multiple EDF files.

    Parameters
    ----------
    dataset_root
        Dataset directory that contains the EDF files.
    eeglab_root
        Location of the EEGLAB installation.
    project_root
        MATLAB runner project path to add to the MATLAB search path.
    blinker_plugin
        EEGLAB plugin folder name to add to the path.
    edf_files
        Optional iterable of EDF paths. When omitted all
        ``seg_annotated_raw.edf`` files below ``dataset_root`` are processed.
    overwrite
        When ``True`` existing Pickle outputs are overwritten.

    Returns
    -------
    int
        Number of EDF files successfully passed through Blinker.
    """

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if not eeglab_root.exists():
        raise FileNotFoundError(f"EEGLAB root does not exist: {eeglab_root}")

    if edf_files is None:
        edf_list = list(iter_edf_files(dataset_root))
    else:
        edf_list = []
        for path in edf_files:
            candidate = Path(path)
            if candidate.exists():
                edf_list.append(candidate)
            else:
                logger.warning("EDF path does not exist and will be skipped: %s", candidate)

    if not edf_list:
        logger.warning("No seg_annotated_raw.edf files found.")
        return 0

    eng = start_matlab(eeglab_root, project_root, blinker_plugin)

    processed = 0
    try:
        for edf_path in edf_list:
            logger.info("Running Blinker on %s", edf_path)
            frames = run_blinker(eng, edf_path)
            save_outputs(edf_path, frames, overwrite=overwrite)
            processed += 1
    finally:
        eng.quit()

    return processed


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.eeglab_root is None:
        raise ValueError(
            "EEGLAB root path not provided. Use --eeglab-root or set $EEGLAB_ROOT."
        )

    dataset_root = args.dataset_root
    if dataset_root is None:
        config = load_config(args.config)
        dataset_root = get_dataset_root(config)

    run_blinker_batch(
        dataset_root=dataset_root,
        eeglab_root=args.eeglab_root,
        project_root=args.project_root,
        blinker_plugin=args.blinker_plugin,
        edf_files=args.edf,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

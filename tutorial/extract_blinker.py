"""Tutorial helper that converts FIF files to EDF and runs MATLAB Blinker."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.matlab_runner import convert_fif_to_edf, execute_blinker
from src.utils.config_utils import get_dataset_root, load_config

CONFIG_PATH = Path("../config/config.yaml")
DEFAULT_EEGLAB_ROOT = Path(r"D:\code development\matlab_plugin\eeglab2025.1.0")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info("Loading configuration from %s", CONFIG_PATH)
    config = load_config(CONFIG_PATH)
    dataset_root = get_dataset_root(config)
    logging.info("Dataset root resolved to %s", dataset_root)

    logging.info("Starting FIF -> EDF conversion")
    converted, total = convert_fif_to_edf.convert_all(dataset_root)
    if total:
        logging.info("Converted %s/%s FIF files", converted, total)
    else:
        logging.warning("No FIF files discovered. Nothing to convert.")

    eeglab_root_env = os.environ.get("EEGLAB_ROOT")
    if eeglab_root_env:
        eeglab_root = Path(eeglab_root_env)
        logging.info("Using EEGLAB_ROOT from environment: %s", eeglab_root)
    else:
        eeglab_root = DEFAULT_EEGLAB_ROOT
        logging.info(
            "EEGLAB_ROOT environment variable not set. Falling back to default path: %s",
            eeglab_root,
        )

    logging.info("Running MATLAB Blinker exports")
    processed = execute_blinker.run_blinker_batch(
        dataset_root=dataset_root,
        eeglab_root=eeglab_root,
        project_root=execute_blinker.DEFAULT_PROJECT_ROOT,
        blinker_plugin="Blinker1.2.0",
        overwrite=False,
    )
    logging.info("Blinker finished for %s EDF files", processed)

if __name__ == "__main__":
    main()

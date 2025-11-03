"""Tutorial script that runs MATLAB Blinker on the bundled sample EDF file."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from src.matlab_runner.execute_blinker import (
    BLINKER_KEYS,
    DEFAULT_PROJECT_ROOT,
    run_blinker,
    start_matlab,
)

DEFAULT_EEGLAB_ROOT = Path(r"D:\\code development\\matlab_plugin\\eeglab2025.1.0")
TEST_EDF_PATH = Path("test/test_files/mne_sample_audvis_raw.edf")
BLINKER_PLUGIN = "Blinker1.2.0"


def main() -> None:
    """Launch MATLAB, run Blinker for the sample EDF, and export the tables."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    edf_path = TEST_EDF_PATH.resolve()
    if not edf_path.exists():
        raise FileNotFoundError(
            "Sample EDF file not found. Ensure the repository test data is present at "
            f"{edf_path}"
        )

    logging.info("Starting MATLAB and loading EEGLAB + Blinker plugin")
    eng = start_matlab(DEFAULT_EEGLAB_ROOT, DEFAULT_PROJECT_ROOT, BLINKER_PLUGIN)

    try:
        logging.info("Running Blinker pipeline on %s", edf_path)
        frames: Dict[str, pd.DataFrame] = run_blinker(eng, edf_path)
    finally:
        logging.info("Shutting down MATLAB engine")
        eng.quit()

    output_root = Path("tutorial") / "sample_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    for key in BLINKER_KEYS:
        frame = frames[key]
        csv_path = output_root / f"{key}.csv"
        frame.to_csv(csv_path, index=False)
        logging.info(
            "Saved %s with shape %s and columns %s",
            csv_path,
            frame.shape,
            list(frame.columns),
        )

    logging.info(
        "Blinker processing complete. CSV outputs can be used to build MNE annotations and plots."
    )


if __name__ == "__main__":
    main()

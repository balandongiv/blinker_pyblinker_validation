"""Tutorial: run MATLAB Blinker on an EDF, build MNE annotations, and plot.

This script demonstrates how to:

1. Launch MATLAB with EEGLAB and the Blinker plugin via :func:`start_matlab`.
2. Run the Blinker pipeline on a local EDF recording.
3. Convert the resulting ``blinkFits`` table into :class:`mne.Annotations`
   (using the ``leftZero`` and ``rightZero`` sample indices), attach them to
   an :class:`mne.io.Raw` object, and visualise the annotated blinks.

Place ``mne_sample_audvis_raw.edf`` next to this file before running the
tutorial (``tutorial/mne_sample_audvis_raw.edf`` relative to the repository
root).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import mne

from src.matlab_runner.execute_blinker import (
    DEFAULT_PROJECT_ROOT,
    run_blinker,
    start_matlab,
)

DEFAULT_EEGLAB_ROOT = Path(r"D:\\code development\\matlab_plugin\\eeglab2025.1.0")
TUTORIAL_ROOT = Path(__file__).resolve().parent
TEST_EDF_PATH = TUTORIAL_ROOT / "mne_sample_audvis_raw.edf"
BLINKER_PLUGIN = "Blinker1.2.0"


def build_blink_annotations(frames: Dict[str, pd.DataFrame], sfreq: float) -> mne.Annotations:
    """Create blink annotations from the ``blinkFits`` table.

    Parameters
    ----------
    frames
        Mapping of Blinker result names to data frames returned by
        :func:`run_blinker`.
    sfreq
        Sampling frequency (Hz) of the EDF recording.

    Returns
    -------
    mne.Annotations
        Annotation structure where each blink spans ``leftZero`` to
        ``rightZero`` converted to seconds.
    """

    blink_fits = frames["blinkFits"]
    required_columns = {"leftZero", "rightZero"}
    missing = required_columns.difference(blink_fits.columns)
    if missing:
        raise KeyError(
            "blinkFits table is missing required columns: " + ", ".join(sorted(missing))
        )

    left_samples = blink_fits["leftZero"].to_numpy(dtype=float)
    right_samples = blink_fits["rightZero"].to_numpy(dtype=float)
    onsets = left_samples / sfreq
    durations = (right_samples - left_samples) / sfreq
    descriptions = ["blink"] * len(blink_fits)

    return mne.Annotations(onsets, durations, descriptions)


def main() -> None:
    """Run Blinker, attach blink annotations to the EDF, and show a plot."""

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

    logging.info("Loading EDF via MNE: %s", edf_path)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    logging.info("Sampling frequency: %.2f Hz", sfreq)

    annotations = build_blink_annotations(frames, sfreq)
    annotations.orig_time = raw.annotations.orig_time
    logging.info("Created %d blink annotations", len(annotations))

    if len(raw.annotations):
        combined = raw.annotations + annotations
    else:
        combined = annotations

    raw.set_annotations(combined)

    logging.info("Launching interactive plot with blink annotations (close the window to exit)")
    raw.plot(block=True)

    logging.info("Blinker processing complete")


if __name__ == "__main__":
    main()

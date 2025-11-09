#!/usr/bin/env python
# ruff: noqa: E402  # allow sys.path adjustments before pyblinker imports

"""Compare PyBlinker detections against manual annotations from a MAT dataset.

This tutorial mirrors the workflow used in
``tutorial/blinker/1_blink_position/understand_diff_in_blink_position.py`` but
operates on the CLA subject MAT recording that ships with this repository.

The heavy lifting now lives in :mod:`pyblinker.utils.evaluation`, leaving this
file as a beginner-friendly walkthrough that wires together the individual
steps:

1. Download the MAT EEG recording (if necessary) and load it with MNE.
2. Run :class:`pyblinker.blinker.pyblinker.BlinkDetector` on the channels of
   interest.
3. Load the manual annotations stored next to the MAT file.
4. Compare the detected vs. ground-truth blink intervals and visualise the
   differences.

Adjust :data:`TOLERANCE_SAMPLES` to change how strict the comparison is.

"""


# 1) Imports (minimal)
import os
import sys
from pathlib import Path


# 2) Basic paths and filenames (adjust these if needed)
#    - SCRIPT_DIR: folder where this script lives
#    - DATA_URL: where to download the MAT file if missing
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_URL = "https://figshare.com/ndownloader/files/12400409"

# 3) Input filenames (provided by the dataset)
MAT_FILENAME = "CLA-SubjectJ-170510-3St-LRHand-Inter.mat"
CSV_FILENAME = "CLA-SubjectJ-170510-3St-LRHand-Inter_annotations.csv"

# 4) Full input paths (MAT EEG and its manual-annotation CSV live next to this script)
MAT_PATH = SCRIPT_DIR / MAT_FILENAME
CSV_PATH = SCRIPT_DIR / CSV_FILENAME

# 5) Configuration parameters (tweak as needed)
SAMPLING_RATE_HZ = 200.0                 # sampling rate for the loaded MAT data
CHANNELS_TO_KEEP = ("CH1", "CH2", "CH3") # subset of channels for detection
TOLERANCE_SAMPLES = 20                   # blink start/end alignment tolerance
N_PREVIEW_ROWS = 10                      # how many preview rows to print in diff table
N_DIFF_ROWS = 30                         # how many differing rows to print in diff table
# RAW_PLOT_SCALINGS = {"eeg": 0.5}       # optional MNE scaling (example)

# 6) Ensure the repository root is on sys.path (so pyblinker imports work)
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

# 7) Import helper utilities from this repository (kept top-level for clarity)
from pyblinker.utils.evaluation import (
	blink_comparison,
	blink_detection,
	mat_data,
	)

# 8) Make sure any additional repo-specific paths are configured
# (no additional configuration required once repo root is on sys.path)

# 9) Ensure the MAT file exists (download if missing), then load it with MNE
mat_path = mat_data.ensure_mat_file(MAT_PATH, DATA_URL)
raw_full = mat_data.load_raw_from_mat(mat_path, SAMPLING_RATE_HZ)

# 10) Keep only the channels of interest
raw = mat_data.pick_channels(raw_full, CHANNELS_TO_KEEP)
print(f"[mne] Loaded MAT file with channels: {raw.ch_names}")

# 11) Run PyBlinker detection on the selected channels
detection = blink_detection.run_pyblinker_detection(raw, sampling_rate_hz=SAMPLING_RATE_HZ)
print(f"[detector] Event table rows: {len(detection.events)}")
hh=1
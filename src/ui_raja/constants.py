"""Raja UI constants and defaults."""

from __future__ import annotations

from pathlib import Path

# Default roots point to the bundled mock data so the UI can run without
# external datasets. These can be overridden via CLI flags or environment
# variables when running ``call_app_raja_ui.py``.
DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "mock_data"
    / "dataset"
    / "drowsy_driving_raja_processed"
)
DEFAULT_CVAT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "mock_data"
    / "CVAT_visual_annotation"
    / "cvat_zip_final"
)

ANNOTATION_COLUMNS = ["onset", "duration", "description"]

# Default video sampling rate (frames per second) for converting CVAT frame
# indices to seconds.
DEFAULT_SAMPLING_RATE = 30.0

# Preferred FIF filename within each session directory.
PRIMARY_FIF_CANDIDATES = ["ear_eog.fif", "ear_eog.fif.gz"]

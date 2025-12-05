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

# Path pair configuration used to present selectable data/CVAT roots in the UI.
DEFAULT_PATH_PAIR_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "raja_path_pairs.yaml"
)

DEFAULT_STATUS_STORE = Path(__file__).resolve().parents[2] / "config" / "raja_status.yaml"

ANNOTATION_COLUMNS = ["onset", "duration", "description"]

# Default video sampling rate (frames per second) for converting CVAT frame
# indices to seconds.
DEFAULT_SAMPLING_RATE = 30.0

# Preferred FIF filename within each session directory.
PRIMARY_FIF_CANDIDATES = ["ear_eog.fif", "ear_eog.fif.gz"]

# Default channel picks used when launching the Raja browser.
DEFAULT_CHANNEL_PICKS = [
    "EAR-avg_ear",
    "EOG-EEG-eog_vert_right",
    "EOG-EEG-eog_vert_left",
    "EEG-E8",
]

# Allow common shorthand channel names to map onto the canonical FIF labels so
# users can type friendlier names without hitting validation warnings.
CHANNEL_ALIASES: dict[str, str] = {
    "eog_vert_right": "EOG-EEG-eog_vert_right",
    "eog_vert_left": "EOG-EEG-eog_vert_left",
    "eog_hor_right": "EOG-EEG-eog_hor_right",
    "eog_hor_left": "EOG-EEG-eog_hor_left",
    "eeg--e8": "EEG-E8",
    "eeg-e8": "EEG-E8",
}

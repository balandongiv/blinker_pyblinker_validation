from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from ui.annotation_io import annotations_from_frame, frame_from_annotations, load_annotation_frame
from ui.annotation_merge import merge_annotations, split_annotations_by_window


def _create_mock_csv(csv_path):
    onsets = [2.5533887323943656] + np.linspace(10, 290, 29).tolist()
    durations = [0.23673802816901457] + [0.3 for _ in range(29)]
    descriptions = ["B"] + [chr(65 + (idx % 26)) for idx in range(29)]
    frame = pd.DataFrame(
        {
            "onset": onsets,
            "duration": durations,
            "description": descriptions,
        }
    )
    frame.to_csv(csv_path, index=False)


def _create_mock_raw(duration=300, sfreq=100):
    n_samples = int(duration * sfreq)
    data = np.zeros((1, n_samples))
    info = mne.create_info(["chan1"], sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


def test_full_duration_manual_add_merge(tmp_path):
    csv_path = tmp_path / "mock.csv"
    _create_mock_csv(csv_path)
    frame = load_annotation_frame(csv_path)

    raw = _create_mock_raw()

    start, end = 0.0, raw.times[-1]
    assert end >= 299.99
    inside, _ = split_annotations_by_window(frame, start, end)
    local_segment = inside.copy()
    local_segment["onset"] = (local_segment["onset"] - start).clip(lower=0.0)
    local_annotations = annotations_from_frame(local_segment)

    manual = mne.Annotations(
        onset=[5 - start, 9 - start, 12 - start],
        duration=[0.0, 0.0, 0.0],
        description=["manual_add"] * 3,
    )
    combined_annotations = local_annotations + manual

    segment_frame = frame_from_annotations(combined_annotations, base_time=start)
    merged, _ = merge_annotations(frame, segment_frame, start, end)

    assert len(merged) == 33
    for target in (5, 9, 12):
        matches = merged[(np.isclose(merged["onset"], target)) & (merged["description"] == "manual_add")]
        assert not matches.empty


def test_middle_segment_manual_add_merge(tmp_path):
    csv_path = tmp_path / "mock.csv"
    _create_mock_csv(csv_path)
    frame = load_annotation_frame(csv_path)

    raw = _create_mock_raw()

    start, end = 100.0, 200.0
    assert raw.times[-1] > end
    inside, outside = split_annotations_by_window(frame, start, end)
    local_segment = inside.copy()
    local_segment["onset"] = (local_segment["onset"] - start).clip(lower=0.0)
    local_annotations = annotations_from_frame(local_segment)

    manual = mne.Annotations(
        onset=[110 - start, 120 - start],
        duration=[0.0, 0.0],
        description=["manual_add"] * 2,
    )
    combined_annotations = local_annotations + manual

    segment_frame = frame_from_annotations(combined_annotations, base_time=start)
    merged, _ = merge_annotations(frame, segment_frame, start, end)

    assert len(merged) == 32
    for target in (110, 120):
        matches = merged[(np.isclose(merged["onset"], target)) & (merged["description"] == "manual_add")]
        assert not matches.empty
    # Ensure original annotations remain both inside and outside the edited window
    assert not merged[merged["description"] == "B"].empty
    assert len(outside) + len(inside) == 30

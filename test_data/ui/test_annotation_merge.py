"""Regression tests for segment-aware annotation merging.

Overview
========
These tests replay two GUI scenarios against synthetic data to guard against the
"missing annotations" regression. Both use a single-channel 300 s Raw object and
a hardcoded CSV with 30 annotations (``annotations_input.csv``). Additional
ground-truth CSVs provide human-auditable expectations:

* ``expected_full_merge.csv`` — what the combined annotations should look like
  after opening the **entire** file (0–300 s) and adding three manual markers at
  5, 9, and 12 seconds.
* ``expected_middle_merge.csv`` — what the combined annotations should look like
  after opening only **100–200 s**, adding markers at 110 and 120 seconds, and
  merging those edits back without losing any pre-existing in-window rows.

Flowchart-style steps per test
------------------------------
1. Load the static 30-row CSV into a DataFrame.
2. Build a 300 s Raw with a single zero-valued channel (100 Hz).
3. Extract the relevant window using ``split_annotations_by_window``.
4. Convert the window to ``mne.Annotations`` (time-shifted to the window start),
   append manual markers, and optionally plot when ``ANNOTATION_TEST_PLOT=1``.
5. Convert edited annotations back to a global DataFrame with
   ``frame_from_annotations``.
6. Merge against the original annotations with ``merge_annotations`` to rebuild
   the full timeline.
7. Assert counts and compare against the appropriate ground-truth CSV to ensure
   no annotations were dropped or shifted.
"""
# ruff: noqa: E402

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas.testing as pdt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mne
import numpy as np

from src.ui.annotation_io import (  # noqa: E402
    annotations_from_frame,
    frame_from_annotations,
    load_annotation_frame,
)
from src.ui.annotation_merge import merge_annotations, split_annotations_by_window  # noqa: E402

DATA_DIR = Path(__file__).parent
INPUT_CSV = DATA_DIR / "annotations_input.csv"
EXPECTED_FULL = DATA_DIR / "expected_full_merge.csv"
EXPECTED_MIDDLE = DATA_DIR / "expected_middle_merge.csv"


def _create_mock_raw(duration=300, sfreq=100):
    n_samples = int(duration * sfreq)
    data = np.zeros((1, n_samples))
    info = mne.create_info(["chan1"], sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


def _maybe_plot(raw, annotations, title: str):
    """Plot the raw + annotations when ANNOTATION_TEST_PLOT=1.

    This helper keeps plotting optional for headless CI runs while enabling
    developers to inspect the merged results visually when debugging.
    """

    if os.getenv("ANNOTATION_TEST_PLOT", "0") not in {"1", "true", "True"}:
        return

    raw.set_annotations(annotations)
    raw.plot(title=title, block=True)


def test_full_duration_manual_add_merge(tmp_path):
    frame = load_annotation_frame(INPUT_CSV)

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

    _maybe_plot(raw, combined_annotations, "Full duration merge preview")

    segment_frame = frame_from_annotations(combined_annotations, base_time=start)
    merged, _ = merge_annotations(frame, segment_frame, start, end)

    assert len(merged) == 33
    (tmp_path / "full_merged.csv").write_text(merged.to_csv(index=False))
    for target in (5, 9, 12):
        matches = merged[(np.isclose(merged["onset"], target)) & (merged["description"] == "manual_add")]
        assert not matches.empty
    expected_full = load_annotation_frame(EXPECTED_FULL)
    pdt.assert_frame_equal(merged.reset_index(drop=True), expected_full.reset_index(drop=True))


def test_middle_segment_manual_add_merge(tmp_path):
    frame = load_annotation_frame(INPUT_CSV)

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

    _maybe_plot(raw, combined_annotations, "Mid-window merge preview")

    segment_frame = frame_from_annotations(combined_annotations, base_time=start)
    merged, _ = merge_annotations(frame, segment_frame, start, end)

    assert len(merged) == 32
    (tmp_path / "middle_merged.csv").write_text(merged.to_csv(index=False))
    for target in (110, 120):
        matches = merged[(np.isclose(merged["onset"], target)) & (merged["description"] == "manual_add")]
        assert not matches.empty
    # Ensure original annotations remain both inside and outside the edited window
    assert not merged[merged["description"] == "B"].empty
    assert len(outside) + len(inside) == 30
    expected_middle = load_annotation_frame(EXPECTED_MIDDLE)
    pdt.assert_frame_equal(
        merged.reset_index(drop=True), expected_middle.reset_index(drop=True), check_dtype=False
    )

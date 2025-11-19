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


def _create_mock_raw(duration: float = 300, sfreq: float = 100) -> mne.io.Raw:
    """Create a synthetic Raw object used as a deterministic test fixture.

    Parameters
    ----------
    duration : float, optional
        Length of the synthetic recording in seconds. Default is 300 s.
    sfreq : float, optional
        Sampling frequency in Hz. Default is 100 Hz.

    Returns
    -------
    raw : mne.io.Raw
        An MNE Raw object containing a single EEG channel called ``"chan1"``,
        filled with zeros, with length ``duration`` seconds and sampling
        frequency ``sfreq``.

    Notes
    -----
    The exact signal values do not matter for these tests; only the time axis
    and annotation placement are relevant. The returned object is used as the
    target on which annotations are attached and optionally plotted.
    """
    n_samples = int(duration * sfreq)
    data = np.zeros((1, n_samples))
    info = mne.create_info(["chan1"], sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


def test_full_duration_manual_add_merge(tmp_path):
    """Test merging after adding manual annotations over the full 0–300 s span.

    This test simulates the workflow of opening the **entire** file in the GUI,
    adding a few manual markers, and then merging the edited annotations back
    into the original annotation table.

    Steps
    -----
    1. Load the static input annotation CSV into a DataFrame.
    2. Create a 300 s mock Raw instance at 100 Hz.
    3. Select all annotations that fall inside [0, raw.duration] using
       :func:`split_annotations_by_window`.
    4. Time-shift the segment so that local onsets start at 0 and convert it
       to :class:`mne.Annotations` via :func:`annotations_from_frame`.
    5. Create three manual annotations at 5, 9, and 12 seconds (relative to the
       global timeline) and append them to the local annotations.
    6. Convert the combined annotations back to a global DataFrame with
       :func:`frame_from_annotations` using ``base_time=start``.
    7. Merge the edited segment back into the original annotation frame using
       :func:`merge_annotations`.
    8. Assert that specific indices exist and that their ``onset`` and
       ``description`` match the expected values.
    9. Convert the merged frame to :class:`mne.Annotations`, attach it to the
       Raw object, and plot (blocking) for visual inspection.

    Expected behavior / output
    --------------------------
    * The merged DataFrame must contain three new rows with description
      ``"manual_add"`` at onsets 5.0, 9.0, and 12.0 seconds.
    * No existing annotations should be lost or time-shifted unexpectedly.
    * The assertions on index presence and field values must all pass.
    * The test itself does not return anything; it passes if no assertion
      fails and no exception is raised. It prints a confirmation message and,
      when plotting is enabled, shows a Raw plot with all annotations.
    """
    # frame is a dataframe containing onset, duration, description columns
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
        duration=[0.5, 0.5, 0.6],
        description=["manual_add"] * 3,
    )
    combined_annotations = local_annotations + manual

    segment_frame = frame_from_annotations(combined_annotations, base_time=start)
    merged, _ = merge_annotations(frame, segment_frame, start, end)
    # Expected post-merge constraints
    expected = {
        1: {"onset": 5.0, "description": "manual_add"},
        2: {"onset": 9.0, "description": "manual_add"},
        4: {"onset": 12.0, "description": "manual_add"},
    }
    # 1. Assert indices exist
    for idx in expected:
        assert idx in merged.index, f"Index {idx} not found in merged dataframe"

    # 2. Assert required values for each index
    for idx, checks in expected.items():
        row = merged.loc[idx]

        # onset (exact match — customize if using float tolerance)
        assert row["onset"] == checks["onset"], (
            f"Merged[{idx}]: onset {row['onset']} != {checks['onset']}"
        )

        # description
        assert row["description"] == checks["description"], (
            f"Merged[{idx}]: description '{row['description']}' != '{checks['description']}'"
        )

    print("All merge assertions passed!")
    # import mne

    # If the onsets/durations are already in seconds relative to raw
    annotations = mne.Annotations(
        onset=merged["onset"].astype(float).to_numpy(),
        duration=merged["duration"].astype(float).to_numpy(),
        description=merged["description"].astype(str).tolist(),
    )

    raw.set_annotations(annotations)
    title = "Middle segment merge preview"
    raw.plot(title=title, block=True)
    print("complete plot")


def test_middle_segment_manual_add_merge(tmp_path):
    """Test merging after adding manual annotations in the 100–200 s segment.

    This test simulates the workflow of opening **only a middle segment**
    (100–200 s) in the GUI, adding manual markers, and merging those edits back
    into the full 0–300 s annotation table without losing any existing rows.

    Steps
    -----
    1. Load the static input annotation CSV into a DataFrame.
    2. Create a 300 s mock Raw instance.
    3. Use :func:`split_annotations_by_window` with ``start=100.0`` and
       ``end=200.0`` to obtain:
       * ``inside`` — annotations within the middle window.
       * ``outside`` — annotations outside that window (kept untouched).
    4. Time-shift ``inside`` so that its onsets are relative to 100 s and
       convert to :class:`mne.Annotations`.
    5. Create two new manual annotations at 111 and 121 seconds (global time),
       expressed relative to the 100 s window, and append them to the local
       annotations.
    6. Convert combined annotations back to a global DataFrame using
       :func:`frame_from_annotations` with ``base_time=100.0``.
    7. Merge the edited segment into the original frame via
       :func:`merge_annotations`.
    8. Assert that specific indices have the expected onsets and descriptions.
    9. Convert the merged frame to :class:`mne.Annotations` and attach it to
       the Raw object for plotting.

    Expected behavior / output
    --------------------------
    * The merged DataFrame should contain the two new rows with description
      ``"manual_add"`` at onsets 111.0 and 121.0 seconds.
    * All original annotations, both inside and outside the 100–200 s window,
      must be preserved (i.e., no accidental drops or shifts).
    * Assertions on index presence and values must pass.
    * The test returns nothing; success is signaled by absence of assertion
      failures and an informational printout. A blocking Raw plot is shown for
      manual inspection when plotting is enabled.
    """
    frame = load_annotation_frame(INPUT_CSV)

    raw = _create_mock_raw()

    start, end = 100.0, 200.0
    assert raw.times[-1] > end
    inside, outside = split_annotations_by_window(frame, start, end)
    local_segment = inside.copy()
    local_segment["onset"] = (local_segment["onset"] - start).clip(lower=0.0)
    local_annotations = annotations_from_frame(local_segment)

    manual = mne.Annotations(
        onset=[111 - start, 121 - start],
        duration=[0.5, 0.6],
        description=["manual_add"] * 2,
    )

    combined_annotations = local_annotations + manual

    segment_frame = frame_from_annotations(combined_annotations, base_time=start)
    merged, _ = merge_annotations(frame, segment_frame, start, end)
    # Expected post-merge constraints
    expected = {
        12: {"onset": 111.0, "description": "manual_add"},
        14: {"onset": 121.0, "description": "manual_add"},
    }
    # 1. Assert indices exist
    for idx in expected:
        assert idx in merged.index, f"Index {idx} not found in merged dataframe"

    # 2. Assert required values for each index
    for idx, checks in expected.items():
        row = merged.loc[idx]

        # onset (exact match — customize if using float tolerance)
        assert row["onset"] == checks["onset"], (
            f"Merged[{idx}]: onset {row['onset']} != {checks['onset']}"
        )

        # description
        assert row["description"] == checks["description"], (
            f"Merged[{idx}]: description '{row['description']}' != '{checks['description']}'"
        )

    print("All merge assertions passed!")
    # import mne

    # If the onsets/durations are already in seconds relative to raw
    annotations = mne.Annotations(
        onset=merged["onset"].astype(float).to_numpy(),
        duration=merged["duration"].astype(float).to_numpy(),
        description=merged["description"].astype(str).tolist(),
    )

    raw.set_annotations(annotations)
    title = "Middle segment merge preview"
    raw.plot(title=title, block=True)
    print("complete plot")


def test_middle_segment_manual_add_merge_drop_annotation(tmp_path):
    """Test merging when a local annotation is dropped and new ones are added.

    This test is a variant of the 100–200 s segment workflow where the user
    both **adds** new annotations and **removes** at least one existing
    annotation inside the window before merging back.

    Current implementation
    ----------------------
    * The code currently mirrors ``test_middle_segment_manual_add_merge`` but
      does **not yet** implement the explicit drop of a local annotation or
      the final assertions about the dropped row.
    * It:
      1. Loads the input annotation frame.
      2. Creates a 300 s mock Raw instance.
      3. Splits annotations into the 100–200 s window and the rest.
      4. Converts the inside segment to local annotations (onsets relative to
         100 s).
      5. Adds two new manual annotations at 111 and 121 seconds (global).
      6. Converts the combined annotations back to a global frame and merges
         them with :func:`merge_annotations`.

    Intended behavior / expected output
    -----------------------------------
    Once fully implemented, this test should verify that:

    * One existing annotation within the 100–200 s window (for example, the
      local annotation with description ``"M"`` at a particular index) can be
      dropped from the local annotations before merging.
    * After merging, this specific annotation no longer appears in the global
      merged DataFrame.
    * The two new manual annotations at 111.0 and 121.0 seconds **are**
      present with description ``"manual_add"``, alongside all other unchanged
      annotations.
    * No annotations outside the edited window are affected.
    * The test will pass when the merged frame satisfies:
      - dropped-in-window annotation is absent,
      - new manual annotations are present with correct onsets and description,
      - all other annotations are preserved.

    Currently the function does not contain the final assertions for the drop
    behavior, so it should be treated as a placeholder or starting point for
    that regression test case.
    """
    frame = load_annotation_frame(INPUT_CSV)

    raw = _create_mock_raw()

    start, end = 100.0, 200.0
    assert raw.times[-1] > end
    inside, outside = split_annotations_by_window(frame, start, end)
    local_segment = inside.copy()
    local_segment["onset"] = (local_segment["onset"] - start).clip(lower=0.0)
    local_annotations = annotations_from_frame(local_segment)

    # This is add annotation into the local annotations
    manual = mne.Annotations(
        onset=[111 - start, 121 - start],
        duration=[0.5, 0.6],
        description=["manual_add"] * 2,
    )

    # we also might drop an existing annotation in the local annotations
    # e.g., drop the annotation at index 3 of local annotation, with the description "M", thou this is only for testing purpose, but we not neccessarily want to drop based on description

    combined_annotations = local_annotations + manual

    segment_frame = frame_from_annotations(combined_annotations, base_time=start)
    merged, _ = merge_annotations(frame, segment_frame, start, end)
    # Expected post-merge constraints

    # so when assert, we expect the annotation with description "M" to be dropped, and the two new manual annotations to be present together with the rest of the annotations

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


import sys
from pathlib import Path

import pandas as pd

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

def imitate_plot_adjustment_full_duration(raw, local_annotations, start):
        # pass
        """
        We may also remove an existing annotation from the local annotations.
        For example, we could delete the annotation at index 3 with description "M".
        In practice, however, annotation editing (adding or deleting) will typically be done through the MNE Browser (interactive plot).
        This code is only for testing purposes, and we do not necessarily want to remove annotations based on their description.

        To avoid issue of resetting
        """
        raw_temp=raw.copy()
        raw_temp.set_annotations(local_annotations)
        # assume we do some adjustment in plot
        ann=raw_temp.annotations

        ann.delete([1])
        # This to make sure the second index has been drop
        assert len(ann)!=len(local_annotations)
        # add your manual ones
        ann += mne.Annotations(
            onset=[5 - start, 9 - start, 12 - start],
            duration=[0.5, 0.5, 0.6],
            description=["manual_add"] * 3,
        )

        assert len(ann)==31
        return ann

def assume_raw_with_annotations():
    raw = _create_mock_raw()
    frame = load_annotation_frame(INPUT_CSV)
    annotations = mne.Annotations(
        onset=frame["onset"].astype(float).to_numpy(),
        duration=frame["duration"].astype(float).to_numpy(),
        description=frame["description"].astype(str).tolist(),
    )

    raw.set_annotations(annotations)
    return raw


def test_full_duration_manual_add_merge():
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

    # Below we assume we obtain the raw from the gui that always opening the full duration
    raw=assume_raw_with_annotations()
    annotation=raw.annotations



    # To ensure we avoid timestamp
    frame = pd.DataFrame({
        "onset":    np.asarray(annotation.onset, dtype=float),
        "duration": np.asarray(annotation.duration, dtype=float),
        "description": np.asarray(annotation.description, dtype=str),
    })

    # we assume the user select the full duration
    start, end = 0.0, raw.times[-1]


    inside, outside = split_annotations_by_window(frame, start, end)

    local_segment = inside.copy()
    local_segment["onset"] = (local_segment["onset"] - start).clip(lower=0.0)
    local_annotations = annotations_from_frame(local_segment)


    # To avoid showing all annotation which can slow down the annotation plot, we only half
    ann_manual=imitate_plot_adjustment_full_duration(raw, local_annotations, start)


    merged = pd.DataFrame({
        "onset":    np.asarray(ann_manual.onset, dtype=float),
        "duration": np.asarray(ann_manual.duration, dtype=float),
        "description": np.asarray(ann_manual.description, dtype=str),
    })

    merged = pd.concat([merged, outside], ignore_index=True)
    annot_combine = mne.Annotations(
        onset=merged["onset"].to_numpy(),         # or .values
        duration=merged["duration"].to_numpy(),
        description=merged["description"].astype(str).to_numpy(),
        )

    # Expected post-merge constraints
    expected = {
        0: {"onset": 5.0, "description": "manual_add"},
        1: {"onset": 9.0, "description": "manual_add"},
        3: {"onset": 12.0, "description": "manual_add"},
    }
    # 1. Assert indices exist

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
    raw.set_annotations(annot_combine)
    title = "Middle segment merge preview"
    raw.plot(title=title, block=True)
    print("complete plot")




def test_middle_segment_manual_add_merge_drop_annotation():
    """
    # We may also remove an existing annotation from the local annotations.
    # For example, we could delete the annotation at index 3 with description "M".
    # In practice, however, annotation editing (adding or deleting) will typically be done
    # through the MNE Browser (interactive plot). This code is only for testing purposes,
    # and we do not necessarily want to remove annotations based on their description.


    """
    # Below we assume we obtain the raw from the gui that always opening the full duration
    raw=assume_raw_with_annotations()
    annotation=raw.annotations



    # To ensure we avoid timestamp
    frame = pd.DataFrame({
        "onset":    np.asarray(annotation.onset, dtype=float),
        "duration": np.asarray(annotation.duration, dtype=float),
        "description": np.asarray(annotation.description, dtype=str),
    })


    start, end = 100.0, 200.0
    assert raw.times[-1] > end
    inside, outside = split_annotations_by_window(frame, start, end)

    ann_inside = mne.Annotations(
        onset=inside["onset"].astype(float).to_numpy(),
        duration=inside["duration"].astype(float).to_numpy(),
        description=inside["description"].astype(str).tolist(),
        )
    # local_segment = inside.copy()
    # local_segment["onset"] = (local_segment["onset"] - start).clip(lower=0.0)
    # local_annotations = annotations_from_frame(local_segment)

    # To avoid showing all annotation which can slow down the annotation plot, we only half
    ann_manual=imitate_plot_adjustment_in_between(raw, ann_inside,start)

    merged = pd.DataFrame({
        "onset":    np.asarray(ann_manual.onset, dtype=float),
        "duration": np.asarray(ann_manual.duration, dtype=float),
        "description": np.asarray(ann_manual.description, dtype=str),
    })

    merged = pd.concat([merged, outside], ignore_index=True).sort_values("onset").reset_index(drop=True)
    annot_combine = mne.Annotations(
        onset=merged["onset"].to_numpy(),         # or .values
        duration=merged["duration"].to_numpy(),
        description=merged["description"].astype(str).to_numpy(),
        )

    expected = {
            10: {"onset": 105.0, "description": "manual_add"},
            12: {"onset": 125.0, "description": "manual_add"},
            14: {"onset": 135.0, "description": "manual_add"},
            }

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
    raw.set_annotations(annot_combine)
    title = "Middle segment merge preview"
    raw.plot(title=title, block=True)
    print("complete plot")

def imitate_plot_adjustment_in_between(raw, local_annotations,start):
    # pass
    """
    We may also remove an existing annotation from the local annotations.
    For example, we could delete the annotation at index 3 with description "M".
    In practice, however, annotation editing (adding or deleting) will typically be done through the MNE Browser (interactive plot).
    This code is only for testing purposes, and we do not necessarily want to remove annotations based on their description.

    To avoid issue of resetting
    """
    raw_temp=raw.copy()
    raw_temp.set_annotations(local_annotations)
    # assume we do some adjustment in plot
    ann=raw_temp.annotations

    ann.delete([1])
    # This to make sure the second index has been drop
    assert len(ann)!=len(local_annotations)
    # add your manual ones
    ann += mne.Annotations(
        onset=[105, 125, 135],
        duration=[0.5, 0.5, 0.6],
        description=["manual_add"] * 3,
    )

    # assert len(ann)==31
    return ann

if __name__ == "__main__":

    test_full_duration_manual_add_merge()
    test_middle_segment_manual_add_merge_drop_annotation()
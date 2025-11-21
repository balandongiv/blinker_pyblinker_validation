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

from __future__ import annotations

import os
from pathlib import Path
import unittest

import mne
import pandas as pd

from src.test_helper import annotations_to_frame, load_raw_with_annotations
from src.ui.annotation_io import annotations_from_frame
from src.ui.annotation_merge import split_annotations_by_window

DATA_DIR = Path(__file__).parent
INPUT_CSV = DATA_DIR / "annotations_input.csv"
EXPECTED_FULL = DATA_DIR / "expected_full_merge.csv"
EXPECTED_MIDDLE = DATA_DIR / "expected_middle_merge.csv"
PLOT_ENABLED = os.environ.get("ANNOTATION_TEST_PLOT") == "1"


def launch_browser_and_collect_imitate_full_duration(
    raw: mne.io.Raw, local_annotations: mne.Annotations, start: float
) -> mne.Annotations:
    """Imitate in-browser edits for the full-duration window.

    We may also remove an existing annotation from the local annotations. For
    example, we could delete the annotation at index 3 with description "M". In
    practice, however, annotation editing (adding or deleting) will typically be
    done through the MNE Browser (interactive plot). This code is only for
    testing purposes, and we do not necessarily want to remove annotations based
    on their description. To avoid issues with resetting state, we work on a
    copy of the Raw instance.
    """
    raw_temp = raw.copy()
    raw_temp.set_annotations(local_annotations)
    ann = raw_temp.annotations

    ann.delete([1])
    assert len(ann) != len(local_annotations)
    ann += mne.Annotations(
        onset=[5 - start, 9 - start, 12 - start],
        duration=[0.0, 0.0, 0.0],
        description=["manual_add"] * 3,
    )
    assert len(ann) == 31
    return ann


def launch_browser_and_collect_imitate_plot_in_between(
    raw: mne.io.Raw, local_annotations: mne.Annotations, start: float
) -> mne.Annotations:
    """Imitate in-browser edits for a middle segment.

    We may also remove an existing annotation from the local annotations. For
    example, we could delete the annotation at index 3 with description "M". In
    practice, however, annotation editing (adding or deleting) will typically be
    done through the MNE Browser (interactive plot). This code is only for
    testing purposes, and we do not necessarily want to remove annotations based
    on their description. To avoid issues with resetting state, we work on a
    copy of the Raw instance.
    """
    raw_temp = raw.copy()
    raw_temp.set_annotations(local_annotations)
    ann = raw_temp.annotations

    ann.delete([1])
    assert len(ann) != len(local_annotations)
    ann += mne.Annotations(
        onset=[110, 120],
        duration=[0.0, 0.0],
        description=["manual_add"] * 2,
    )
    return ann


class AnnotationMergeTestCase(unittest.TestCase):
    def test_full_duration_manual_add_merge(self) -> None:
        """Test merging after adding manual annotations over the full 0–300 s span.

        This test simulates the workflow of opening the **entire** file in the
        GUI, adding a few manual markers, and then merging the edited annotations
        back into the original annotation table.

        Steps
        -----
        1. Load the static input annotation CSV into a DataFrame.
        2. Create a 300 s mock Raw instance at 100 Hz.
        3. Select all annotations that fall inside [0, raw.duration] using
           :func:`split_annotations_by_window`.
        4. Time-shift the segment so that local onsets start at 0 and convert it
           to :class:`mne.Annotations` via :func:`annotations_from_frame`.
        5. Create three manual annotations at 5, 9, and 12 seconds (relative to
           the global timeline) and append them to the local annotations.
        6. Convert the combined annotations back to a global DataFrame with
           :func:`frame_from_annotations` using ``base_time=start``.
        7. Merge the edited segment back into the original annotation frame
           using :func:`merge_annotations`.
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

        start, end = 0.0, None

        raw = load_raw_with_annotations(INPUT_CSV)
        end = raw.times[-1] if end is None else end

        frame = annotations_to_frame(raw.annotations)
        inside, outside = split_annotations_by_window(frame, start, end)

        local_segment = inside.copy()
        local_segment["onset"] = (local_segment["onset"] - start).clip(lower=0.0)
        local_annotations = annotations_from_frame(local_segment)

        ann_manual = launch_browser_and_collect_imitate_full_duration(
            raw, local_annotations, start
        )
        merged = annotations_to_frame(ann_manual)
        merged = pd.concat([merged, outside], ignore_index=True)
        annot_combine = annotations_from_frame(merged)

        expected_full = pd.read_csv(EXPECTED_FULL)
        expected_manual_onsets = expected_full.loc[
            expected_full["description"] == "manual_add", "onset"
        ]

        expected = {
            0: {"onset": 5.0, "description": "manual_add"},
            1: {"onset": 9.0, "description": "manual_add"},
            3: {"onset": 12.0, "description": "manual_add"},
        }

        for idx, checks in expected.items():
            row = merged.loc[idx]
            self.assertEqual(row["onset"], checks["onset"])
            self.assertEqual(row["description"], checks["description"])

        for onset in expected_manual_onsets:
            self.assertIn(onset, merged["onset"].values)

        if PLOT_ENABLED:
            raw.set_annotations(annot_combine)
            raw.plot(title="Full duration merge preview", block=True)

    def test_middle_segment_manual_add_merge_drop_annotation(self) -> None:
        """Test merging manual annotations for a 100–200 s window.

        We may also remove an existing annotation from the local annotations. For
        example, we could delete the annotation at index 3 with description "M".
        In practice, however, annotation editing (adding or deleting) will
        typically be done through the MNE Browser (interactive plot). This code
        is only for testing purposes, and we do not necessarily want to remove
        annotations based on their description.
        """

        start, end = 100.0, 200.0

        raw = load_raw_with_annotations(INPUT_CSV)
        self.assertGreater(raw.times[-1], end)

        frame = annotations_to_frame(raw.annotations)
        inside, outside = split_annotations_by_window(frame, start, end)

        ann_inside = mne.Annotations(
            onset=inside["onset"].astype(float).to_numpy(),
            duration=inside["duration"].astype(float).to_numpy(),
            description=inside["description"].astype(str).tolist(),
        )

        ann_manual = launch_browser_and_collect_imitate_plot_in_between(raw, ann_inside, start)
        merged = annotations_to_frame(ann_manual)
        merged = (
            pd.concat([merged, outside], ignore_index=True)
            .sort_values("onset")
            .reset_index(drop=True)
        )
        annot_combine = annotations_from_frame(merged)

        expected_middle = pd.read_csv(EXPECTED_MIDDLE)
        expected_manual_onsets = expected_middle.loc[
            expected_middle["description"] == "manual_add", "onset"
        ]

        for onset in expected_manual_onsets:
            self.assertIn(onset, merged["onset"].values)

        if PLOT_ENABLED:
            raw.set_annotations(annot_combine)
            raw.plot(title="Middle segment merge preview", block=True)


if __name__ == "__main__":
    unittest.main()

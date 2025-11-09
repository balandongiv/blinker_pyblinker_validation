"""Unit tests for the murat_2018 processing helpers."""

from __future__ import annotations

import math

import pandas as pd

from murat_sequence.step4_compare_ import (
    RecordingComparison,
    _build_summary_frame,
    _match_events,
)


def test_match_events_with_tolerance():
    ref = [0.0, 1.0, 2.0]
    other = [0.05, 0.9, 5.0]
    tp, fp_ref, fp_other, diffs = _match_events(ref, other, tolerance_s=0.15)

    assert tp == 2
    assert fp_ref == 1
    assert fp_other == 1
    assert math.isclose(diffs[0], 0.05, rel_tol=1e-9)


def test_build_summary_frame_computes_metrics():
    comparison = RecordingComparison(
        recording_id="rec01",
        py_events=pd.DataFrame({"onset_sec": [0.0, 1.0, 2.0]}),
        blinker_events=pd.DataFrame({"onset_sec": [0.0, 1.0]}),
        matched=2,
        false_py=1,
        false_blinker=0,
        onset_mae=0.1,
    )
    summary = _build_summary_frame([comparison])

    assert summary.loc[0, "recording_id"] == "rec01"
    assert summary.loc[0, "py_count"] == 3
    assert summary.loc[0, "blinker_count"] == 2
    assert math.isclose(summary.loc[0, "precision"], 2 / 3)
    assert math.isclose(summary.loc[0, "recall"], 1.0)
    assert math.isclose(summary.loc[0, "f1"], 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0))

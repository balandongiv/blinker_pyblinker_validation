"""Unit tests for the murat_2018 processing helpers."""

from __future__ import annotations

import math

import pandas as pd

from murat_sequence.step4_compare_ import RecordingComparison, _build_summary_frame


def test_build_summary_frame_computes_metrics():
    comparison = RecordingComparison(
        recording_id="rec01",
        py_events=pd.DataFrame({"start_blink": [100, 200, 300], "end_blink": [120, 220, 320]}),
        blinker_events=pd.DataFrame({"start_blink": [110, 210], "end_blink": [130, 230]}),
        metrics={
            "total_detected": 3.0,
            "total_ground_truth": 2.0,
            "matches_within_tolerance": 2.0,
            "detected_only": 1.0,
            "ground_truth_only": 0.0,
            "pairs_outside_tolerance": 0.0,
        },
        onset_mae_sec=0.1,
    )
    summary = _build_summary_frame([comparison])

    assert summary.loc[0, "recording_id"] == "rec01"
    assert math.isclose(summary.loc[0, "py_count"], 3.0)
    assert math.isclose(summary.loc[0, "blinker_count"], 2.0)
    assert math.isclose(summary.loc[0, "precision"], 2 / 3, rel_tol=1e-9)
    assert math.isclose(summary.loc[0, "recall"], 1.0, rel_tol=1e-9)
    assert math.isclose(summary.loc[0, "f1"], 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0), rel_tol=1e-9)

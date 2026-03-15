from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from src.validation.raja_pyblinker import (
    SegmentRecord,
    annotation_table_to_events,
    compare_segment_with_human_annotations,
    discover_segments_for_subject,
    iter_segments,
    resolve_existing_pyblinker_output_path,
    resolve_pyblinker_output_path,
)


def test_discover_and_filter_segments(tmp_path):
    processed_root = tmp_path / "processed"
    subject_dir = processed_root / "S1"
    (subject_dir / "S01_20170519_043933").mkdir(parents=True)
    (subject_dir / "S01_20170519_043933_2").mkdir(parents=True)
    (subject_dir / "blinker_pyblinker_validation").mkdir(parents=True)
    (processed_root / "S2" / "S02_foo").mkdir(parents=True)

    segments = discover_segments_for_subject(subject_dir)
    assert [segment.segment_id for segment in segments] == [
        "S01_20170519_043933",
        "S01_20170519_043933_2",
    ]

    assert [
        (segment.subject_id, segment.segment_id)
        for segment in iter_segments(processed_root, filter_subject_id="S1")
    ] == [
        ("S1", "S01_20170519_043933"),
        ("S1", "S01_20170519_043933_2"),
    ]

    assert [
        (segment.subject_id, segment.segment_id)
        for segment in iter_segments(processed_root, filter_filename="S01_20170519_043933_2")
    ] == [("S1", "S01_20170519_043933_2")]

    assert [
        (segment.subject_id, segment.segment_id)
        for segment in iter_segments(
            processed_root,
            filter_subject_id="S1",
            filter_filename="S01_20170519_043933",
        )
    ] == [("S1", "S01_20170519_043933")]


def test_annotation_table_to_events_coerces_to_one_based():
    annotation_table = pd.DataFrame(
        {
            "onset": [0.0, 1.25],
            "duration": [0.2, 0.15],
            "description": ["Blink", "Blink"],
        }
    )

    events = annotation_table_to_events(annotation_table, sampling_rate_hz=100.0)

    assert list(events.columns) == ["start_blink", "end_blink"]
    assert events.to_dict(orient="records") == [
        {"start_blink": 1, "end_blink": 21},
        {"start_blink": 126, "end_blink": 141},
    ]


def test_resolve_existing_pyblinker_output_prefers_canonical_path(tmp_path):
    segment_dir = tmp_path / "S1" / "S01_20170519_043933"
    canonical = resolve_pyblinker_output_path(segment_dir)
    legacy = segment_dir / "seg_data_raw" / "pyblinker_results.pkl"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"canonical")
    legacy.write_bytes(b"legacy")

    assert resolve_existing_pyblinker_output_path(segment_dir) == canonical


def test_compare_segment_with_human_annotations_uses_saved_payload(tmp_path):
    processed_root = tmp_path / "processed"
    human_root = tmp_path / "human"
    segment_dir = processed_root / "S1" / "S01_20170519_043933"
    segment = SegmentRecord("S1", "S01_20170519_043933", segment_dir)

    py_path = resolve_pyblinker_output_path(segment_dir)
    py_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "comparison_events": pd.DataFrame(
            [
                {"start_blink": 10, "end_blink": 20},
                {"start_blink": 50, "end_blink": 60},
            ]
        ),
        "representative_signal": np.concatenate(
            [
                np.zeros(10, dtype=np.float32),
                np.ones(11, dtype=np.float32),
                np.zeros(29, dtype=np.float32),
                np.ones(11, dtype=np.float32),
                np.zeros(39, dtype=np.float32),
            ]
        ),
        "metrics": {"channel": "E8", "sampling_rate_hz": 100.0},
        "params": {"resample_rate": 100.0},
    }
    with py_path.open("wb") as handle:
        pickle.dump(payload, handle)

    annotation_path = human_root / "S1" / "S01_20170519_043933" / "ear_eog.csv"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "onset": [0.10, 0.50],
            "duration": [0.10, 0.10],
            "description": ["HB_CL", "HB_CL"],
        }
    ).to_csv(annotation_path, index=False)

    status, result = compare_segment_with_human_annotations(
        segment,
        human_annotation_root=human_root,
        tolerance_samples=0,
    )

    assert status.status == "compared"
    assert result is not None
    assert result.recording_comparison.metrics["total_detected"] == 2.0
    assert result.recording_comparison.metrics["total_ground_truth"] == 2.0
    assert result.recording_comparison.metrics["share_within_tolerance"] == 4.0

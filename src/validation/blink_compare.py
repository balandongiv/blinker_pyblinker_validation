"""Utilities for comparing PyBlinker and MATLAB Blinker outputs."""

from __future__ import annotations

import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Mapping

import mne
import numpy as np
import pandas as pd
from blink_evaluation import evaluate_annotations


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.stat import RecordingComparison


LOGGER = logging.getLogger(__name__)


def _coerce_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def events_to_annotations(events_df: pd.DataFrame, sfreq: float) -> mne.Annotations:
    """Convert a start_blink/end_blink DataFrame (1-indexed samples) to mne.Annotations."""
    if events_df.empty:
        return mne.Annotations(onset=[], duration=[], description=[])
    start = events_df["start_blink"].to_numpy(dtype=float)
    end = events_df["end_blink"].to_numpy(dtype=float)
    return mne.Annotations(
        onset=(start - 1.0) / sfreq,
        duration=(end - start) / sfreq,
        description="blink",
    )


def build_comparison_metrics(result) -> dict[str, float]:
    """Map a blink_evaluation EvaluationResult to the RecordingComparison metrics dict."""
    tp = result.event_metrics.tp
    fp = result.event_metrics.fp
    fn = result.event_metrics.fn
    total_gt = tp + fn
    total_pred = tp + fp
    unique_total = total_gt + total_pred
    share = 2.0 * tp
    share_pct = (share / unique_total * 100.0) if unique_total else float("nan")
    return {
        "total_ground_truth": float(total_gt),
        "total_detected": float(total_pred),
        "ground_truth_only": float(fn),
        "detected_only": float(fp),
        "share_within_tolerance": share,
        "matches_within_tolerance": 0.0,
        "pairs_outside_tolerance": 0.0,
        "unique_total": float(unique_total),
        "share_within_tolerance_percent": share_pct,
    }


def compare_events(
    py_payload: Mapping,
    blinker_payload: Mapping,
    sfreq: float,
    recording_duration: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Compare PyBlinker and MATLAB Blinker events using blink_evaluation."""
    py_events, blinker_events = prepare_event_tables(py_payload, blinker_payload)
    result = evaluate_annotations(
        events_to_annotations(blinker_events, sfreq),
        events_to_annotations(py_events, sfreq),
        target_label="blink",
        iou_threshold=0.5,
        sample_rate=sfreq,
        recording_duration=recording_duration,
    )
    return py_events, blinker_events, build_comparison_metrics(result)


def _load_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported raw file type: {path}")


def load_pickle(path: str | Path):
    """Load a pickle payload from ``path``."""

    path = _coerce_path(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def extract_events(payload: Mapping, *, fallback_key: str) -> pd.DataFrame:
    """Extract event tables from ``payload``."""

    events = payload.get("events")
    if isinstance(events, pd.DataFrame):
        return events
    if events is not None:
        return pd.DataFrame(events)

    frames = payload.get("frames", {})
    if isinstance(frames, Mapping) and fallback_key in frames:
        candidate = frames[fallback_key]
        if isinstance(candidate, pd.DataFrame):
            return candidate
        return pd.DataFrame(candidate)
    return pd.DataFrame()


def to_samples(
    start: pd.Series,
    end: pd.Series,
    *,
    source_rate: float | None,
    target_rate: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    start_vals = pd.to_numeric(start, errors="coerce").to_numpy(dtype=float)
    end_vals = pd.to_numeric(end, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(start_vals) & np.isfinite(end_vals)
    start_vals = start_vals[mask]
    end_vals = end_vals[mask]

    if source_rate and target_rate and not math.isclose(source_rate, target_rate):
        scale = target_rate / source_rate
        start_vals = np.round(start_vals * scale)
        end_vals = np.round(end_vals * scale)

    return start_vals, end_vals


def normalise_events(
    frame: pd.DataFrame,
    *,
    sample_rate: float | None,
    target_rate: float | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["start_blink", "end_blink"], dtype=int)

    columns = {col.lower(): col for col in frame.columns}

    def _pick(*names: str) -> str | None:
        for name in names:
            key = name.lower()
            if key in columns:
                return columns[key]
        return None

    start_col = _pick("start_blink", "start", "leftzero", "left_zero")
    end_col = _pick("end_blink", "end", "rightzero", "right_zero")
    source_rate = sample_rate

    if start_col and end_col:
        start_vals, end_vals = to_samples(
            frame[start_col],
            frame[end_col],
            source_rate=source_rate,
            target_rate=target_rate,
        )
    else:
        onset_col = _pick("onset_sec", "latency_sec", "latency")
        duration_col = _pick("duration_sec", "duration")
        if onset_col and duration_col and sample_rate:
            onset_samples = pd.to_numeric(frame[onset_col], errors="coerce") * sample_rate
            duration_samples = pd.to_numeric(frame[duration_col], errors="coerce") * sample_rate
            start_vals = onset_samples.to_numpy(dtype=float)
            end_vals = start_vals + duration_samples.to_numpy(dtype=float)
        else:
            return pd.DataFrame(columns=["start_blink", "end_blink"], dtype=int)

        start_vals, end_vals = to_samples(
            pd.Series(start_vals),
            pd.Series(end_vals),
            source_rate=source_rate,
            target_rate=target_rate,
        )

    start_vals = start_vals.astype(int, copy=False)
    end_vals = end_vals.astype(int, copy=False)
    mask = end_vals > start_vals
    normalised = pd.DataFrame({"start_blink": start_vals[mask], "end_blink": end_vals[mask]})
    if target_rate is not None:
        normalised.attrs["sampling_rate_hz"] = float(target_rate)
    normalised = normalised.sort_values("start_blink", kind="mergesort").reset_index(drop=True)
    return normalised


def prepare_event_tables(
    py_payload: Mapping,
    blinker_payload: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare comparable event tables from PyBlinker and MATLAB Blinker payloads."""

    py_events = (
        py_payload["events"][["left_zero", "right_zero", "max_value"]]
        .rename(
            columns={
                "left_zero": "start_blink",
                "right_zero": "end_blink",
                "max_value": "maxValue",
            }
        )
        .copy()
    )
    py_events[["start_blink", "end_blink"]] = py_events[["start_blink", "end_blink"]].astype(int) + 1
    py_events = py_events.sort_values(by="start_blink", kind="mergesort").reset_index(drop=True)

    blinker_events = (
        blinker_payload["frames"]["blinkFits"][["leftZero", "rightZero", "maxValue"]]
        .rename(columns={"leftZero": "start_blink", "rightZero": "end_blink"})
        .sort_values(by="start_blink", kind="mergesort")
        .reset_index(drop=True)
    )
    return py_events, blinker_events



def process_recording_comparison(
    recording_dir: str | Path,
    py_path: str | Path,
    blinker_path: str | Path,
    fif_path: str | Path,
    fif_fname: str,
    *,
    tolerance_samples: int,
    overwrite: bool,
) -> RecordingComparison:
    del fif_fname, overwrite, tolerance_samples

    recording_dir = _coerce_path(recording_dir)
    py_path = _coerce_path(py_path)
    blinker_path = _coerce_path(blinker_path)
    fif_path = _coerce_path(fif_path)

    py_payload = load_pickle(py_path)
    blinker_payload = load_pickle(blinker_path)

    raw = _load_raw(fif_path)
    sfreq = float(raw.info["sfreq"])
    recording_duration = float(raw.times[-1])

    py_events, blinker_events, metrics = compare_events(
        py_payload, blinker_payload, sfreq, recording_duration
    )

    return RecordingComparison(
        recording_id=recording_dir.name,
        py_events=py_events,
        blinker_events=blinker_events,
        metrics=metrics,
    )



def main() -> int:
    recording_id = "12400406"
    recording_dir = Path(f"D:/dataset/murat_2018/{recording_id}")
    process_recording_comparison(
        recording_dir,
        recording_dir / "pyblinker_results.pkl",
        recording_dir / "blinker_results.pkl",
        recording_dir / f"{recording_id}.fif",
        recording_id,
        tolerance_samples=20,
        overwrite=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

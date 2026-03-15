"""Utilities for comparing PyBlinker and MATLAB Blinker outputs."""

from __future__ import annotations

import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Iterable, Mapping

import mne
import numpy as np
import pandas as pd
from pyblinker.utils.evaluation import blink_comparison


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.stat import RecordingComparison


LOGGER = logging.getLogger(__name__)


def _coerce_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


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


def iter_recordings(root: Path) -> Iterable[Path]:
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir():
            yield candidate


def render_report(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    overall: pd.Series | None = None,
) -> Path:
    lines = ["# murat_2018 PyBlinker vs Blinker comparison", ""]
    lines.append(
        "| Recording | Detected | Ground truth | share_within_tolerance | matches_within_tolerance | Py-only | Blinker-only | Precision (strict) | Recall (strict) | F1 (strict) | Precision (lenient) | Recall (lenient) | F1 (lenient) |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    for _, row in summary.iterrows():
        def _fmt(value):
            if pd.isna(value):
                return "-"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(int(value))

        lines.append(
            "| {recording_id} | {py_count} | {blinker_count} | {share} | {matches} | {py_only} | {blinker_only} | {precision_strict} | {recall_strict} | {f1_strict} | {precision_lenient} | {recall_lenient} | {f1_lenient} |".format(
                recording_id=row["recording_id"],
                py_count=_fmt(row["total_detected"]),
                blinker_count=_fmt(row["total_ground_truth"]),
                share=_fmt(row["share_within_tolerance"]),
                matches=_fmt(row["matches_within_tolerance"]),
                py_only=_fmt(row["detected_only"]),
                blinker_only=_fmt(row["ground_truth_only"]),
                precision_strict=_fmt(row["precision_strict"]),
                recall_strict=_fmt(row["recall_strict"]),
                f1_strict=_fmt(row["f1_strict"]),
                precision_lenient=_fmt(row["precision_lenient"]),
                recall_lenient=_fmt(row["recall_lenient"]),
                f1_lenient=_fmt(row["f1_lenient"]),
            )
        )

    if overall is not None and not overall.empty:
        lines.extend(["", "## Overall summary", "", "| Metric | Value |", "| --- | ---: |"])

        def _fmt_overall(value: float) -> str:
            if pd.isna(value):
                return "-"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        for metric, value in overall.items():
            lines.append(f"| {metric} | {_fmt_overall(value)} |")

    report_path = output_dir / "summary_report.md"
    report_path.write_text("\n".join(lines), encoding="utf8")
    return report_path


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
    del fif_fname, overwrite

    recording_dir = _coerce_path(recording_dir)
    py_path = _coerce_path(py_path)
    blinker_path = _coerce_path(blinker_path)
    fif_path = _coerce_path(fif_path)

    py_payload = load_pickle(py_path)
    blinker_payload = load_pickle(blinker_path)

    channel = py_payload["metrics"]["channel"]
    raw = _load_raw(fif_path)
    signal = raw.get_data(picks=[channel])[0]
    py_events, blinker_events = prepare_event_tables(py_payload, blinker_payload)

    comparison = blink_comparison.compare_detected_vs_ground_truth(
        py_events,
        blinker_events,
        float(raw.info["sfreq"]),
        tolerance_samples=tolerance_samples,
        n_preview_rows=10,
        n_diff_rows=20,
        detected_signal=signal,
    )

    return RecordingComparison(
        recording_id=recording_dir.name,
        py_events=py_events,
        blinker_events=blinker_events,
        metrics=comparison.metrics,
    )


def compare_recordings_blinker_vs_pyblinker(
    root: Path,
    *,
    tolerance_samples: int,
    overwrite: bool = False,
) -> list[RecordingComparison]:
    """Compute per-recording comparisons for a dataset root."""

    comparisons: list[RecordingComparison] = []

    for recording_dir in iter_recordings(root):
        py_path = recording_dir / "pyblinker_results.pkl"
        blinker_path = recording_dir / "blinker_results.pkl"
        fif_path = recording_dir / f"{recording_dir.name}.fif"

        if not py_path.exists() or not blinker_path.exists():
            LOGGER.debug("Skipping %s because expected pickle files are missing", recording_dir.name)
            continue

        if not fif_path.exists():
            LOGGER.info("Skipping %s because raw file is missing: %s", recording_dir.name, fif_path)
            continue

        comparisons.append(
            process_recording_comparison(
                recording_dir,
                py_path,
                blinker_path,
                fif_path,
                recording_dir.name,
                tolerance_samples=tolerance_samples,
                overwrite=overwrite,
            )
        )

    return comparisons


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

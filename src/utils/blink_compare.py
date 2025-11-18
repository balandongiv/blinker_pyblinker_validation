"""Utilities for comparing PyBlinker and MATLAB Blinker outputs."""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import mne
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RecordingComparison:
    """Container for per-recording comparison results."""

    recording_id: str
    py_events: pd.DataFrame
    blinker_events: pd.DataFrame
    metrics: Mapping[str, float]
    onset_mae_sec: float | None


def load_pickle(path: Path):
    """Load a Pickle payload from ``path``."""

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


def extract_py_sampling_rate(payload: Mapping) -> float | None:
    params = payload.get("params", {})
    rate = params.get("resample_rate") if isinstance(params, Mapping) else None
    if rate is None and isinstance(payload.get("frames"), Mapping):
        frames = payload["frames"]
        params_frame = frames.get("params")
        if isinstance(params_frame, pd.DataFrame) and "resample_rate" in params_frame.columns:
            try:
                rate = float(pd.to_numeric(params_frame["resample_rate"], errors="coerce").iloc[0])
            except Exception:  # noqa: BLE001 - fall back to None
                rate = None
    if rate is None:
        return None
    try:
        return float(rate)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def extract_blinker_sampling_rate(payload: Mapping) -> float | None:
    frames = payload.get("frames")
    if isinstance(frames, Mapping):
        blinks = frames.get("blinks")
        if isinstance(blinks, pd.DataFrame) and "srate" in blinks.columns and not blinks.empty:
            try:
                rate = float(pd.to_numeric(blinks["srate"], errors="coerce").iloc[0])
                if not math.isnan(rate):
                    return rate
            except Exception:  # noqa: BLE001 - defensive
                return None
    return None


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
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    py_rate = extract_py_sampling_rate(py_payload)
    blinker_rate = extract_blinker_sampling_rate(blinker_payload)
    target_rate = blinker_rate or py_rate

    py_events_raw = py_payload["events"]
    py_events_raw = py_events_raw[["left_zero", "right_zero", "max_value"]].rename(
        columns={
                "left_zero": "start_blink",
                "right_zero": "end_blink",
                "max_value": "maxValue",
                }
        )
    py_events=py_events_raw.sort_values(by="start_blink").reset_index(drop=True)

    blinker_event=blinker_payload["frames"]["blinkFits"]
    blinker_events = blinker_event[["leftZero", "rightZero", "maxValue"]].rename(
        columns={"leftZero": "start_blink", "rightZero": "end_blink"}
        )
    blinker_events =blinker_events.sort_values(by="start_blink").reset_index(drop=True)
    return py_events, blinker_events, target_rate


def compute_onset_mae(
    alignments: Sequence,
    *,
    sampling_rate_hz: float | None,
    tolerance_samples: int,
) -> float | None:
    if sampling_rate_hz is None or not alignments:
        return None

    diffs = [
        abs(alignment.start_diff) / sampling_rate_hz
        for alignment in alignments
        if getattr(alignment, "start_diff", None) is not None
        and alignment.is_match(tolerance_samples)
    ]
    if not diffs:
        return None
    return float(sum(diffs) / len(diffs))


def iter_recordings(root: Path) -> Iterable[Path]:
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir():
            yield candidate


def build_summary_frame(comparisons: Sequence[RecordingComparison]) -> pd.DataFrame:
    rows: list[dict] = []
    for item in comparisons:
        metrics = item.metrics
        matched = metrics.get("matches_within_tolerance", 0.0)
        py_only = metrics.get("detected_only", 0.0)
        blinker_only = metrics.get("ground_truth_only", 0.0)

        precision = matched / (matched + py_only) if (matched + py_only) else math.nan
        recall = matched / (matched + blinker_only) if (matched + blinker_only) else math.nan
        if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
            f1 = math.nan
        else:
            f1 = 2 * precision * recall / (precision + recall)

        rows.append(
            {
                "recording_id": item.recording_id,
                "py_count": metrics.get("total_detected", float(len(item.py_events))),
                "blinker_count": metrics.get("total_ground_truth", float(len(item.blinker_events))),
                "matched": matched,
                "py_only": py_only,
                "blinker_only": blinker_only,
                "pairs_outside_tolerance": metrics.get("pairs_outside_tolerance", math.nan),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "onset_mae_sec": item.onset_mae_sec,
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("recording_id").reset_index(drop=True)
    return summary


def build_overall_summary(summary: pd.DataFrame) -> pd.Series:
    if summary.empty:
        return pd.Series(dtype=float)

    matched_total = summary["matched"].sum(skipna=True)
    py_only_total = summary["py_only"].sum(skipna=True)
    blinker_only_total = summary["blinker_only"].sum(skipna=True)

    precision_micro = matched_total / (matched_total + py_only_total) if (matched_total + py_only_total) else math.nan
    recall_micro = matched_total / (matched_total + blinker_only_total) if (matched_total + blinker_only_total) else math.nan

    if (
        math.isnan(precision_micro)
        or math.isnan(recall_micro)
        or (precision_micro + recall_micro) == 0
    ):
        f1_micro = math.nan
    else:
        f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)

    mae_values = summary["onset_mae_sec"].dropna()
    mae_mean = mae_values.mean() if not mae_values.empty else math.nan

    return pd.Series(
        {
            "recording_count": len(summary),
            "py_count_total": summary["py_count"].sum(skipna=True),
            "blinker_count_total": summary["blinker_count"].sum(skipna=True),
            "matched_total": matched_total,
            "py_only_total": py_only_total,
            "blinker_only_total": blinker_only_total,
            "pairs_outside_tolerance_total": summary[
                "pairs_outside_tolerance"
            ].sum(skipna=True),
            "precision_macro": summary["precision"].mean(skipna=True),
            "recall_macro": summary["recall"].mean(skipna=True),
            "f1_macro": summary["f1"].mean(skipna=True),
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "onset_mae_sec_mean": mae_mean,
        }
    )


def render_report(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    overall: pd.Series | None = None,
) -> Path:
    lines = ["# murat_2018 PyBlinker vs Blinker comparison", ""]
    lines.append("| Recording | PyBlinker | Blinker | Matched | Py-only | Blinker-only | Precision | Recall | F1 | MAE (s) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for _, row in summary.iterrows():
        def _fmt(value):
            if pd.isna(value):
                return "-"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(int(value))

        lines.append(
            "| {recording_id} | {py_count} | {blinker_count} | {matched} | {py_only} | {blinker_only} | {precision} | {recall} | {f1} | {mae} |".format(
                recording_id=row["recording_id"],
                py_count=_fmt(row["py_count"]),
                blinker_count=_fmt(row["blinker_count"]),
                matched=_fmt(row["matched"]),
                py_only=_fmt(row["py_only"]),
                blinker_only=_fmt(row["blinker_only"]),
                precision=_fmt(row["precision"]),
                recall=_fmt(row["recall"]),
                f1=_fmt(row["f1"]),
                mae=_fmt(row["onset_mae_sec"]),
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



def compare_recordings(
    root: Path,
    *,
    tolerance_samples: int,
    comparator,
) -> list[RecordingComparison]:
    """Compute recording comparisons using ``comparator`` for alignment metrics."""

    comparisons: list[RecordingComparison] = []

    for recording_dir in iter_recordings(root):
        py_path = recording_dir / "pyblinker_results.pkl"
        blinker_path = recording_dir / "blinker_results.pkl"
        fif_fname = f"{recording_dir.name}.fif"

        if not py_path.exists() or not blinker_path.exists():
            LOGGER.debug(
                "Skipping %s because expected pickle files are missing", recording_dir.name,
            )
            continue

        py_payload = load_pickle(py_path)
        blinker_payload = load_pickle(blinker_path)

        channel = py_payload["metrics"]["channel"]
        raw = mne.io.read_raw_fif(recording_dir / fif_fname, preload=True)
        signal = raw.get_data(picks=[channel])[0]
        py_events, blinker_events, sample_rate = prepare_event_tables(
            py_payload,
            blinker_payload,
        )

        n_preview_rows = 10
        n_diff_rows = 20

        comparison = comparator.compare_detected_vs_ground_truth(
            py_events,
            blinker_events,
            sample_rate,
            tolerance_samples=tolerance_samples,
            n_preview_rows=n_preview_rows,
            n_diff_rows=n_diff_rows,
            detected_signal=signal,
        )

        comparisons.append(
            RecordingComparison(
                recording_id=recording_dir.name,
                py_events=py_events,
                blinker_events=blinker_events,
                metrics=comparison.metrics,
                onset_mae_sec=compute_onset_mae(
                    comparison.alignments,
                    sampling_rate_hz=sample_rate,
                    tolerance_samples=tolerance_samples,
                ),
            )
        )

        to_plot = False
        if to_plot:
            raw.set_annotations(comparison.annotations)
            raw.plot(block=True)

    return comparisons

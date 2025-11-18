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
    """Compute summary metrics describing alignment quality.

    Metrics are derived directly from ``diff_table`` (typically produced by
    :func:`pyblinker.utils.evaluation.reporting.make_diff_table`) and include:

    ``total_ground_truth``
        Number of ground truth events.
    ``total_detected``
        Number of detected events.
    ``ground_truth_only``
        Ground truth events without a detected counterpart.
    ``detected_only``
        Detected events without a ground truth counterpart.
    ``share_within_tolerance``
        Count of unique events (detected plus ground truth) participating in
        amplitude- and overlap-satisfying pairs.
    ``matches_within_tolerance``
        Count of unique events belonging to pairs that met the tolerance window
        but failed at least one amplitude/overlap requirement.
    ``pairs_outside_tolerance``
        Count of unique events in pairs whose boundaries exceeded the tolerance
        window regardless of amplitude/overlap success.
    ``share_within_tolerance_percent``
        Percentage of unique events that participate in amplitude- and overlap-
        satisfying pairs.

    ``diff_table`` must include ``match_category`` and ``within_tolerance``
    columns describing how each paired event was classified:

    ``"share_within_tolerance"``
        Assigned when both amplitude and overlap conditions are satisfied for a
        detected/ground-truth pair. ``within_tolerance`` may be either ``True``
        or ``False`` depending on whether the start/end boundaries also fall
        within the tolerance window.
    ``"matches_within_tolerance"``
        Assigned to pairs whose start and end indices fall within the tolerance
        window but fail at least one amplitude/overlap requirement. These pairs
        are within the boundary window (``within_tolerance=True``) yet are not
        counted toward ``share_within_tolerance`` because the quality checks did
        not pass.
    ``"pairs_outside_tolerance"``
        Assigned when a paired event violates the tolerance window
        (``within_tolerance=False``), even if amplitude/overlap conditions are
        otherwise satisfied.

    The ``within_tolerance`` column is a boolean flag indicating whether both
    start and end differences for a paired event fall within the configured
    tolerance. Rows without a detected/ground-truth pairing use ``NaN`` for both
    ``match_category`` and ``within_tolerance``.

    Example
    -------
    Imagine ``tolerance_samples`` is ``1`` with three ground truth blinks
    (``G1``-``G3``) and three detected blinks (``D1``-``D3``). Suppose ``G1``
    and ``D1`` overlap and have similar amplitudes, so they receive
    ``match_category="share_within_tolerance"`` with ``within_tolerance=True``.
    ``G2`` and ``D2`` overlap but the detected start is two samples early, so
    they receive ``match_category="pairs_outside_tolerance"`` with
    ``within_tolerance=False`` even though amplitude checks pass. ``G3`` and
    ``D3`` align within the tolerance window but the amplitudes differ, leading
    to ``match_category="matches_within_tolerance"`` with
    ``within_tolerance=True``. The resulting metrics would be:

    * ``total_ground_truth`` = 3 and ``total_detected`` = 3.
    * ``ground_truth_only`` = 0 and ``detected_only`` = 0 because all events are
      paired.
    * ``share_within_tolerance`` = 2 because only ``G1`` and ``D1`` satisfy both
      amplitude and overlap checks; ``share_within_tolerance_percent`` is
      therefore ``2 / 6 * 100`` when measured against the six unique events.
    * ``matches_within_tolerance`` = 2 for the ``G3``/``D3`` pair whose
      amplitudes differ despite boundary agreement.
    * ``pairs_outside_tolerance`` = 2 for the ``G2``/``D2`` pair whose boundaries
      violate the tolerance window.
    """

    def _core_metrics(tp: float, fp: float, fn: float) -> tuple[float, float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) else math.nan
        recall = tp / (tp + fn) if (tp + fn) else math.nan
        if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
            f1 = math.nan
        else:
            f1 = 2 * precision * recall / (precision + recall)

        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else math.nan
        return precision, recall, f1, accuracy

    rows: list[dict] = []
    for item in comparisons:
        metrics = item.metrics

        total_ground_truth = metrics.get("total_ground_truth", float(len(item.blinker_events)))
        total_detected = metrics.get("total_detected", float(len(item.py_events)))
        ground_truth_only = metrics.get("ground_truth_only", 0.0)
        detected_only = metrics.get("detected_only", 0.0)
        share_within_tolerance = metrics.get("share_within_tolerance", 0.0)
        matches_within_tolerance = metrics.get("matches_within_tolerance", 0.0)
        pairs_outside_tolerance = metrics.get("pairs_outside_tolerance", 0.0)
        unique_total = metrics.get("unique_total")
        if unique_total is None:
            unique_total = total_ground_truth + total_detected

        share_within_tolerance_percent = (
            share_within_tolerance / unique_total * 100 if unique_total else math.nan
        )

        tp_strict = share_within_tolerance
        tp_lenient = share_within_tolerance + matches_within_tolerance
        fp = detected_only
        fn = ground_truth_only

        precision_strict, recall_strict, f1_strict, accuracy_strict = _core_metrics(
            tp_strict, fp, fn
        )
        precision_lenient, recall_lenient, f1_lenient, accuracy_lenient = _core_metrics(
            tp_lenient, fp, fn
        )

        rows.append(
            {
                "recording_id": item.recording_id,
                 "unique_total":unique_total,
                "total_detected": total_detected,
                "total_ground_truth": total_ground_truth,
                "ground_truth_only": ground_truth_only,
                "detected_only": detected_only,
                "share_within_tolerance": share_within_tolerance,
                "matches_within_tolerance": matches_within_tolerance,
                "pairs_outside_tolerance": pairs_outside_tolerance,
                "share_within_tolerance_percent": share_within_tolerance_percent,
                "tp_strict": tp_strict,
                "tp_lenient": tp_lenient,
                "fp": fp,
                "fn": fn,
                "precision_strict": precision_strict,
                "recall_strict": recall_strict,
                "f1_strict": f1_strict,
                "accuracy_strict": accuracy_strict,
                "precision_lenient": precision_lenient,
                "recall_lenient": recall_lenient,
                "f1_lenient": f1_lenient,
                "accuracy_lenient": accuracy_lenient,
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

    tp_strict_total = summary["tp_strict"].sum(skipna=True)
    tp_lenient_total = summary["tp_lenient"].sum(skipna=True)
    fp_total = summary["fp"].sum(skipna=True)
    fn_total = summary["fn"].sum(skipna=True)

    def _macro(column: str) -> float:
        return summary[column].mean(skipna=True)

    def _micro(tp: float) -> tuple[float, float, float, float]:
        precision = tp / (tp + fp_total) if (tp + fp_total) else math.nan
        recall = tp / (tp + fn_total) if (tp + fn_total) else math.nan
        if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
            f1 = math.nan
        else:
            f1 = 2 * precision * recall / (precision + recall)
        accuracy = tp / (tp + fp_total + fn_total) if (tp + fp_total + fn_total) else math.nan
        return precision, recall, f1, accuracy

    precision_strict_micro, recall_strict_micro, f1_strict_micro, accuracy_strict_micro = _micro(
        tp_strict_total
    )
    (
        precision_lenient_micro,
        recall_lenient_micro,
        f1_lenient_micro,
        accuracy_lenient_micro,
    ) = _micro(tp_lenient_total)

    mae_values = summary["onset_mae_sec"].dropna()
    mae_mean = mae_values.mean() if not mae_values.empty else math.nan

    return pd.Series(
        {
            "recording_count": len(summary),
            "total_detected_total": summary["total_detected"].sum(skipna=True),
            "total_ground_truth_total": summary["total_ground_truth"].sum(skipna=True),
            "share_within_tolerance_total": summary["share_within_tolerance"].sum(
                skipna=True
            ),
            "matches_within_tolerance_total": summary["matches_within_tolerance"].sum(
                skipna=True
            ),
            "pairs_outside_tolerance_total": summary[
                "pairs_outside_tolerance"
            ].sum(skipna=True),
            "detected_only_total": summary["detected_only"].sum(skipna=True),
            "ground_truth_only_total": summary["ground_truth_only"].sum(skipna=True),
            "tp_strict_total": tp_strict_total,
            "tp_lenient_total": tp_lenient_total,
            "fp_total": fp_total,
            "fn_total": fn_total,
            "precision_strict_macro": _macro("precision_strict"),
            "recall_strict_macro": _macro("recall_strict"),
            "f1_strict_macro": _macro("f1_strict"),
            "accuracy_strict_macro": _macro("accuracy_strict"),
            "precision_lenient_macro": _macro("precision_lenient"),
            "recall_lenient_macro": _macro("recall_lenient"),
            "f1_lenient_macro": _macro("f1_lenient"),
            "accuracy_lenient_macro": _macro("accuracy_lenient"),
            "precision_strict_micro": precision_strict_micro,
            "recall_strict_micro": recall_strict_micro,
            "f1_strict_micro": f1_strict_micro,
            "accuracy_strict_micro": accuracy_strict_micro,
            "precision_lenient_micro": precision_lenient_micro,
            "recall_lenient_micro": recall_lenient_micro,
            "f1_lenient_micro": f1_lenient_micro,
            "accuracy_lenient_micro": accuracy_lenient_micro,
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
    lines.append(
        "| Recording | Detected | Ground truth | share_within_tolerance | matches_within_tolerance | Py-only | Blinker-only | Precision (strict) | Recall (strict) | F1 (strict) | Precision (lenient) | Recall (lenient) | F1 (lenient) | MAE (s) |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    for _, row in summary.iterrows():
        def _fmt(value):
            if pd.isna(value):
                return "-"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(int(value))

        lines.append(
            "| {recording_id} | {py_count} | {blinker_count} | {share} | {matches} | {py_only} | {blinker_only} | {precision_strict} | {recall_strict} | {f1_strict} | {precision_lenient} | {recall_lenient} | {f1_lenient} | {mae} |".format(
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

        if comparison.annotations is not None:
            annotations_path = (recording_dir / fif_fname).with_suffix(".csv")
            LOGGER.info(
                "Saving comparison annotations to %s", annotations_path,
            )
            annotations_frame = pd.DataFrame(
                {
                    "onset": comparison.annotations.onset,
                    "duration": comparison.annotations.duration,
                    "description": comparison.annotations.description,
                }
            )
            annotations_frame.to_csv(annotations_path, index=False)

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

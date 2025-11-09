"""Compare pyblinker and blinker outputs and generate a summary report."""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency during tests
    from pyblinker.utils.evaluation import blink_comparison as _blink_comparison
except ModuleNotFoundError:  # pragma: no cover - pyblinker not installed for tests
    _blink_comparison = None
else:
    from pyblinker.utils.evaluation import blink_comparison  # noqa: F401

try:  # pragma: no cover - optional helpers
    from pyblinker.utils.evaluation import blink_detection, mat_data  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - pyblinker extras missing
    blink_detection = None  # type: ignore[assignment]
    mat_data = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_RAW = os.environ.get("MURAT_DATASET_ROOT")
DEFAULT_ROOT = Path(DEFAULT_ROOT_RAW) if DEFAULT_ROOT_RAW else REPO_ROOT / "data" / "murat_2018"
TOLERANCE_SAMPLES = 20
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RecordingComparison:
    recording_id: str
    py_events: pd.DataFrame
    blinker_events: pd.DataFrame
    metrics: dict[str, float]
    onset_mae_sec: float | None


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _extract_events(payload: dict, *, fallback_key: str) -> pd.DataFrame:
    if "events" in payload:
        frame = payload["events"]
        if isinstance(frame, pd.DataFrame):
            return frame
        return pd.DataFrame(frame)

    frames = payload.get("frames", {})
    if isinstance(frames, dict) and fallback_key in frames:
        candidate = frames[fallback_key]
        if isinstance(candidate, pd.DataFrame):
            return candidate
        return pd.DataFrame(candidate)
    return pd.DataFrame()


def _extract_py_sampling_rate(payload: dict) -> float | None:
    params = payload.get("params", {})
    rate = params.get("resample_rate") if isinstance(params, dict) else None
    if rate is None and isinstance(payload.get("frames"), dict):
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


def _extract_blinker_sampling_rate(payload: dict) -> float | None:
    frames = payload.get("frames")
    if isinstance(frames, dict):
        blinks = frames.get("blinks")
        if isinstance(blinks, pd.DataFrame) and "srate" in blinks.columns and not blinks.empty:
            try:
                rate = float(pd.to_numeric(blinks["srate"], errors="coerce").iloc[0])
                if not math.isnan(rate):
                    return rate
            except Exception:  # noqa: BLE001 - defensive
                return None
    return None


def _to_samples(
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


def _normalise_events(
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
        start_vals, end_vals = _to_samples(
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

        start_vals, end_vals = _to_samples(
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


def _prepare_event_tables(
    py_payload: dict,
    blinker_payload: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, float | None]:
    py_rate = _extract_py_sampling_rate(py_payload)
    blinker_rate = _extract_blinker_sampling_rate(blinker_payload)
    target_rate = blinker_rate or py_rate

    py_events_raw = _extract_events(py_payload, fallback_key="events")
    blinker_events_raw = _extract_events(blinker_payload, fallback_key="blinkFits")

    py_events = _normalise_events(py_events_raw, sample_rate=py_rate, target_rate=target_rate)
    blinker_events = _normalise_events(
        blinker_events_raw,
        sample_rate=blinker_rate,
        target_rate=target_rate,
    )

    if target_rate is None and not py_events.empty:
        target_rate = float(py_events.attrs.get("sampling_rate_hz", 0) or 0) or None

    if target_rate is None and not blinker_events.empty:
        target_rate = float(blinker_events.attrs.get("sampling_rate_hz", 0) or 0) or None

    return py_events, blinker_events, target_rate


def _compute_onset_mae(
    alignments: Sequence,
    *,
    sampling_rate_hz: float | None,
    tolerance_samples: int,
) -> float | None:
    r"""Return the mean absolute onset timing error for matched blink events.

    The alignments produced by ``blink_comparison.compute_alignments_and_metrics``
    capture the difference between each detected blink onset and its closest
    ground-truth counterpart in sample units.  The *mean absolute error (MAE)* is
    the arithmetic mean of the absolute onset differences expressed in seconds.

    MAE is a widely used regression metric defined as

    .. math:: \text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|,

    where ``y_i`` is the reference value (blinker onset) and ``\hat{y}_i`` is the
    predicted value (pyblinker onset).  In this context it quantifies the
    average temporal deviation between the two pipelines: lower values indicate
    that detections are, on average, closer to the reference events, while higher
    values point to greater onset disagreement.  Only alignments that qualify as
    a match within ``tolerance_samples`` contribute to the MAE; mismatched pairs
    are excluded to avoid inflating the error with unrelated events.

    Parameters
    ----------
    alignments:
        Iterable of alignment objects returned by the comparison helpers.
    sampling_rate_hz:
        Sampling frequency used to convert sample offsets into seconds.  If the
        sampling rate is unknown the MAE cannot be computed and ``None`` is
        returned.
    tolerance_samples:
        Maximum onset difference (in samples) for an alignment to be treated as
        a valid match.

    Returns
    -------
    float | None
        The mean absolute onset error in seconds for matched pairs, or ``None``
        when the sampling rate is missing or there are no qualifying alignments.
    """
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


def _iter_recordings(root: Path) -> Iterable[Path]:
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir():
            yield candidate


def compare_recordings(root: Path, tolerance_samples: int = TOLERANCE_SAMPLES) -> list[RecordingComparison]:
    comparisons: list[RecordingComparison] = []

    for recording_dir in _iter_recordings(root):
        py_path = recording_dir / "pyblinker_results.pkl"
        blinker_path = recording_dir / "blinker_results.pkl"
        if not py_path.exists() or not blinker_path.exists():
            LOGGER.warning("Skipping %s – missing outputs", recording_dir.name)
            continue

        try:
            py_payload = _load_pickle(py_path)
            blinker_payload = _load_pickle(blinker_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load outputs for %s: %s", recording_dir, exc)
            continue

        py_events, blinker_events, sample_rate = _prepare_event_tables(
            py_payload,
            blinker_payload,
        )

        if py_events.empty or blinker_events.empty:
            LOGGER.warning("Skipping %s – insufficient event data for comparison", recording_dir.name)
            continue

        if _blink_comparison is None:
            LOGGER.error(
                "pyblinker.utils.evaluation.blink_comparison is unavailable; install pyblinker to run comparisons"
            )
            continue

        try:
            alignments, metrics = _blink_comparison.compute_alignments_and_metrics(
                detected_df=py_events,
                ground_truth_df=blinker_events,
                tolerance_samples=tolerance_samples,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to compute comparison metrics for %s: %s", recording_dir.name, exc)
            continue

        onset_mae_sec = _compute_onset_mae(
            alignments,
            sampling_rate_hz=sample_rate,
            tolerance_samples=tolerance_samples,
        )

        comparisons.append(
            RecordingComparison(
                recording_id=recording_dir.name,
                py_events=py_events,
                blinker_events=blinker_events,
                metrics=metrics,
                onset_mae_sec=onset_mae_sec,
            )
        )

    return comparisons


def _build_summary_frame(comparisons: Sequence[RecordingComparison]) -> pd.DataFrame:
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
                "pairs_outside_tolerance": metrics.get("pairs_outside_tolerance", 0.0),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "onset_mae_sec": item.onset_mae_sec,
            }
        )
    summary = pd.DataFrame(rows)
    numeric_cols = [
        "py_count",
        "blinker_count",
        "matched",
        "py_only",
        "blinker_only",
        "pairs_outside_tolerance",
        "precision",
        "recall",
        "f1",
        "onset_mae_sec",
    ]
    for col in numeric_cols:
        if col in summary:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return summary


def _build_overall_summary(summary: pd.DataFrame) -> pd.Series:
    """Aggregate per-recording metrics into a global ""ultimate" summary."""
    if summary.empty:
        return pd.Series(dtype=float)

    matched_total = summary["matched"].sum(skipna=True)
    py_only_total = summary["py_only"].sum(skipna=True)
    blinker_only_total = summary["blinker_only"].sum(skipna=True)

    precision_micro = (
        matched_total / (matched_total + py_only_total)
        if (matched_total + py_only_total) > 0
        else math.nan
    )
    recall_micro = (
        matched_total / (matched_total + blinker_only_total)
        if (matched_total + blinker_only_total) > 0
        else math.nan
    )
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


def _render_report(
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=TOLERANCE_SAMPLES,
        help="Blink start/end alignment tolerance in samples.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = args.root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = compare_recordings(args.root, tolerance_samples=args.tolerance_samples)
    if not comparisons:
        LOGGER.warning("No recordings had both pyblinker and blinker outputs")
        return 0

    summary = _build_summary_frame(comparisons)
    summary_path = output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)

    overall = _build_overall_summary(summary)
    overall_path = None
    if not overall.empty:
        overall_path = output_dir / "summary_metrics_overall.csv"
        overall_frame = (
            overall.to_frame(name="value")
            .reset_index()
            .rename(columns={"index": "metric"})
        )
        overall_frame.to_csv(overall_path, index=False)

    report_path = _render_report(summary, output_dir, overall=overall)

    LOGGER.info("Summary metrics saved to %s", summary_path)
    if overall_path:
        LOGGER.info("Overall summary saved to %s", overall_path)
    LOGGER.info("Summary report saved to %s", report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

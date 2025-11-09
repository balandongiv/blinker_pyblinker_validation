"""Compare pyblinker and blinker outputs and generate a summary report."""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


DEFAULT_ROOT = Path(os.environ.get("MURAT_DATASET_ROOT", "D:/dataset/murat_2018"))
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RecordingComparison:
    recording_id: str
    py_events: pd.DataFrame
    blinker_events: pd.DataFrame
    matched: int
    false_py: int
    false_blinker: int
    onset_mae: float | None


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


def _ensure_onset_column(frame: pd.DataFrame, *, sfreq: float | None = None) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    columns = {col.lower(): col for col in frame.columns}
    for key in ("onset_sec", "latency_sec", "time", "latency", "start", "blink_start"):
        if key in columns:
            series = pd.to_numeric(frame[columns[key]], errors="coerce")
            if key == "latency" and sfreq is not None and series.max() > 1e3:
                return series / sfreq
            return series
    if "sample" in columns and sfreq is not None:
        return pd.to_numeric(frame[columns["sample"]], errors="coerce") / sfreq
    return pd.Series(dtype=float)


def _match_events(
    ref_times: Sequence[float],
    other_times: Sequence[float],
    tolerance_s: float,
) -> Tuple[int, int, int, List[float]]:
    ref_sorted = sorted((t, idx) for idx, t in enumerate(ref_times) if not math.isnan(t))
    other_sorted = sorted((t, idx) for idx, t in enumerate(other_times) if not math.isnan(t))

    matched_ref = set()
    matched_other = set()
    diffs: List[float] = []

    j = 0
    for ref_time, ref_idx in ref_sorted:
        best_j = None
        best_delta = None
        while j < len(other_sorted) and other_sorted[j][0] < ref_time - tolerance_s:
            j += 1
        k = j
        while k < len(other_sorted) and other_sorted[k][0] <= ref_time + tolerance_s:
            other_time, other_idx = other_sorted[k]
            delta = abs(other_time - ref_time)
            if best_delta is None or delta < best_delta:
                if other_idx not in matched_other:
                    best_delta = delta
                    best_j = other_idx
            k += 1

        if best_j is not None:
            matched_ref.add(ref_idx)
            matched_other.add(best_j)
            diffs.append(best_delta if best_delta is not None else 0.0)

    tp = len(matched_ref)
    fp_ref = len(ref_times) - tp
    fp_other = len(other_times) - len(matched_other)
    return tp, fp_ref, fp_other, diffs


def _compute_metrics(
    py_events: pd.DataFrame,
    blinker_events: pd.DataFrame,
    tolerance_s: float,
) -> RecordingComparison:
    py_onset = _ensure_onset_column(py_events)
    blinker_onset = _ensure_onset_column(blinker_events)

    tp, fp_py, fp_blinker, diffs = _match_events(py_onset, blinker_onset, tolerance_s)
    onset_mae = float(sum(diffs) / len(diffs)) if diffs else None

    return RecordingComparison(
        recording_id="",
        py_events=py_events,
        blinker_events=blinker_events,
        matched=tp,
        false_py=fp_py,
        false_blinker=fp_blinker,
        onset_mae=onset_mae,
    )


def _iter_recordings(root: Path) -> Iterable[Path]:
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir():
            yield candidate


def compare_recordings(root: Path, tolerance_ms: float) -> list[RecordingComparison]:
    tolerance_s = tolerance_ms / 1000.0
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

        py_events = _extract_events(py_payload, fallback_key="events")
        blinker_events = _extract_events(blinker_payload, fallback_key="blinkFits")

        metrics = _compute_metrics(py_events, blinker_events, tolerance_s)
        metrics.recording_id = recording_dir.name
        comparisons.append(metrics)

    return comparisons


def _build_summary_frame(comparisons: Sequence[RecordingComparison]) -> pd.DataFrame:
    rows: list[dict] = []
    for item in comparisons:
        precision = item.matched / (item.matched + item.false_py) if (item.matched + item.false_py) else None
        recall = item.matched / (item.matched + item.false_blinker) if (item.matched + item.false_blinker) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall)
            else None
        )
        rows.append(
            {
                "recording_id": item.recording_id,
                "py_count": len(item.py_events),
                "blinker_count": len(item.blinker_events),
                "matched": item.matched,
                "py_only": item.false_py,
                "blinker_only": item.false_blinker,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "onset_mae_sec": item.onset_mae,
            }
        )
    return pd.DataFrame(rows)


def _render_report(summary: pd.DataFrame, output_dir: Path) -> Path:
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
        "--tolerance-ms",
        type=float,
        default=150.0,
        help="Matching window in milliseconds.",
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

    comparisons = compare_recordings(args.root, tolerance_ms=args.tolerance_ms)
    if not comparisons:
        LOGGER.warning("No recordings had both pyblinker and blinker outputs")
        return 0

    summary = _build_summary_frame(comparisons)
    summary_path = output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    report_path = _render_report(summary, output_dir)

    LOGGER.info("Summary metrics saved to %s", summary_path)
    LOGGER.info("Summary report saved to %s", report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

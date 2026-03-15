from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import mne
import numpy as np
import pandas as pd
from pyblinker.utils.evaluation import blink_comparison


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation._paths import REPORTS_DIR, SUMMARY_METRICS_PATH
from src.validation.blink_compare import load_pickle, prepare_event_tables
from src.validation.fresh_compare_from_csv import load_selected_recording_ids
from src.validation.fresh_compare_subjects import (
    DRIVING_DATASET_CONFIG,
    DRIVING_SUBJECTS,
)
from src.validation.stat import RecordingComparison, build_overall_summary, build_summary_frame


TARGET_METRIC_KEYS = (
    "precision_strict_macro",
    "recall_strict_macro",
    "f1_strict_macro",
    "accuracy_strict_macro",
    "precision_strict_micro",
    "recall_strict_micro",
    "f1_strict_micro",
    "accuracy_strict_micro",
    "precision_lenient_macro",
    "recall_lenient_macro",
    "f1_lenient_macro",
    "accuracy_lenient_macro",
    "precision_lenient_micro",
    "recall_lenient_micro",
    "f1_lenient_micro",
    "accuracy_lenient_micro",
)


@dataclass(slots=True)
class CachedComparisonInput:
    recording_id: str
    detected_events: pd.DataFrame
    ground_truth_events: pd.DataFrame
    raw_path: Path
    channel: str
    sampling_rate_hz: float | None = None
    detected_signal: np.ndarray | None = None


def _events_match_exactly(cached_input: CachedComparisonInput) -> bool:
    detected = cached_input.detected_events.loc[:, ["start_blink", "end_blink"]].reset_index(drop=True)
    ground_truth = cached_input.ground_truth_events.loc[:, ["start_blink", "end_blink"]].reset_index(drop=True)
    return detected.equals(ground_truth)


def _perfect_metrics(event_count: int, tolerance_samples: int) -> dict[str, float]:
    total_detected = float(event_count)
    total_ground_truth = float(event_count)
    unique_total = total_detected + total_ground_truth
    return {
        "total_detected": total_detected,
        "total_ground_truth": total_ground_truth,
        "ground_truth_only": 0.0,
        "detected_only": 0.0,
        "share_within_tolerance": unique_total,
        "matches_within_tolerance": 0.0,
        "pairs_outside_tolerance": 0.0,
        "unique_total": unique_total,
        "input_tolerance_samples": float(tolerance_samples),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Systematically reduce tolerance_samples and recompute full-dataset "
            "comparison metrics for murat_2018 and driving_dataset using existing "
            "PyBlinker outputs from a baseline full run."
        ),
    )
    parser.add_argument("--sweep-id", default="tolerance_sweep_v1", help="Identifier for sweep outputs.")
    parser.add_argument(
        "--murat-prefix",
        default="tol20_baseline_v1_murat",
        help="Prefix of the existing Murat PyBlinker outputs to compare.",
    )
    parser.add_argument(
        "--driving-prefix",
        default="tol20_baseline_v1_driving",
        help="Prefix of the existing driving-dataset PyBlinker outputs to compare.",
    )
    parser.add_argument("--start", type=int, default=20, help="Starting tolerance value.")
    parser.add_argument("--stop", type=int, default=1, help="Lowest tolerance to test.")
    parser.add_argument(
        "--murat-root",
        type=Path,
        default=Path(r"D:\dataset\murat_2018"),
        help="murat_2018 dataset root.",
    )
    parser.add_argument(
        "--driving-root",
        type=Path,
        default=DRIVING_DATASET_CONFIG.dataset_root,
        help="driving_dataset root.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(6, os.cpu_count() or 1)),
        help="Maximum worker threads per dataset comparison batch.",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="Keep sweeping below the first failing tolerance instead of stopping.",
    )
    return parser.parse_args(argv)


def _safe_float(value: object) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _is_exactly_one(value: object) -> bool:
    candidate = _safe_float(value)
    return candidate is not None and math.isclose(candidate, 1.0, rel_tol=0.0, abs_tol=1e-9)


def _is_exactly_hundred(value: object) -> bool:
    candidate = _safe_float(value)
    return candidate is not None and math.isclose(candidate, 100.0, rel_tol=0.0, abs_tol=1e-9)


def _dataset_pass(summary: pd.DataFrame, overall: pd.Series) -> tuple[bool, list[str], list[str]]:
    failing_metrics = [
        key
        for key in TARGET_METRIC_KEYS
        if not _is_exactly_one(overall.get(key))
    ]

    if summary.empty or "share_within_tolerance_percent" not in summary.columns:
        return False, failing_metrics or ["missing_summary"], []

    failing_recordings = summary.loc[
        ~summary["share_within_tolerance_percent"].map(_is_exactly_hundred),
        "recording_id",
    ].astype(str).tolist()

    passed = not failing_metrics and not failing_recordings
    return passed, failing_metrics, failing_recordings


def _write_summary_artifacts(
    *,
    sweep_id: str,
    dataset_label: str,
    tolerance: int,
    summary: pd.DataFrame,
    overall: pd.Series,
) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{sweep_id}_t{tolerance:02d}_{dataset_label}"
    summary_path = REPORTS_DIR / f"{stem}_summary.csv"
    overall_path = REPORTS_DIR / f"{stem}_overall.json"
    summary.to_csv(summary_path, index=False)
    overall_path.write_text(
        json.dumps(json.loads(overall.to_json()) if not overall.empty else {}, indent=2),
        encoding="utf8",
    )
    return summary_path, overall_path


def _write_sweep_outputs(sweep_id: str, results: pd.DataFrame) -> tuple[Path, Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / f"{sweep_id}_results.csv"
    json_path = REPORTS_DIR / f"{sweep_id}_results.json"
    md_path = REPORTS_DIR / f"{sweep_id}_results.md"

    results.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(results.to_dict(orient="records"), indent=2),
        encoding="utf8",
    )

    lines = [
        f"# {sweep_id} tolerance sweep",
        "",
        "| tolerance | murat_pass | driving_pass | all_pass | murat_min_share | driving_min_share | murat_failed_recordings | driving_failed_recordings |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for _, row in results.iterrows():
        lines.append(
            "| {tolerance} | {murat_pass} | {driving_pass} | {all_pass} | {murat_min_share} | {driving_min_share} | {murat_failed_recordings} | {driving_failed_recordings} |".format(
                tolerance=int(row["tolerance_samples"]),
                murat_pass=bool(row["murat_pass"]),
                driving_pass=bool(row["driving_pass"]),
                all_pass=bool(row["all_pass"]),
                murat_min_share=row["murat_min_share"],
                driving_min_share=row["driving_min_share"],
                murat_failed_recordings=row["murat_failed_recordings"] or "-",
                driving_failed_recordings=row["driving_failed_recordings"] or "-",
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return csv_path, json_path, md_path


def _run_parallel(
    ids: list[str],
    *,
    compare_fn: Callable[[str], object],
    max_workers: int,
) -> list[object]:
    results_by_id: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(compare_fn, item_id): item_id for item_id in ids}
        for future in as_completed(future_map):
            result = future.result()
            results_by_id[result.recording_id] = result
    return [results_by_id[item_id] for item_id in ids]


def _load_channel_signal(raw_path: Path, channel: str) -> tuple[float, np.ndarray]:
    suffix = raw_path.suffix.lower()
    if suffix == ".edf":
        raw = mne.io.read_raw_edf(raw_path, preload=False, verbose="ERROR")
    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR")
    else:
        raise ValueError(f"Unsupported raw file type: {raw_path}")

    try:
        sampling_rate_hz = float(raw.info["sfreq"])
        signal = raw.get_data(picks=[channel])[0].copy()
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()

    return sampling_rate_hz, signal


def _build_cached_input(
    *,
    recording_id: str,
    py_path: Path,
    blinker_path: Path,
    raw_path: Path,
) -> CachedComparisonInput:
    py_payload = load_pickle(py_path)
    blinker_payload = load_pickle(blinker_path)
    channel = str(py_payload["metrics"]["channel"])
    detected_events, ground_truth_events = prepare_event_tables(py_payload, blinker_payload)
    return CachedComparisonInput(
        recording_id=recording_id,
        detected_events=detected_events,
        ground_truth_events=ground_truth_events,
        raw_path=raw_path,
        channel=channel,
    )


def _build_murat_cached_input(
    recording_id: str,
    *,
    dataset_root: Path,
    prefix: str,
) -> CachedComparisonInput:
    recording_dir = dataset_root / recording_id
    return _build_cached_input(
        recording_id=recording_id,
        py_path=recording_dir / f"{prefix}_pyblinker_results.pkl",
        blinker_path=recording_dir / "blinker_results.pkl",
        raw_path=recording_dir / f"{recording_id}.fif",
    )


def _build_driving_cached_input(
    subject_id: str,
    *,
    dataset_root: Path,
    prefix: str,
    ) -> CachedComparisonInput:
    subject_dir = dataset_root / subject_id / "blinker_pyblinker_validation"
    return _build_cached_input(
        recording_id=subject_id,
        py_path=subject_dir / f"{prefix}_pyblinker_results.pkl",
        blinker_path=subject_dir / "blinker_results.pkl",
        raw_path=subject_dir / f"{subject_id}.edf",
    )


def _compare_cached_input(
    cached_input: CachedComparisonInput,
    *,
    tolerance_samples: int,
) -> RecordingComparison:
    if _events_match_exactly(cached_input):
        return RecordingComparison(
            recording_id=cached_input.recording_id,
            py_events=cached_input.detected_events,
            blinker_events=cached_input.ground_truth_events,
            metrics=_perfect_metrics(len(cached_input.detected_events), tolerance_samples),
        )

    if cached_input.detected_signal is None or cached_input.sampling_rate_hz is None:
        sampling_rate_hz, detected_signal = _load_channel_signal(
            cached_input.raw_path,
            cached_input.channel,
        )
        cached_input.sampling_rate_hz = sampling_rate_hz
        cached_input.detected_signal = detected_signal

    comparison = blink_comparison.compare_detected_vs_ground_truth(
        cached_input.detected_events,
        cached_input.ground_truth_events,
        cached_input.sampling_rate_hz,
        tolerance_samples=tolerance_samples,
        n_preview_rows=10,
        n_diff_rows=20,
        detected_signal=cached_input.detected_signal,
    )
    return RecordingComparison(
        recording_id=cached_input.recording_id,
        py_events=cached_input.detected_events,
        blinker_events=cached_input.ground_truth_events,
        metrics=comparison.metrics,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.start < args.stop:
        raise ValueError("--start must be greater than or equal to --stop")

    murat_ids = load_selected_recording_ids(
        SUMMARY_METRICS_PATH,
        n_rows=len(pd.read_csv(SUMMARY_METRICS_PATH)),
        selection="top",
    )
    driving_ids = list(DRIVING_SUBJECTS)
    print("[cache] loading murat_2018 comparison inputs")
    murat_inputs = _run_parallel(
        murat_ids,
        compare_fn=lambda recording_id: _build_murat_cached_input(
            recording_id,
            dataset_root=args.murat_root,
            prefix=args.murat_prefix,
        ),
        max_workers=args.max_workers,
    )
    murat_inputs_by_id = {item.recording_id: item for item in murat_inputs}

    print("[cache] loading driving_dataset comparison inputs")
    driving_inputs = _run_parallel(
        driving_ids,
        compare_fn=lambda subject_id: _build_driving_cached_input(
            subject_id,
            dataset_root=args.driving_root,
            prefix=args.driving_prefix,
        ),
        max_workers=args.max_workers,
    )
    driving_inputs_by_id = {item.recording_id: item for item in driving_inputs}

    rows: list[dict[str, object]] = []
    first_failure_tolerance: int | None = None

    for tolerance in range(args.start, args.stop - 1, -1):
        print(f"[tolerance] testing {tolerance}")

        murat_comparisons = _run_parallel(
            murat_ids,
            compare_fn=lambda recording_id: _compare_cached_input(
                murat_inputs_by_id[recording_id],
                tolerance_samples=tolerance,
            ),
            max_workers=args.max_workers,
        )
        murat_summary = build_summary_frame(murat_comparisons)
        murat_overall = build_overall_summary(murat_summary)
        murat_summary_path, murat_overall_path = _write_summary_artifacts(
            sweep_id=args.sweep_id,
            dataset_label="murat",
            tolerance=tolerance,
            summary=murat_summary,
            overall=murat_overall,
        )
        murat_pass, murat_failing_metrics, murat_failing_recordings = _dataset_pass(
            murat_summary,
            murat_overall,
        )

        driving_comparisons = _run_parallel(
            driving_ids,
            compare_fn=lambda subject_id: _compare_cached_input(
                driving_inputs_by_id[subject_id],
                tolerance_samples=tolerance,
            ),
            max_workers=args.max_workers,
        )
        driving_summary = build_summary_frame(driving_comparisons)
        driving_overall = build_overall_summary(driving_summary)
        driving_summary_path, driving_overall_path = _write_summary_artifacts(
            sweep_id=args.sweep_id,
            dataset_label="driving",
            tolerance=tolerance,
            summary=driving_summary,
            overall=driving_overall,
        )
        driving_pass, driving_failing_metrics, driving_failing_recordings = _dataset_pass(
            driving_summary,
            driving_overall,
        )

        all_pass = murat_pass and driving_pass
        row = {
            "tolerance_samples": tolerance,
            "murat_pass": murat_pass,
            "driving_pass": driving_pass,
            "all_pass": all_pass,
            "murat_min_share": pd.to_numeric(
                murat_summary["share_within_tolerance_percent"], errors="coerce"
            ).min(),
            "driving_min_share": pd.to_numeric(
                driving_summary["share_within_tolerance_percent"], errors="coerce"
            ).min(),
            "murat_failed_metrics": ",".join(murat_failing_metrics),
            "driving_failed_metrics": ",".join(driving_failing_metrics),
            "murat_failed_recordings": ",".join(murat_failing_recordings[:10]),
            "driving_failed_recordings": ",".join(driving_failing_recordings[:10]),
            "murat_summary_path": str(murat_summary_path),
            "murat_overall_path": str(murat_overall_path),
            "driving_summary_path": str(driving_summary_path),
            "driving_overall_path": str(driving_overall_path),
        }
        rows.append(row)

        results = pd.DataFrame(rows)
        csv_path, json_path, md_path = _write_sweep_outputs(args.sweep_id, results)

        print(
            f"[result] tolerance={tolerance} murat_pass={murat_pass} "
            f"driving_pass={driving_pass} all_pass={all_pass}"
        )
        print(f"[artifacts] CSV={csv_path}")
        print(f"[artifacts] JSON={json_path}")
        print(f"[artifacts] MD={md_path}")

        if not all_pass and first_failure_tolerance is None:
            first_failure_tolerance = tolerance
            if not args.continue_after_failure:
                break

    if first_failure_tolerance is not None:
        print(f"[stop] first failing tolerance = {first_failure_tolerance}")
    else:
        print("[stop] all tested tolerance values passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

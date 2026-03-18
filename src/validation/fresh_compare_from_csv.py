from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd
from pyblinker.blinker.pyblinker import BlinkDetector
from pyblinker.utils.annotation_utils import create_annotation


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation._paths import REPORTS_DIR
from src.validation.blinker_params import build_experiment_blink_params
from src.validation.blink_compare import load_pickle, process_recording_comparison
from src.validation.blink_compare_from_csv import CSV_PATH, DATASET_ROOT, TOLERANCE_SAMPLES
from src.validation.stat import RecordingComparison, build_overall_summary, build_summary_frame


RESULTS_DIR = REPORTS_DIR
DEFAULT_TARGET_SHARE_PERCENT = 100.0


@dataclass(slots=True)
class RecordingRunResult:
    recording_id: str
    comparison: RecordingComparison
    py_path: Path
    artifact_status: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fresh PyBlinker detections for selected murat_2018 recordings, "
            "save prefixed pickle outputs beside each subject, and compare them "
            "against MATLAB Blinker results."
        ),
    )
    parser.add_argument("--n", type=int, default=10, help="Use the selected N recording IDs.")
    parser.add_argument(
        "--selection",
        choices=("top", "bottom"),
        default="top",
        help="Select the first N rows or the last N rows from summary_metrics.csv.",
    )
    parser.add_argument("--prefix", default="exp01", help="Prefix for generated PyBlinker outputs.")
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH, help="Ordered summary_metrics.csv path.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT, help="murat_2018 root.")
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=TOLERANCE_SAMPLES,
        help="Tolerance window used for comparison metrics.",
    )
    parser.add_argument("--plot", action="store_true", help="Display EEG with blink annotations.")
    parser.add_argument(
        "--execution-mode",
        choices=("auto", "single", "thread"),
        default="auto",
        help="Use threaded processing automatically when more than 10 recordings are selected.",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="Maximum worker threads.")
    parser.add_argument(
        "--target-share-percent",
        type=float,
        default=DEFAULT_TARGET_SHARE_PERCENT,
        help="Required share_within_tolerance_percent for reusing an existing pickle.",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Regenerate outputs even if they exist.")
    return parser.parse_args(argv)


def _fresh_py_path(recording_dir: Path, prefix: str) -> Path:
    return recording_dir / f"{prefix}_pyblinker_results.pkl"


def load_selected_recording_ids(
    csv_path: Path,
    *,
    n_rows: int,
    selection: str,
) -> list[str]:
    summary = pd.read_csv(csv_path, dtype={"recording_id": "string"})
    if "recording_id" not in summary.columns:
        raise KeyError(f"Column 'recording_id' not found in {csv_path}")

    recording_series = summary["recording_id"].dropna().astype("string").str.strip()
    recording_series = recording_series[recording_series != ""]
    if n_rows < 1:
        return []

    selected = recording_series.tail(n_rows) if selection == "bottom" else recording_series.head(n_rows)
    return selected.tolist()


def _load_raw(recording_dir: Path, recording_id: str) -> mne.io.BaseRaw:
    edf_path = recording_dir / f"{recording_id}.edf"
    if edf_path.exists():
        return mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    fif_path = recording_dir / f"{recording_id}.fif"
    if fif_path.exists():
        return mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

    raise FileNotFoundError(
        f"No EDF or FIF input was found for recording {recording_id} in {recording_dir}"
    )


def _plot_with_annotations(raw: mne.io.BaseRaw, annotations: mne.Annotations, channel: str) -> None:
    raw_plot = raw.copy()
    raw_plot.set_annotations(annotations)
    print(f"Displaying EEG with blink detections (based on channel {channel})...")
    raw_plot.plot(block=True, title=f"Eye close based on channel {channel}")


def _plot_from_saved_payload(recording_id: str, recording_dir: Path, py_path: Path) -> None:
    payload = load_pickle(py_path)
    raw = _load_raw(recording_dir, recording_id)
    annotations = create_annotation(payload["events"].copy(), float(raw.info["sfreq"]), "eye_blink")
    _plot_with_annotations(raw, annotations, str(payload["metrics"]["channel"]))


def _run_detector(recording_id: str, recording_dir: Path) -> tuple[dict, mne.Annotations, str]:
    raw = _load_raw(recording_dir, recording_id)
    sampling_rate = float(raw.info["sfreq"])
    blink_params = build_experiment_blink_params()

    detector = BlinkDetector(
        raw.copy(),
        visualize=False,
        annot_label="eye_blink",
        filter_low=1.0,
        filter_high=20.0,
        resample_rate=int(round(sampling_rate)),
        n_jobs=1,
        use_multiprocessing=False,
        blink_params=blink_params,
    )
    annotations, channel, n_good, blink_details, _fig_data, selected = detector.get_blink()

    payload = {
        "events": blink_details.copy(),
        "metrics": {
            "channel": channel,
            "n_good_blinks": int(n_good),
            "sampling_rate_hz": float(detector.raw_data.info["sfreq"]),
        },
        "selected_channel": selected.copy(),
        "params": {
            "blink_params": blink_params,
        },
    }
    return payload, annotations, channel


def run_fresh_detection(
    recording_id: str,
    recording_dir: Path,
    *,
    prefix: str,
    plot: bool,
) -> Path:
    payload, annotations, channel = _run_detector(recording_id, recording_dir)

    output_path = _fresh_py_path(recording_dir, prefix)
    payload["metrics"]["result_file"] = output_path.name

    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)

    if plot:
        raw = _load_raw(recording_dir, recording_id)
        _plot_with_annotations(raw, annotations, channel)

    return output_path


def _run_comparison(
    recording_id: str,
    *,
    dataset_root: Path,
    py_path: Path,
    tolerance_samples: int,
) -> RecordingComparison:
    recording_dir = dataset_root / recording_id
    return process_recording_comparison(
        recording_dir,
        py_path,
        recording_dir / "blinker_results.pkl",
        recording_dir / f"{recording_id}.fif",
        recording_id,
        tolerance_samples=tolerance_samples,
        overwrite=True,
    )


def _share_is_good(value: float | None, target: float) -> bool:
    if value is None or not math.isfinite(float(value)):
        return False
    return math.isclose(float(value), float(target), rel_tol=0.0, abs_tol=1e-9)


def process_recording(
    recording_id: str,
    *,
    dataset_root: Path,
    prefix: str,
    tolerance_samples: int,
    plot: bool,
    target_share_percent: float,
    force_rerun: bool,
) -> RecordingRunResult:
    recording_dir = dataset_root / recording_id
    py_path = _fresh_py_path(recording_dir, prefix)

    if force_rerun and py_path.exists():
        py_path.unlink()

    if py_path.exists():
        try:
            comparison = _run_comparison(
                recording_id,
                dataset_root=dataset_root,
                py_path=py_path,
                tolerance_samples=tolerance_samples,
            )
            share = comparison.metrics.get("share_within_tolerance_percent")
            if _share_is_good(share, target_share_percent):
                if plot:
                    _plot_from_saved_payload(recording_id, recording_dir, py_path)
                return RecordingRunResult(
                    recording_id=recording_id,
                    comparison=comparison,
                    py_path=py_path,
                    artifact_status="reused_good",
                )
            py_path.unlink(missing_ok=True)
            artifact_status = "rerun_after_poor"
        except Exception:
            py_path.unlink(missing_ok=True)
            artifact_status = "rerun_after_error"
    else:
        artifact_status = "new"

    py_path = run_fresh_detection(recording_id, recording_dir, prefix=prefix, plot=plot)
    comparison = _run_comparison(
        recording_id,
        dataset_root=dataset_root,
        py_path=py_path,
        tolerance_samples=tolerance_samples,
    )
    share = comparison.metrics.get("share_within_tolerance_percent")
    if not _share_is_good(share, target_share_percent):
        artifact_status = f"{artifact_status}_still_poor"

    return RecordingRunResult(
        recording_id=recording_id,
        comparison=comparison,
        py_path=py_path,
        artifact_status=artifact_status,
    )


def _order_summary(summary: pd.DataFrame, recording_ids: list[str]) -> pd.DataFrame:
    if summary.empty:
        return summary

    order = {recording_id: idx for idx, recording_id in enumerate(recording_ids)}
    ordered = summary.copy()
    ordered["_order"] = ordered["recording_id"].map(order)
    ordered = ordered.sort_values("_order", kind="mergesort").drop(columns="_order")
    return ordered.reset_index(drop=True)


def _ensure_summary_frame(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty and "recording_id" not in summary.columns:
        return pd.DataFrame({"recording_id": pd.Series(dtype="string")})
    return summary


def _write_experiment_outputs(
    *,
    summary: pd.DataFrame,
    overall: pd.Series,
    prefix: str,
    n_subjects: int,
    selection: str,
    recording_ids: list[str],
) -> tuple[Path, Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_label = f"{selection}{n_subjects}"
    summary_path = RESULTS_DIR / f"{prefix}_{run_label}_summary.csv"
    overall_path = RESULTS_DIR / f"{prefix}_{run_label}_overall.json"
    selection_path = RESULTS_DIR / f"{prefix}_{run_label}_recordings.csv"

    summary.to_csv(summary_path, index=False)
    overall_dict = json.loads(overall.to_json()) if not overall.empty else {}
    overall_path.write_text(json.dumps(overall_dict, indent=2), encoding="utf8")
    pd.DataFrame(
        {
            "selection": selection,
            "recording_order": list(range(1, len(recording_ids) + 1)),
            "recording_id": recording_ids,
        }
    ).to_csv(selection_path, index=False)

    return summary_path, overall_path, selection_path


def _resolve_execution_mode(requested_mode: str, recording_count: int, *, plot: bool) -> str:
    if plot and recording_count > 1:
        return "single"
    if requested_mode == "auto":
        return "thread" if recording_count > 10 else "single"
    return requested_mode


def _resolve_max_workers(recording_count: int, explicit_max_workers: int | None) -> int:
    if explicit_max_workers is not None:
        return max(1, explicit_max_workers)
    return max(1, min(8, os.cpu_count() or 1, recording_count))


def _print_recording_result(result: RecordingRunResult) -> None:
    share = result.comparison.metrics.get("share_within_tolerance_percent")
    print(
        f"[fresh-run] {result.recording_id}: {result.artifact_status}, "
        f"{result.py_path.name}, share_within_tolerance_percent={share}"
    )


def _run_all_recordings(
    recording_ids: list[str],
    *,
    dataset_root: Path,
    prefix: str,
    tolerance_samples: int,
    plot: bool,
    execution_mode: str,
    max_workers: int,
    target_share_percent: float,
    force_rerun: bool,
) -> list[RecordingRunResult]:
    if execution_mode == "thread":
        results_by_id: dict[str, RecordingRunResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    process_recording,
                    recording_id,
                    dataset_root=dataset_root,
                    prefix=prefix,
                    tolerance_samples=tolerance_samples,
                    plot=False,
                    target_share_percent=target_share_percent,
                    force_rerun=force_rerun,
                ): recording_id
                for recording_id in recording_ids
            }
            for future in as_completed(future_map):
                result = future.result()
                results_by_id[result.recording_id] = result
                _print_recording_result(result)
        return [results_by_id[recording_id] for recording_id in recording_ids]

    results: list[RecordingRunResult] = []
    for recording_id in recording_ids:
        result = process_recording(
            recording_id,
            dataset_root=dataset_root,
            prefix=prefix,
            tolerance_samples=tolerance_samples,
            plot=plot,
            target_share_percent=target_share_percent,
            force_rerun=force_rerun,
        )
        results.append(result)
        _print_recording_result(result)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recording_ids = load_selected_recording_ids(args.csv_path, n_rows=args.n, selection=args.selection)
    execution_mode = _resolve_execution_mode(args.execution_mode, len(recording_ids), plot=args.plot)
    max_workers = _resolve_max_workers(len(recording_ids), args.max_workers)

    print(
        f"[config] selection={args.selection}, recordings={len(recording_ids)}, "
        f"execution_mode={execution_mode}, max_workers={max_workers}, "
        f"plot={args.plot}, force_rerun={args.force_rerun}"
    )

    results = _run_all_recordings(
        recording_ids,
        dataset_root=args.dataset_root,
        prefix=args.prefix,
        tolerance_samples=args.tolerance_samples,
        plot=args.plot,
        execution_mode=execution_mode,
        max_workers=max_workers,
        target_share_percent=args.target_share_percent,
        force_rerun=args.force_rerun,
    )

    summary = _ensure_summary_frame(
        _order_summary(build_summary_frame([result.comparison for result in results]), recording_ids)
    )
    summary["artifact_status"] = summary["recording_id"].map(
        {result.recording_id: result.artifact_status for result in results}
    )
    summary["result_file"] = f"{args.prefix}_pyblinker_results.pkl"
    summary["selection"] = args.selection

    overall = build_overall_summary(summary)
    summary_path, overall_path, selection_path = _write_experiment_outputs(
        summary=summary,
        overall=overall,
        prefix=args.prefix,
        n_subjects=len(recording_ids),
        selection=args.selection,
        recording_ids=recording_ids,
    )

    print()
    print(summary.to_csv(index=False))
    print(json.dumps(json.loads(overall.to_json()) if not overall.empty else {}, indent=2))
    print()
    print(f"[summary] CSV: {summary_path}")
    print(f"[summary] JSON: {overall_path}")
    print(f"[summary] selection CSV: {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

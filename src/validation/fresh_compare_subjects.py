from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import threading
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from pyblinker.blinker.pyblinker import BlinkDetector
from pyblinker.utils.annotation_utils import create_annotation


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation._paths import REPORTS_DIR
from src.validation.blinker_params import build_experiment_blink_params
from src.validation.blink_compare import compare_events, load_pickle
from src.validation.stat import RecordingComparison, build_overall_summary, build_summary_frame


RESULTS_DIR = REPORTS_DIR
DEFAULT_TARGET_SHARE_PERCENT = 100.0
DEFAULT_TOLERANCE_SAMPLES = 20

DRIVING_SUBJECTS = (
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S10",
    "S11",
    "S12",
    "S13",
    "S16",
    "S17",
    "S18",
    "S19",
    "S20",
    "S21",
    "S22",
    "S23",
    "S24",
    "S26",
    "S27",
)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str
    dataset_root: Path
    subject_ids: tuple[str, ...]
    py_raw_template: str
    comparison_raw_template: str
    blinker_results_template: str
    output_dir_template: str
    pick_types_options: dict[str, bool]


@dataclass(slots=True)
class SubjectRunResult:
    subject_id: str
    comparison: RecordingComparison
    py_path: Path
    artifact_status: str
    py_raw_path: Path
    comparison_raw_path: Path


DRIVING_DATASET_CONFIG = DatasetConfig(
    name="driving_dataset",
    dataset_root=Path("D:/dataset/drowsy_driving_raja_processed"),
    subject_ids=DRIVING_SUBJECTS,
    py_raw_template="{id}/{id}.fif",
    comparison_raw_template="{id}/blinker_pyblinker_validation/{id}.edf",
    blinker_results_template="{id}/blinker_pyblinker_validation/blinker_results.pkl",
    output_dir_template="{id}/blinker_pyblinker_validation",
    pick_types_options={"eeg": True, "eog": True},
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fresh PyBlinker detections for an explicit ordered subject list, "
            "save prefixed pickle outputs, and compare them against MATLAB Blinker results."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("driving_dataset", "custom"),
        default="driving_dataset",
        help="Dataset preset to use. 'custom' requires explicit path templates.",
    )
    parser.add_argument("--prefix", default="drvexp01", help="Prefix for generated PyBlinker pickle files.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override the preset dataset root.")
    parser.add_argument("--subjects", default=None, help="Comma-separated ordered subject IDs.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit processing to the first N subjects.")
    parser.add_argument(
        "--py-raw-template",
        default=None,
        help="Path template for the raw file used by PyBlinker. Use '{id}' as the subject placeholder.",
    )
    parser.add_argument(
        "--comparison-raw-template",
        default=None,
        help="Path template for the raw file used to extract the comparison signal.",
    )
    parser.add_argument(
        "--blinker-results-template",
        default=None,
        help="Path template for the MATLAB blinker_results.pkl file.",
    )
    parser.add_argument(
        "--output-dir-template",
        default=None,
        help="Path template for where the fresh PyBlinker pickle should be saved.",
    )
    parser.add_argument("--pick-types", default=None, help="Comma-separated MNE pick types to enable.")
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=DEFAULT_TOLERANCE_SAMPLES,
        help="Tolerance window used for comparison metrics.",
    )
    parser.add_argument("--plot", action="store_true", help="Display EEG with blink annotations.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel worker threads. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--target-share-percent",
        type=float,
        default=DEFAULT_TARGET_SHARE_PERCENT,
        help="Required share_within_tolerance_percent for reusing an existing pickle.",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing prefixed pickle files.")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue past the first non-100 subject instead of stopping.",
    )
    parser.add_argument(
        "--restrict-py-to-comparison-channels",
        action="store_true",
        help=(
            "Restrict the PyBlinker raw to only channels also present in the comparison raw, "
            "preserve comparison channel order, and align FIF samples to EDF quantization when needed."
        ),
    )
    return parser.parse_args(argv)


def _resolve_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    if args.dataset == "driving_dataset":
        config = DRIVING_DATASET_CONFIG
    else:
        if not all(
            (
                args.dataset_root,
                args.subjects,
                args.py_raw_template,
                args.blinker_results_template,
                args.output_dir_template,
            )
        ):
            raise ValueError(
                "Custom datasets require --dataset-root, --subjects, --py-raw-template, "
                "--blinker-results-template, and --output-dir-template."
            )
        config = DatasetConfig(
            name="custom",
            dataset_root=args.dataset_root,
            subject_ids=tuple(),
            py_raw_template=args.py_raw_template,
            comparison_raw_template=args.comparison_raw_template or args.py_raw_template,
            blinker_results_template=args.blinker_results_template,
            output_dir_template=args.output_dir_template,
            pick_types_options=_parse_pick_types(args.pick_types) or {"eeg": True},
        )

    dataset_root = args.dataset_root or config.dataset_root
    return DatasetConfig(
        name=config.name,
        dataset_root=dataset_root,
        subject_ids=config.subject_ids,
        py_raw_template=args.py_raw_template or config.py_raw_template,
        comparison_raw_template=args.comparison_raw_template or config.comparison_raw_template,
        blinker_results_template=args.blinker_results_template or config.blinker_results_template,
        output_dir_template=args.output_dir_template or config.output_dir_template,
        pick_types_options=_parse_pick_types(args.pick_types) or config.pick_types_options,
    )


def _parse_pick_types(pick_types: str | None) -> dict[str, bool]:
    if not pick_types:
        return {}
    return {item.strip(): True for item in pick_types.split(",") if item.strip()}


def _resolve_subject_ids(args: argparse.Namespace, config: DatasetConfig) -> list[str]:
    if args.subjects:
        subject_ids = [item.strip() for item in args.subjects.split(",") if item.strip()]
    else:
        subject_ids = list(config.subject_ids)

    if args.max_subjects is not None:
        if args.max_subjects < 1:
            return []
        subject_ids = subject_ids[: args.max_subjects]

    return subject_ids


def _render_subject_path(dataset_root: Path, template: str, subject_id: str) -> Path:
    candidate = Path(template.format(id=subject_id))
    return candidate if candidate.is_absolute() else dataset_root / candidate


def _load_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if suffix == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported raw file type for comparison: {path}")


def _plot_with_annotations(raw: mne.io.BaseRaw, annotations: mne.Annotations, channel: str) -> None:
    raw_plot = raw.copy()
    raw_plot.set_annotations(annotations)
    print(f"Displaying EEG with blink detections (based on channel {channel})...")
    raw_plot.plot(block=True, title=f"Eye close based on channel {channel}")


def _plot_from_saved_payload(subject_id: str, py_raw_path: Path, py_path: Path) -> None:
    del subject_id
    payload = load_pickle(py_path)
    raw = _load_raw(py_raw_path)
    annotations = create_annotation(payload["events"].copy(), float(raw.info["sfreq"]), "eye_blink")
    _plot_with_annotations(raw, annotations, str(payload["metrics"]["channel"]))


def _fresh_py_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f"{prefix}_pyblinker_results.pkl"


def _shared_channels_in_comparison_order(
    py_raw: mne.io.BaseRaw,
    comparison_raw_path: Path,
) -> list[str]:
    comparison_raw = _load_raw(comparison_raw_path)
    py_lookup = {name.casefold(): name for name in py_raw.ch_names}
    shared_channels = [
        py_lookup[name.casefold()]
        for name in comparison_raw.ch_names
        if name.casefold() in py_lookup
    ]
    if not shared_channels:
        raise ValueError(
            "No shared channels were found between "
            f"{comparison_raw_path.name} and the PyBlinker raw."
        )
    return shared_channels


def _align_fif_to_edf_parity(py_raw: mne.io.BaseRaw, comparison_raw_path: Path) -> None:
    comparison_raw = mne.io.read_raw_edf(comparison_raw_path, preload=False, verbose="ERROR")
    edf_extra = comparison_raw._raw_extras[0]
    comparison_idx = {name.casefold(): idx for idx, name in enumerate(comparison_raw.ch_names)}

    for py_idx, py_name in enumerate(py_raw.ch_names):
        edf_idx = comparison_idx[py_name.casefold()]
        cal = float(edf_extra["cal"][edf_idx])
        offset = float(edf_extra["offsets"][edf_idx])
        unit_scale = float(edf_extra["units"][edf_idx])
        signal_phys = py_raw._data[py_idx] / unit_scale
        digital = np.round((signal_phys - offset) / cal)
        py_raw._data[py_idx] = (digital * cal + offset) * unit_scale

    target_n_times = int(comparison_raw.n_times)
    current_n_times = int(py_raw.n_times)
    if target_n_times > current_n_times:
        pad_width = target_n_times - current_n_times
        tail = np.repeat(py_raw._data[:, -1:], pad_width, axis=1)
        py_raw._data = np.concatenate([py_raw._data, tail], axis=1)
        py_raw._last_samps[0] = py_raw._first_samps[0] + py_raw._data.shape[1] - 1
    elif target_n_times < current_n_times:
        py_raw.crop(tmax=(target_n_times - 1) / float(py_raw.info["sfreq"]))


def _run_detector(
    py_raw_path: Path,
    *,
    comparison_raw_path: Path,
    pick_types_options: dict[str, bool],
    restrict_py_to_comparison_channels: bool,
) -> tuple[dict, mne.Annotations, str]:
    raw = _load_raw(py_raw_path)
    if restrict_py_to_comparison_channels:
        shared_channels = _shared_channels_in_comparison_order(raw, comparison_raw_path)
        raw.pick(shared_channels)
        if py_raw_path.suffix.lower() == ".fif" and comparison_raw_path.suffix.lower() == ".edf":
            _align_fif_to_edf_parity(raw, comparison_raw_path)

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
        pick_types_options=pick_types_options,
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
    subject_id: str,
    *,
    py_raw_path: Path,
    comparison_raw_path: Path,
    output_dir: Path,
    prefix: str,
    plot: bool,
    pick_types_options: dict[str, bool],
    restrict_py_to_comparison_channels: bool,
) -> Path:
    payload, annotations, channel = _run_detector(
        py_raw_path,
        comparison_raw_path=comparison_raw_path,
        pick_types_options=pick_types_options,
        restrict_py_to_comparison_channels=restrict_py_to_comparison_channels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _fresh_py_path(output_dir, prefix)
    payload["metrics"]["result_file"] = output_path.name
    payload["metrics"]["subject_id"] = subject_id
    payload["metrics"]["source_raw_file"] = py_raw_path.name

    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)

    if plot:
        raw = _load_raw(py_raw_path)
        _plot_with_annotations(raw, annotations, channel)

    return output_path


def _compare_subject(
    subject_id: str,
    *,
    py_path: Path,
    blinker_path: Path,
    comparison_raw_path: Path,
    tolerance_samples: int,
) -> RecordingComparison:
    del tolerance_samples
    py_payload = load_pickle(py_path)
    blinker_payload = load_pickle(blinker_path)

    raw = _load_raw(comparison_raw_path)
    sfreq = float(raw.info["sfreq"])
    recording_duration = float(raw.times[-1])

    py_events, blinker_events, metrics = compare_events(
        py_payload, blinker_payload, sfreq, recording_duration
    )

    return RecordingComparison(
        recording_id=subject_id,
        py_events=py_events,
        blinker_events=blinker_events,
        metrics=metrics,
    )


def _share_is_good(value: float | None, target: float) -> bool:
    if value is None or not math.isfinite(float(value)):
        return False
    return math.isclose(float(value), float(target), rel_tol=0.0, abs_tol=1e-9)


def process_subject(
    subject_id: str,
    *,
    config: DatasetConfig,
    prefix: str,
    tolerance_samples: int,
    plot: bool,
    target_share_percent: float,
    force_rerun: bool,
    restrict_py_to_comparison_channels: bool,
) -> SubjectRunResult:
    py_raw_path = _render_subject_path(config.dataset_root, config.py_raw_template, subject_id)
    comparison_raw_path = _render_subject_path(
        config.dataset_root,
        config.comparison_raw_template,
        subject_id,
    )
    blinker_path = _render_subject_path(
        config.dataset_root,
        config.blinker_results_template,
        subject_id,
    )
    output_dir = _render_subject_path(
        config.dataset_root,
        config.output_dir_template,
        subject_id,
    )
    py_path = _fresh_py_path(output_dir, prefix)

    if force_rerun and py_path.exists():
        py_path.unlink()

    if py_path.exists():
        try:
            comparison = _compare_subject(
                subject_id,
                py_path=py_path,
                blinker_path=blinker_path,
                comparison_raw_path=comparison_raw_path,
                tolerance_samples=tolerance_samples,
            )
            share = comparison.metrics.get("share_within_tolerance_percent")
            if _share_is_good(share, target_share_percent):
                if plot:
                    _plot_from_saved_payload(subject_id, py_raw_path, py_path)
                return SubjectRunResult(
                    subject_id=subject_id,
                    comparison=comparison,
                    py_path=py_path,
                    artifact_status="reused_good",
                    py_raw_path=py_raw_path,
                    comparison_raw_path=comparison_raw_path,
                )
            py_path.unlink(missing_ok=True)
            artifact_status = "rerun_after_poor"
        except Exception:
            py_path.unlink(missing_ok=True)
            artifact_status = "rerun_after_error"
    else:
        artifact_status = "new"

    py_path = run_fresh_detection(
        subject_id,
        py_raw_path=py_raw_path,
        comparison_raw_path=comparison_raw_path,
        output_dir=output_dir,
        prefix=prefix,
        plot=plot,
        pick_types_options=config.pick_types_options,
        restrict_py_to_comparison_channels=restrict_py_to_comparison_channels,
    )
    comparison = _compare_subject(
        subject_id,
        py_path=py_path,
        blinker_path=blinker_path,
        comparison_raw_path=comparison_raw_path,
        tolerance_samples=tolerance_samples,
    )
    share = comparison.metrics.get("share_within_tolerance_percent")
    if not _share_is_good(share, target_share_percent):
        artifact_status = f"{artifact_status}_still_poor"

    return SubjectRunResult(
        subject_id=subject_id,
        comparison=comparison,
        py_path=py_path,
        artifact_status=artifact_status,
        py_raw_path=py_raw_path,
        comparison_raw_path=comparison_raw_path,
    )


def _order_summary(summary: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    if summary.empty:
        return summary

    order = {subject_id: idx for idx, subject_id in enumerate(subject_ids)}
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
    dataset_name: str,
    subject_ids: list[str],
) -> tuple[Path, Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_label = f"{dataset_name}_{len(subject_ids)}subjects"
    summary_path = RESULTS_DIR / f"{prefix}_{run_label}_summary.csv"
    overall_path = RESULTS_DIR / f"{prefix}_{run_label}_overall.json"
    subject_path = RESULTS_DIR / f"{prefix}_{run_label}_subjects.csv"

    summary.to_csv(summary_path, index=False)
    overall_dict = json.loads(overall.to_json()) if not overall.empty else {}
    overall_path.write_text(json.dumps(overall_dict, indent=2), encoding="utf8")
    pd.DataFrame(
        {
            "dataset": dataset_name,
            "subject_order": list(range(1, len(subject_ids) + 1)),
            "subject_id": subject_ids,
        }
    ).to_csv(subject_path, index=False)

    return summary_path, overall_path, subject_path


def _print_subject_result(result: SubjectRunResult) -> None:
    share = result.comparison.metrics.get("share_within_tolerance_percent")
    print(
        f"[fresh-run] {result.subject_id}: {result.artifact_status}, "
        f"{result.py_path.name}, share_within_tolerance_percent={share}"
    )


def _resolve_max_workers(subject_count: int, explicit_max_workers: int | None) -> int:
    if explicit_max_workers is not None:
        return max(1, explicit_max_workers)
    return max(1, min(os.cpu_count() or 1, subject_count))


def _run_subjects(
    subject_ids: list[str],
    *,
    config: DatasetConfig,
    prefix: str,
    tolerance_samples: int,
    plot: bool,
    target_share_percent: float,
    force_rerun: bool,
    continue_on_failure: bool,
    restrict_py_to_comparison_channels: bool,
    max_workers: int = 1,
) -> list[SubjectRunResult]:
    if max_workers == 1 or plot:
        results: list[SubjectRunResult] = []
        for subject_id in subject_ids:
            result = process_subject(
                subject_id,
                config=config,
                prefix=prefix,
                tolerance_samples=tolerance_samples,
                plot=plot,
                target_share_percent=target_share_percent,
                force_rerun=force_rerun,
                restrict_py_to_comparison_channels=restrict_py_to_comparison_channels,
            )
            results.append(result)
            _print_subject_result(result)

            share = result.comparison.metrics.get("share_within_tolerance_percent")
            if not _share_is_good(share, target_share_percent) and not continue_on_failure:
                print(
                    f"[stop] {subject_id} did not reach the target "
                    f"{target_share_percent:.1f}% share. Stopping incremental run."
                )
                break
        return results

    total = len(subject_ids)
    in_progress: set[str] = set()
    completed_count = 0
    lock = threading.Lock()

    print(f"[parallel] {max_workers} workers | {total} subjects to process")

    def _worker(subject_id: str) -> SubjectRunResult:
        with lock:
            in_progress.add(subject_id)
            print(
                f"[start] {subject_id} | workers={max_workers} "
                f"in-progress={len(in_progress)}/{total}: {sorted(in_progress)}"
            )
        try:
            return process_subject(
                subject_id,
                config=config,
                prefix=prefix,
                tolerance_samples=tolerance_samples,
                plot=False,
                target_share_percent=target_share_percent,
                force_rerun=force_rerun,
                restrict_py_to_comparison_channels=restrict_py_to_comparison_channels,
            )
        finally:
            with lock:
                in_progress.discard(subject_id)

    results_by_id: dict[str, SubjectRunResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_worker, subject_id): subject_id for subject_id in subject_ids}
        for future in as_completed(future_map):
            result = future.result()
            results_by_id[result.subject_id] = result
            with lock:
                completed_count += 1
                remaining = total - completed_count
            print(
                f"[done] {result.subject_id} | completed={completed_count}/{total} remaining={remaining} "
                f"in-progress={len(in_progress)}: {sorted(in_progress)}"
            )
            _print_subject_result(result)
    return [results_by_id[subject_id] for subject_id in subject_ids if subject_id in results_by_id]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _resolve_dataset_config(args)
    subject_ids = _resolve_subject_ids(args, config)

    max_workers = _resolve_max_workers(len(subject_ids), getattr(args, "max_workers", None))
    print(
        f"[config] dataset={config.name}, subjects={len(subject_ids)}, "
        f"prefix={args.prefix}, plot={args.plot}, force_rerun={args.force_rerun}, "
        f"max_workers={max_workers}, continue_on_failure={args.continue_on_failure}, "
        f"restrict_py_to_comparison_channels={args.restrict_py_to_comparison_channels}"
    )
    print(
        f"[config] py_raw_template={config.py_raw_template}, "
        f"comparison_raw_template={config.comparison_raw_template}, "
        f"pick_types={sorted(config.pick_types_options)}"
    )

    results = _run_subjects(
        subject_ids,
        config=config,
        prefix=args.prefix,
        tolerance_samples=args.tolerance_samples,
        plot=args.plot,
        target_share_percent=args.target_share_percent,
        force_rerun=args.force_rerun,
        continue_on_failure=args.continue_on_failure,
        restrict_py_to_comparison_channels=args.restrict_py_to_comparison_channels,
        max_workers=max_workers,
    )

    processed_ids = [result.subject_id for result in results]
    summary = _ensure_summary_frame(
        _order_summary(build_summary_frame([result.comparison for result in results]), processed_ids)
    )
    summary["artifact_status"] = summary["recording_id"].map(
        {result.subject_id: result.artifact_status for result in results}
    )
    summary["result_file"] = summary["recording_id"].map(
        {result.subject_id: result.py_path.name for result in results}
    )
    summary["dataset"] = config.name
    summary["py_raw_path"] = summary["recording_id"].map(
        {result.subject_id: str(result.py_raw_path) for result in results}
    )
    summary["comparison_raw_path"] = summary["recording_id"].map(
        {result.subject_id: str(result.comparison_raw_path) for result in results}
    )

    overall = build_overall_summary(summary)
    summary_path, overall_path, subject_path = _write_experiment_outputs(
        summary=summary,
        overall=overall,
        prefix=args.prefix,
        dataset_name=config.name,
        subject_ids=processed_ids,
    )

    print()
    print(summary.to_csv(index=False))
    print(json.dumps(json.loads(overall.to_json()) if not overall.empty else {}, indent=2))
    print()
    print(f"[summary] CSV: {summary_path}")
    print(f"[summary] JSON: {overall_path}")
    print(f"[summary] subjects CSV: {subject_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

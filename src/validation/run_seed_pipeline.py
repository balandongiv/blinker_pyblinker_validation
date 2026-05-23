"""Run the MATLAB Blinker, PyBlinker, and comparison for the SEED dataset."""

from __future__ import annotations

try:
    import matlab.engine
except ImportError:
    pass

import argparse
import json
import logging
import os
import pickle
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import mne
import pandas as pd
from pyblinker.blinker.pyblinker import BlinkDetector
from pyblinker.utils.annotation_utils import create_annotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation._paths import REPORTS_DIR
from src.validation.blink_compare import process_recording_comparison
from src.validation.blinker_params import build_experiment_blink_params
from src.validation.stat import build_overall_summary, build_summary_frame

try:
    from src.matlab_runner.execute_blinker import (
        BLINKER_KEYS,
        DEFAULT_PROJECT_ROOT,
        run_blinker as matlab_run_blinker,
        start_matlab as matlab_start_matlab,
    )
except ModuleNotFoundError:
    from src.matlab_runner.execute_blinker import (
        BLINKER_KEYS,
        DEFAULT_PROJECT_ROOT,
        run_blinker as matlab_run_blinker,
        start_matlab as matlab_start_matlab,
    )
from src.utils.blink_events import prepare_blinker_frame
from src.utils.config_utils import (
    DEFAULT_CONFIG_PATH,
    get_default_blinker_plugin,
    get_path_setting,
    load_config,
)

CONFIG = load_config(DEFAULT_CONFIG_PATH)
DEFAULT_EEGLAB_ROOT = get_path_setting(CONFIG, "eeglab_root", env_var="EEGLAB_ROOT")
DEFAULT_BLINKER_PLUGIN = get_default_blinker_plugin(CONFIG) or "Blinker1.2.0"
LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full SEED dataset validation.")
    parser.add_argument("--prefix", default="seed_exp01", help="Experiment prefix.")
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        type=Path,
        default=[
            Path(r"D:\dataset\SEED_VLA_VRW\VLA_VRW\real\EEG"),
            Path(r"D:\dataset\SEED_VLA_VRW\VLA_VRW\lab\EEG"),
        ],
        help="Directories containing the SEED EDF files.",
    )
    parser.add_argument(
        "--eeglab-root",
        type=Path,
        default=DEFAULT_EEGLAB_ROOT,
        help="Path to the EEGLAB installation.",
    )
    parser.add_argument(
        "--blinker-plugin",
        type=str,
        default=DEFAULT_BLINKER_PLUGIN,
        help="Name of the Blinker plugin folder inside EEGLAB\'s plugins directory.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Path containing MATLAB helpers to add to the MATLAB path.",
    )
    parser.add_argument(
        "--tolerance-samples",
        type=int,
        default=20,
        help="Tolerance window used for comparison metrics.",
    )
    parser.add_argument("--heartbeat-seconds", type=int, default=10, help="Status heartbeat interval.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(6, os.cpu_count() or 1)),
        help="Maximum worker threads for PyBlinker.",
    )
    parser.add_argument("--force-matlab", action="store_true", help="Regenerate MATLAB outputs.")
    parser.add_argument("--force-rerun", action="store_true", help="Regenerate PyBlinker outputs.")
    return parser.parse_args()


def get_matlab_result_path(edf_path: Path) -> Path:
    return edf_path.parent / f"blinker_results_{edf_path.stem}.pkl"

def get_pyblinker_result_path(edf_path: Path, prefix: str) -> Path:
    return edf_path.parent / f"{prefix}_pyblinker_results_{edf_path.stem}.pkl"

def run_matlab_blinker_if_needed(edf_files: list[Path], args: argparse.Namespace) -> None:
    missing_edfs = [
        edf for edf in edf_files
        if args.force_matlab or not get_matlab_result_path(edf).exists()
    ]
    if not missing_edfs:
        LOGGER.info("All MATLAB Blinker results exist. Skipping MATLAB run.")
        return

    LOGGER.info("Starting MATLAB to process %d EDF files...", len(missing_edfs))
    eng = matlab_start_matlab(
        args.eeglab_root,
        project_root=args.project_root,
        blinker_plugin=args.blinker_plugin,
    )
    try:
        for edf_path in missing_edfs:
            target = get_matlab_result_path(edf_path)
            LOGGER.info("Running MATLAB Blinker on %s", edf_path)
            frames = {}
            try:
                output = matlab_run_blinker(eng, edf_path)
                for key in BLINKER_KEYS:
                    frames[key] = prepare_blinker_frame(output.get(key, pd.DataFrame()))
            except Exception as exc:
                LOGGER.error("Failed MATLAB run on %s: %s", edf_path, exc)
                continue

            payload = {"frames": frames, "params": {}}
            events = frames.get("blinkFits")
            if events is not None and not events.empty:
                columns = {col.lower(): col for col in events.columns}
                if "latency" in columns:
                    payload["events_onset_sec"] = pd.to_numeric(events[columns["latency"]], errors="coerce").tolist()
                if "duration" in columns:
                    payload["events_duration_sec"] = pd.to_numeric(events[columns["duration"]], errors="coerce").tolist()

            serialisable = {key: frame.reset_index(drop=True) for key, frame in frames.items()}
            payload["frames"] = serialisable
            with target.open("wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            LOGGER.info("Saved %s", target)
    finally:
        eng.quit()


def _run_pyblinker(edf_path: Path, target_path: Path) -> Path:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    sampling_rate = float(raw.info["sfreq"])
    blink_params = build_experiment_blink_params(
        amplitude_gate_tolerance=5e-8,
        amplitude_gate_end_window_seconds=30.0,
    )

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
            "result_file": target_path.name,
        },
        "selected_channel": selected.copy(),
        "params": {
            "blink_params": blink_params,
        },
    }

    with target_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return target_path


def process_recording(edf_path: Path, args: argparse.Namespace) -> object:
    recording_id = f"{edf_path.parents[1].name}_{edf_path.stem}"
    matlab_path = get_matlab_result_path(edf_path)
    py_path = get_pyblinker_result_path(edf_path, args.prefix)

    if not matlab_path.exists():
        raise FileNotFoundError(f"Missing MATLAB output: {matlab_path}")

    if args.force_rerun or not py_path.exists():
        _run_pyblinker(edf_path, py_path)
        artifact_status = "new"
    else:
        artifact_status = "reused"

    comparison = process_recording_comparison(
        recording_dir=edf_path.parent,
        py_path=py_path,
        blinker_path=matlab_path,
        fif_path=edf_path,
        fif_fname=edf_path.name,
        tolerance_samples=args.tolerance_samples,
        overwrite=True,
    )
    comparison.recording_id = recording_id  # override to just the stem
    return comparison, artifact_status


def _write_status(
    *,
    prefix: str,
    started_at: str,
    total: int,
    completed: list[dict],
    in_progress: list[str],
    failed: list[dict],
    log_path: Path,
    status_json_path: Path,
    status_md_path: Path,
    finished: bool,
):
    payload = {
        "prefix": prefix,
        "dataset": "SEED_VLA_VRW",
        "recording_count": total,
        "started_at": started_at,
        "last_heartbeat": _utc_now(),
        "finished": finished,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "in_progress_count": len(in_progress),
        "completed": completed,
        "in_progress": in_progress,
        "failed": failed,
        "log_path": str(log_path),
    }
    status_json_path.write_text(json.dumps(payload, indent=2), encoding="utf8")

    heading = f"# {prefix} SEED Live Status"
    lines = [
        heading,
        "",
        f"- started_at: {started_at}",
        f"- last_heartbeat: {payload['last_heartbeat']}",
        f"- finished: {finished}",
        f"- total: {total}",
        f"- completed: {len(completed)}",
        f"- in_progress: {len(in_progress)}",
        f"- failed: {len(failed)}",
        f"- log: {log_path}",
        "",
        "## In Progress",
    ]
    if in_progress:
        lines.extend(f"- {name}" for name in in_progress)
    else:
        lines.append("- none")
    lines.extend(["", "## Recent Completed"])
    if completed:
        lines.extend(
            f"- {entry['recording_id']}: share={entry['share_within_tolerance_percent']}, status={entry['artifact_status']}"
            for entry in completed[-10:]
        )
    else:
        lines.append("- none")
    lines.extend(["", "## Failed"])
    if failed:
        lines.extend(f"- {entry['recording_id']}: {entry['error']}" for entry in failed[-10:])
    else:
        lines.append("- none")
    status_md_path.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> int:
    args = parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / f"{args.prefix}_seed_live_log.txt"
    status_json_path = REPORTS_DIR / f"{args.prefix}_seed_live_status.json"
    status_md_path = REPORTS_DIR / f"{args.prefix}_seed_live_status.md"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w", encoding="utf8"), logging.StreamHandler(sys.stdout)],
    )

    edf_files = []
    for root in args.dataset_roots:
        if root.exists():
            edf_files.extend(list(root.rglob("*.edf")))
    
    if not edf_files:
        LOGGER.error("No EDF files found in the specified dataset roots.")
        return 1

    LOGGER.info("Discovered %d EDF files.", len(edf_files))

    run_matlab_blinker_if_needed(edf_files, args)

    started_at = _utc_now()
    completed = []
    failed = []
    results = []
    future_to_edf = {}

    LOGGER.info("Starting PyBlinker processing with %d max workers...", args.max_workers)
    _write_status(
        prefix=args.prefix, started_at=started_at, total=len(edf_files),
        completed=completed, in_progress=[], failed=failed,
        log_path=log_path, status_json_path=status_json_path, status_md_path=status_md_path, finished=False
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for edf in edf_files:
            future = executor.submit(process_recording, edf, args)
            future_to_edf[future] = edf

        pending_futures = set(future_to_edf)
        while pending_futures:
            done, pending_futures = wait(
                pending_futures, timeout=args.heartbeat_seconds, return_when=FIRST_COMPLETED
            )
            in_progress = [f"{future_to_edf[f].parents[1].name}_{future_to_edf[f].stem}" for f in pending_futures]

            for future in done:
                edf = future_to_edf[future]
                recording_id = f"{edf.parents[1].name}_{edf.stem}"
                try:
                    comparison, artifact_status = future.result()
                    results.append(comparison)
                    share = comparison.metrics.get("share_within_tolerance_percent")
                    completed.append({
                        "recording_id": recording_id,
                        "share_within_tolerance_percent": float(share) if share is not None else None,
                        "artifact_status": artifact_status,
                    })
                    LOGGER.info("Completed %s: share=%s", recording_id, share)
                except Exception as exc:
                    failed.append({"recording_id": recording_id, "error": repr(exc)})
                    LOGGER.exception("Recording failed: %s", recording_id)

            _write_status(
                prefix=args.prefix, started_at=started_at, total=len(edf_files),
                completed=completed, in_progress=in_progress, failed=failed,
                log_path=log_path, status_json_path=status_json_path, status_md_path=status_md_path, finished=False
            )

    summary = build_summary_frame(results) if results else pd.DataFrame()
    if not summary.empty:
        summary_path = REPORTS_DIR / f"{args.prefix}_seed_summary.csv"
        overall_path = REPORTS_DIR / f"{args.prefix}_seed_overall.json"
        
        summary.to_csv(summary_path, index=False)
        overall = build_overall_summary(summary)
        overall_dict = json.loads(overall.to_json()) if not overall.empty else {}
        overall_path.write_text(json.dumps(overall_dict, indent=2), encoding="utf8")
        LOGGER.info("Saved summary CSV: %s", summary_path)
        LOGGER.info("Saved overall JSON: %s", overall_path)

    _write_status(
        prefix=args.prefix, started_at=started_at, total=len(edf_files),
        completed=completed, in_progress=[], failed=failed,
        log_path=log_path, status_json_path=status_json_path, status_md_path=status_md_path, finished=True
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
